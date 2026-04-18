import json
import torch
import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from scipy.ndimage import maximum_filter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sys
import os

# Add the root of your SAM2 clone to the system path
sam2_root = os.path.abspath("segment-anything-2")
if sam2_root not in sys.path:
    sys.path.append(sam2_root)

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoImageProcessor,
    AutoModel,
    CLIPProcessor,
    CLIPModel,
    AutoModelForCausalLM,
    AutoTokenizer
)
from torchvision.ops import batched_nms

# Optimized for 24GB VRAM: No swapping, Batched CLIP, and Torch Compile
class VisionPipeline24GB:
    def __init__(self):
        self.device = "cuda"
        self.dtype = torch.float16  # FP16 is perfect for your 24GB card
        print(f"🚀 Initializing Pipeline on {torch.cuda.get_device_name(0)} (24GB VRAM detected)")

        self.output_dir = Path("output_images")
        self.output_dir.mkdir(exist_ok=True)

        # --- Model 1: Llama 3.1 8B (Decomposition) ---
        self.slm_id = "meta-llama/Llama-3.1-8B-Instruct"
        self.slm_tokenizer = AutoTokenizer.from_pretrained(self.slm_id)
        self.slm_model = AutoModelForCausalLM.from_pretrained(
            self.slm_id, torch_dtype=self.dtype
        ).to(self.device)

        # --- Model 2: GroundingDINO (Exemplar Mining) ---
        self.gd_id = "IDEA-Research/grounding-dino-tiny"
        self.gd_processor = AutoProcessor.from_pretrained(self.gd_id)
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.gd_id, torch_dtype=torch.float32 # GD usually requires FP32 for stability
        ).to(self.device)

        # --- Model 3: DINOv2 (Semantic Search) ---
        self.dino_id = "facebook/dinov2-base"
        self.dinov2_processor = AutoImageProcessor.from_pretrained(self.dino_id)
        self.dinov2_model = AutoModel.from_pretrained(
            self.dino_id, torch_dtype=self.dtype
        ).to(self.device)

        # --- Model 4: CLIP (Attribute Filtering) ---
        self.clip_id = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_id)
        self.clip_model = CLIPModel.from_pretrained(
            self.clip_id, torch_dtype=self.dtype
        ).to(self.device)

        # Model 5: SAM2
        self.sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "sam2.1/sam2.1_hiera_l.yaml"
        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # Optional: Speed up inference with torch.compile (PyTorch 2.0+)
        # print("⚡ Compiling models for extra speed...")
        # self.slm_model = torch.compile(self.slm_model)
        # self.dinov2_model = torch.compile(self.dinov2_model)

    @torch.no_grad()
    def stage_1_decompose(self, query: str) -> dict:
        # 1. Use a structured Chat Template for 8B models to minimize hallucinations
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise computer vision data extractor. Extract the subject of a counting query into JSON. "
                    "Rules:\n"
                    "- target_noun: Singular form of the physical object.\n"
                    "- attributes: Appearance descriptors only (color, size, material).\n"
                    "- interactions: Actions or locations.\n"
                    "- Ignore metadata like price, time, or currency.\n"
                    "- If a field is missing, return an empty list []."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Query: '{query}'\n\n"
                    "Output JSON Schema:\n"
                    "{\n"
                    "  \"reasoning\": \"briefly explain the logic\",\n"
                    "  \"target_noun\": \"singular object\",\n"
                    "  \"attributes\": [],\n"
                    "  \"interactions\": []\n"
                    "}"
                )
            }
        ]

        # 2. Apply chat template (standard for Llama-3/Qwen-2.5 8B Instruct)
        prompt = self.slm_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.slm_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 3. Generation with Greedy Decoding
        outputs = self.slm_model.generate(
            **inputs, 
            max_new_tokens=250, # Increased slightly to accommodate the 'reasoning' field
            pad_token_id=self.slm_tokenizer.eos_token_id,
            do_sample=False, 
            temperature=0.0
        )
        
        # 4. Clean Extraction
        decoded_text = self.slm_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        try:
            # Robust JSON slicing in case the model adds conversational filler
            start_idx = decoded_text.find('{')
            end_idx = decoded_text.rfind('}') + 1
            parsed = json.loads(decoded_text[start_idx:end_idx])
            
            return {
                "target_noun": parsed.get("target_noun", "object"),
                "adjectives": parsed.get("attributes", []),
                "verbs": parsed.get("interactions", []) # Matched schema key to extraction key
            }
        except Exception:
            # Simple fallback logic: extract the last word as a noun
            fallback_noun = query.strip("?.").split()[-1]
            return {"target_noun": fallback_noun, "adjectives": [], "verbs": []}

    @torch.no_grad()
    def stage_2_exemplar_mining(self, image: Image.Image, noun: str, threshold=0.30) -> List[List]:
        # Ensure the noun ends with a period (GroundingDINO convention)
        # and try to keep it singular
        text_prompt = f"{noun}." 
        
        inputs = self.gd_processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        outputs = self.gd_model(**inputs)
        
        logits = outputs.logits[0].sigmoid()
        boxes = outputs.pred_boxes[0]
        
        # Get max score across all phrases (usually just the one noun)
        scores = logits.max(dim=-1)[0]
        
        # Use a lower threshold for "seeds" to ensure we get at least 1-2
        keep = scores > threshold
        valid_boxes = boxes[keep]
        
        cx, cy, w, h = valid_boxes.unbind(-1)
        xmin = (cx - 0.5 * w) * image.width
        ymin = (cy - 0.5 * h) * image.height
        xmax = (cx + 0.5 * w) * image.width
        ymax = (cy + 0.5 * h) * image.height
        
        return torch.stack([xmin, ymin, xmax, ymax], dim=-1).cpu().numpy().tolist()

    @torch.no_grad()

    def stage_3_semantic_search(self, image: Image.Image, seed_boxes: List[List]) -> np.ndarray:
        # 1. Prepare image for DINOv2
        inputs = self.dinov2_processor(images=image, return_tensors="pt").to(self.device, dtype=self.dtype)
        outputs = self.dinov2_model(**inputs)
        
        # Extract patch tokens (excluding the CLS token at index 0)
        # Shape: [1, num_patches, embedding_dim]
        patch_embeddings = outputs.last_hidden_state[:, 1:, :] 
        
        # Calculate grid dimensions (e.g., 16x16 for 224x224 input)
        num_patches = patch_embeddings.shape[1]
        grid_size = int(np.sqrt(num_patches))
        
        # 2. Extract Seed Embeddings
        # We map the bounding boxes to the patch grid to find which tokens represent our seeds
        seed_features = []
        w, h = image.size
        for box in seed_boxes:
            xmin, ymin, xmax, ymax = box
            # Convert pixel coords to grid coords
            col_start, col_end = int(xmin * grid_size / w), int(xmax * grid_size / w)
            row_start, row_end = int(ymin * grid_size / h), int(ymax * grid_size / h)
            
            for r in range(row_start, min(row_end + 1, grid_size)):
                for c in range(col_start, min(col_end + 1, grid_size)):
                    idx = r * grid_size + c
                    seed_features.append(patch_embeddings[0, idx])

        if not seed_features:
            return np.zeros((h, w))

        # Average the seed features to create a 'target' descriptor
        target_embedding = torch.stack(seed_features).mean(dim=0, keepdim=True) # [1, dim]
        
        # 3. Compute Cosine Similarity
        # Normalize for cosine similarity
        patch_embeddings_norm = torch.nn.functional.normalize(patch_embeddings, dim=-1)
        target_embedding_norm = torch.nn.functional.normalize(target_embedding, dim=-1)
        
        # Similarity map: [1, num_patches]
        similarity = torch.matmul(patch_embeddings_norm, target_embedding_norm.T).squeeze()
        
        # Reshape and upscale to original image size
        heatmap = similarity.reshape(grid_size, grid_size).cpu().float().numpy()
        heatmap = cv2.resize(heatmap, image.size, interpolation=cv2.INTER_CUBIC)
        
        # Optional: ReLU to remove negative correlations and normalize to [0, 1]
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
            
        return heatmap
    
    def get_persistent_peaks(self, heatmap: np.ndarray, threshold: float = 0.15) -> List[Tuple[int, int]]:
        """Identifies stable local maxima using topological prominence."""
        local_max = maximum_filter(heatmap, size=15) == heatmap
        local_max = local_max & (heatmap > threshold)
        y_coords, x_coords = np.where(local_max)
        
        # Sort by intensity to process dominant peaks first
        peaks = sorted(zip(x_coords, y_coords), key=lambda p: heatmap[p[1], p[0]], reverse=True)
        
        persistent_peaks = []
        for pt in peaks:
            # Check for proximity to already accepted peaks (Suppress redundant seeds)
            if all(np.linalg.norm(np.array(pt) - np.array(prev)) > 20 for prev in persistent_peaks):
                persistent_peaks.append(pt)
        return persistent_peaks

    @torch.no_grad()
    def stage_4_amodal_refinement(self, image: Image.Image, heatmap: np.ndarray) -> List[List]:
        """Uses SAM 2 to convert points into amodal bounding boxes."""
        peaks = self.get_persistent_peaks(heatmap)
        if not peaks: return []

        img_array = np.array(image)
        self.sam2_predictor.set_image(img_array)
        
        refined_boxes = []
        input_points = np.array(peaks)
        input_labels = np.ones(len(peaks)) # 1 = Foreground

        # SAM 2 can handle batched point prompts for efficiency
        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=input_points[:, None, :],
            point_labels=input_labels[:, None],
            multimask_output=False
        )

        for i, mask in enumerate(masks):
            # The mask here is amodal; it predicts the shape behind occlusions
            # We convert the binary mask to an axis-aligned bounding box
            coords = np.argwhere(mask[0])
            if coords.size > 0:
                ymin, xmin = coords.min(axis=0)
                ymax, xmax = coords.max(axis=0)
                refined_boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
        
        return refined_boxes
        

    @torch.no_grad()
    def stage_5_attribute_filtering(self, image: Image.Image, boxes: List, semantics: dict, margin=0.6) -> List:
        if not boxes: return []
        
        # --- BATCHED OPTIMIZATION ---
        crops = [image.crop((b[0], b[1], b[2], b[3])) for b in boxes]
        pos_prompt = f"A { ' '.join(semantics['adjectives'])} {semantics['target_noun']} { ' '.join(semantics['verbs'])}".strip()
        neg_prompt = "background noise or irrelevant object"

        # Processing all crops in one GPU call
        inputs = self.clip_processor(
            text=[pos_prompt, neg_prompt], 
            images=crops, 
            return_tensors="pt", 
            padding=True
        ).to(self.device, dtype=self.dtype)

        outputs = self.clip_model(**inputs)
        # Calculate probabilities across the batch
        probs = outputs.logits_per_image.softmax(dim=1).cpu().float().numpy()

        filtered_boxes = [boxes[i] for i, p in enumerate(probs) if p[0] > margin]
        return filtered_boxes

    def run(self, dict_entry: dict):
        img_path = Path(dict_entry["save_path"])
        query = dict_entry["question"]
        
        # 0. Load Image
        image = Image.open(img_path).convert("RGB")
        filename_stem = img_path.stem

        # Create a specific directory for this image's intermediate steps
        img_out_dir = self.output_dir / filename_stem
        img_out_dir.mkdir(exist_ok=True, parents=True)

        print(f"\n🚀 Processing: {query} on {filename_stem}")

        # --- Stage 1: Decompose ---
        semantics = self.stage_1_decompose(query)
        with open(img_out_dir / "stage_1_semantics.json", "w") as f:
            json.dump({"query": query, "extracted": semantics}, f, indent=4)

        # --- Stage 2: Exemplar Mining (Seeds) ---
        seeds = self.stage_2_exemplar_mining(image, semantics["target_noun"])
        # CALL SAVING FUNCTION FOR SEEDS
        self.draw_boxes_and_save(image, seeds, img_out_dir / "stage_2_seeds.jpg", (255, 0, 0), "Seeds")

        # --- Stage 3: Semantic Search (Heatmap) ---
        heatmap = self.stage_3_semantic_search(image, seeds)
        # CALL SAVING FUNCTION FOR HEATMAP
        self.draw_heatmap_and_save(image, heatmap, img_out_dir / "stage_3_heatmap.jpg")

        # --- Stage 4: Mask Generation (Candidates) ---
        candidate_boxes = self.stage_4_amodal_refinement(image, heatmap)

        # Draw the refined amodal candidates
        self.draw_boxes_and_save(
            image, 
            candidate_boxes, 
            img_out_dir / "stage_4_amodal_candidates.jpg", 
            (0, 165, 255), 
            "Amodal Candidates"
        )

        # --- Stage 5: Attribute Filtering (Final) ---
        final_boxes = self.stage_5_attribute_filtering(image, candidate_boxes, semantics)
        
        # CALL SAVING FUNCTION FOR FINAL OUTPUT
        self.draw_boxes_and_save(image, final_boxes, img_out_dir / "stage_6_final.jpg", (0, 255, 0), "Final Count")

        print(f"✅ Final Count: {len(final_boxes)}. Files saved in: {img_out_dir}")
        return len(final_boxes)

    # ==========================================
    # Visualization Helpers
    # ==========================================
    def draw_boxes_and_save(self, image: Image.Image, boxes: List[List], path: Path, color: Tuple[int, int, int],
                            title: str) -> None:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box)
            cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), color, 2)

        cv2.putText(img_cv, f"{title}: {len(boxes)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.imwrite(str(path), img_cv)

    def draw_heatmap_and_save(self, image: Image.Image, heatmap: np.ndarray, path: Path) -> None:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Normalize heatmap to 0-255 for colormap
        heatmap_norm = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        heatmap_norm = np.uint8(heatmap_norm)

        # Apply Jet colormap
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        # Overlay heatmap onto original image (50% opacity)
        blended = cv2.addWeighted(img_cv, 0.5, heatmap_color, 0.5, 0)
        cv2.imwrite(str(path), blended)


def get_img_paths() -> List[dict]:
    path = Path("labels/CountBench_test.json")
    if not path.exists():
        return [{"image_url": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
                 "question": "Count the dogs."}]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["database"]


def download_imgs(dataset: list) -> list:
    dir_path = Path("images")
    dir_path.mkdir(exist_ok=True)
    validated = []
    
    # Add these headers to bypass basic bot protection
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for d in dataset:
        filename = d["image_url"].split("/")[-1].split("?")[0]
        save_path = dir_path / filename
        d["save_path"] = str(save_path)

        if not save_path.exists():
            try:
                # Use the headers here
                resp = requests.get(d["image_url"], headers=headers, timeout=10)
                
                # Check if we actually got an image and not an error page
                if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
                    with open(save_path, "wb") as f:
                        f.write(resp.content)
                else:
                    print(f" [!] Failed to download {filename}: Access Denied or Not an Image")
            except Exception as e:
                print(f" [!] Download error: {e}")

        if save_path.exists():
            validated.append(d)
    return validated     


if __name__ == "__main__":
    dataset = get_img_paths()
    validated_dataset = download_imgs(dataset)

    # Initialize pipeline
    pipeline_app = VisionPipeline24GB()
    for data in validated_dataset:
        pipeline_app.run(data)