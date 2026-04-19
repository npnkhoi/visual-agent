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
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir

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
    AutoTokenizer,
    AutoConfig
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
        config = AutoConfig.from_pretrained(self.dino_id)
        config.output_attentions = True
        self.dinov2_model = AutoModel.from_pretrained(
            self.dino_id, torch_dtype=self.dtype, config=config,
        ).to(self.device)

        # --- Model 4: CLIP (Attribute Filtering) ---
        self.clip_id = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_id)
        self.clip_model = CLIPModel.from_pretrained(
            self.clip_id, torch_dtype=self.dtype
        ).to(self.device)

        # Model 5: SAM2
        # 1. Define absolute paths to avoid Windows pathing confusion
        project_root = os.getcwd() # This is 'visual-agent'
        sam2_dir = os.path.join(project_root, "segment-anything-2")
        config_path = os.path.join(sam2_dir, "sam2", "configs")

        # 2. Ensure Python can find the 'sam2' module
        if sam2_dir not in sys.path:
            sys.path.append(sam2_dir)

        # 3. FORCE Hydra to look in the correct folder
        # We clear existing Hydra instances to prevent conflicts
        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=config_path, version_base="1.2")

        # 4. Set your variables
        self.sam2_checkpoint = os.path.join(sam2_dir, "checkpoints", "sam2.1_hiera_large.pt")
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

    # ============================================================
    # Stage 3 v2: Multi-scale + SAM-refined seeds + CLIP gating
    #             + edge gating + iterative seed expansion
    # ============================================================

    @torch.no_grad()
    def _refine_seeds_with_sam(self, image: Image.Image, seed_boxes: List[List]) -> List[np.ndarray]:
        """Convert loose GroundingDINO boxes into tight foreground masks."""
        if not seed_boxes:
            return []
        self.sam2_predictor.set_image(np.array(image))
        masks = []
        for b in seed_boxes:
            try:
                m, scores, _ = self.sam2_predictor.predict(
                    box=np.array(b, dtype=np.float32),
                    multimask_output=False,
                )
                masks.append(m[0].astype(bool))
            except Exception:
                continue
        return masks


    @torch.no_grad()
    def _clip_relevancy_map(self, image: Image.Image, semantics: dict, out_size: Tuple[int, int]) -> np.ndarray:
        """Coarse CLIP text-image relevancy via sliding-window crops. Cheap: ~16 crops."""
        w, h = out_size
        noun = semantics.get("target_noun", "object")
        adj = " ".join(semantics.get("adjectives", []))
        pos_prompt = f"a photo of a {adj} {noun}".strip()
        neg_prompt = "a photo of background or empty space"

        # 4x4 grid of overlapping crops
        grid = 4
        cw, ch = int(w / grid * 1.5), int(h / grid * 1.5)
        sx, sy = (w - cw) // (grid - 1), (h - ch) // (grid - 1)

        crops, centers = [], []
        for i in range(grid):
            for j in range(grid):
                x0, y0 = j * sx, i * sy
                x1, y1 = min(x0 + cw, w), min(y0 + ch, h)
                crops.append(image.crop((x0, y0, x1, y1)))
                centers.append(((x0 + x1) // 2, (y0 + y1) // 2))

        inputs = self.clip_processor(
            text=[pos_prompt, neg_prompt], images=crops,
            return_tensors="pt", padding=True
        ).to(self.device, dtype=self.dtype)
        outputs = self.clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().float().numpy()[:, 0]  # pos prob

        # Scatter probs onto a coarse map, then smooth/upsample
        coarse = np.zeros((grid, grid), dtype=np.float32)
        for idx, (cx, cy) in enumerate(centers):
            i, j = idx // grid, idx % grid
            coarse[i, j] = probs[idx]
        rel = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
        rel = cv2.GaussianBlur(rel, (51, 51), 0)
        if rel.max() > 0:
            rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-8)
        return rel


    @torch.no_grad()
    def _dino_sim_at_scale(
        self,
        image: Image.Image,
        seed_masks: List[np.ndarray],
        input_res: int,
    ) -> np.ndarray:
        """Single-scale contrastive DINOv2 similarity map using mask-interior patches."""
        inputs = self.dinov2_processor(
            images=image, size={"height": input_res, "width": input_res}, return_tensors="pt"
        ).to(self.device, dtype=self.dtype)
        outputs = self.dinov2_model(**inputs, output_attentions=True)

        feats = outputs.last_hidden_state[:, 1:, :]  # [1, N, D]
        N = feats.shape[1]
        G = int(np.sqrt(N))
        w, h = image.size

        # Build seed_mask from SAM masks (mask-interior only → pure foreground prototype)
        seed_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        for sam_mask in seed_masks:
            small = cv2.resize(sam_mask.astype(np.uint8), (G, G), interpolation=cv2.INTER_AREA)
            seed_mask |= torch.from_numpy(small > 0).flatten().to(self.device)

        if seed_mask.sum() == 0:
            return np.zeros((h, w), dtype=np.float32)

        p = torch.nn.functional.normalize(feats[0], dim=-1)
        pos = torch.nn.functional.normalize(feats[0, seed_mask], dim=-1)

        # Max-sim over all seed patches (preserves appearance diversity)
        sim_pos = (p @ pos.T).max(dim=-1).values  # [N]

        # Data-driven negative prototype from bottom-20% positively-similar patches
        k_neg = max(10, int(0.2 * N))
        neg_idx = torch.topk(sim_pos, k=k_neg, largest=False).indices
        neg = torch.nn.functional.normalize(p[neg_idx].mean(0, keepdim=True), dim=-1)
        sim_neg = (p @ neg.T).squeeze(-1)

        contrast = sim_pos - sim_neg
        contrast = (contrast - contrast.min()) / (contrast.max() - contrast.min() + 1e-8)
        sharp = torch.pow(contrast, 1.8)

        hm = sharp.reshape(G, G).cpu().float().numpy()
        hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return hm


    @torch.no_grad()
    def stage_3_semantic_search(
        self,
        image: Image.Image,
        seed_boxes: List[List],
        semantics: dict = None,
        iterations: int = 2,
    ) -> np.ndarray:
        """
        Multi-scale DINOv2 similarity with:
        - SAM-refined foreground-only seeds
        - CLIP text-image relevancy gating
        - Edge/texture gating
        - Iterative seed expansion (peaks from pass 1 become seeds for pass 2)
        """
        w, h = image.size
        if not seed_boxes:
            return np.zeros((h, w), dtype=np.float32)

        # ---- 1. Tight foreground masks from GroundingDINO boxes ----
        seed_masks = self._refine_seeds_with_sam(image, seed_boxes)
        if not seed_masks:
            # Fallback: treat each box as a coarse rectangular mask
            seed_masks = []
            for b in seed_boxes:
                m = np.zeros((h, w), dtype=bool)
                x0, y0, x1, y1 = map(int, b)
                m[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
                seed_masks.append(m)

        # ---- 2. Edge-density gate (suppresses flat regions like sky/paper) ----
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        edge_density = cv2.GaussianBlur(edges, (31, 31), 0)
        if edge_density.max() > 0:
            edge_density = (edge_density - edge_density.min()) / (edge_density.max() - edge_density.min() + 1e-8)
        edge_gate = 0.3 + 0.7 * edge_density  # never fully zero out

        # ---- 3. Optional CLIP relevancy gate ----
        if semantics is not None:
            clip_rel = self._clip_relevancy_map(image, semantics, (w, h))
            clip_gate = 0.4 + 0.6 * clip_rel
        else:
            clip_gate = np.ones((h, w), dtype=np.float32)

        def _compute(masks):
            maps = []
            for res in (392, 518, 714):       # 28x28, 37x37, 51x51 grids
                maps.append(self._dino_sim_at_scale(image, masks, res))
            hm = np.mean(maps, axis=0)
            hm = hm * edge_gate * clip_gate
            hm = cv2.GaussianBlur(hm, (11, 11), 0)
            if hm.max() > 0:
                hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
            return hm

        # ---- 4. Iterative refinement: use pass-1 peaks as extra seeds ----
        heatmap = _compute(seed_masks)

        for _ in range(max(0, iterations - 1)):
            peaks = self.get_persistent_peaks(heatmap, threshold=0.45)[:8]
            if not peaks:
                break
            extra_boxes = []
            # Size heuristic: use the median seed-mask size to build pseudo-boxes
            med_area = np.median([m.sum() for m in seed_masks]) if seed_masks else (w * h * 0.01)
            r = max(15, int(np.sqrt(med_area) * 0.6))
            for (x, y) in peaks:
                extra_boxes.append([max(0, x - r), max(0, y - r),
                                    min(w, x + r), min(h, y + r)])
            extra_masks = self._refine_seeds_with_sam(image, extra_boxes)
            if not extra_masks:
                break
            combined_masks = seed_masks + extra_masks
            new_hm = _compute(combined_masks)
            # Blend: trust old 40%, new 60% (new has more evidence)
            heatmap = 0.4 * heatmap + 0.6 * new_hm
            seed_masks = combined_masks

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
        """Uses Watershed and SAM 2 to convert heatmap clusters into discrete bounding boxes."""
        
        # 1. Binarize the heatmap
        # Using a threshold to create a mask of likely object areas
        thresh = (heatmap > 0.15).astype(np.uint8) * 255
        
        # 2. Distance Transform & Watershed (To separate overlapping spoons)
        # This creates 'markers' for SAM 2 based on the centers of heatmap clusters
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        ret, last_markers = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        last_markers = np.uint8(last_markers)
        
        # Find individual peaks to use as SAM 2 points
        num_labels, labels = cv2.connectedComponents(last_markers)
        
        refined_boxes = []
        img_array = np.array(image)
        self.sam2_predictor.set_image(img_array)

        # 3. Iterate through each separated component
        for i in range(1, num_labels): # Label 0 is background
            component_mask = (labels == i).astype(np.uint8)
            y, x = np.where(component_mask)
            
            if len(x) == 0: continue
            
            # Use the centroid of the watershed peak as the SAM prompt point
            cx, cy = int(np.mean(x)), int(np.mean(y))
            
            input_points = np.array([[cx, cy]])
            input_labels = np.array([1]) # Foreground

            # SAM 2 refinement
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False
            )

            # Convert resulting mask to Bbox
            mask = masks[0]
            coords = np.argwhere(mask)
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
        heatmap = self.stage_3_semantic_search(image, seeds, semantics=semantics, iterations=2)
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
