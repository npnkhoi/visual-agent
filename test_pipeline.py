import os
import json
import torch
import cv2
import requests
import gc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from scipy.ndimage import maximum_filter

from transformers import (
    pipeline,
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

# Adjust these imports based on your separate SAM2 installation
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("[!] SAM 2 not found. Ensure it is installed in your environment.")


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
    for d in dataset:
        filename = d["image_url"].split("/")[-1].split("?")[0]
        save_path = dir_path / filename
        d["save_path"] = str(save_path)
        if not save_path.exists():
            try:
                resp = requests.get(d["image_url"], timeout=10)
                if resp.status_code == 200:
                    with open(save_path, "wb") as f:
                        f.write(resp.content)
            except Exception:
                pass
        if save_path.exists():
            validated.append(d)
    return validated


class VRAMOptimizedPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16  # Use FP16 for large models to save VRAM
        print(f"Initializing models in RAM (CPU)...")

        # Create output directory for annotated images
        self.output_dir = Path("output_images")
        self.output_dir.mkdir(exist_ok=True)

        # Stage 1: SLM
        self.slm_id = "meta-llama/Llama-3.2-8B-Instruct"
        self.slm_tokenizer = AutoTokenizer.from_pretrained(self.slm_id)
        self.slm_model = AutoModelForCausalLM.from_pretrained(
            self.slm_id, torch_dtype=self.dtype, device_map="cpu"
        )

        # Stage 2: GroundingDINO (Float32 to prevent internal type mismatch bugs)
        self.gd_id = "IDEA-Research/grounding-dino-tiny"
        self.gd_processor = AutoProcessor.from_pretrained(self.gd_id)
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.gd_id, torch_dtype=torch.float32
        ).to("cpu")

        # Stage 3: DINOv2
        self.dino_id = "facebook/dinov2-base"
        self.dinov2_processor = AutoImageProcessor.from_pretrained(self.dino_id)
        self.dinov2_model = AutoModel.from_pretrained(
            self.dino_id, torch_dtype=self.dtype
        ).to("cpu")

        # Stage 5: CLIP
        self.clip_id = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_id)
        self.clip_model = CLIPModel.from_pretrained(
            self.clip_id, torch_dtype=self.dtype
        ).to("cpu")

    def _swap_to_gpu(self, model):
        """Moves model to GPU."""
        model.to(self.device)

    def _swap_to_cpu(self, model):
        """Moves model to CPU and flushes VRAM."""
        model.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

    def stage_1_decompose(self, query: str) -> dict:
        self._swap_to_gpu(self.slm_model)

        # Using your specific query structure, modified to request JSON
        prompt = (
            f"Analyze the following query to extract linguistic descriptors for a countable subject: '{query}'\n\n"
            "1. Subject: Identify the primary noun or noun phrase representing the entity to be counted.\n"
            "2. Adjectives: List all attributive and predicative adjectives that modify the subject.\n"
            "3. Verbs/Predicates: Identify the specific verbs or verb phrases that describe the state, action, or condition of the subject (the predicate).\n"
            "Output your findings ONLY as a JSON object with keys: "
            "'target_noun', 'attributes', and 'interactions'"
        )

        inputs = self.slm_tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.slm_model.generate(
                **inputs,
                max_new_tokens=150,  # Increased to handle the longer analysis
                pad_token_id=self.slm_tokenizer.eos_token_id
            )

        self._swap_to_cpu(self.slm_model)

        # Decode only the newly generated tokens
        input_length = inputs['input_ids'].shape[1]
        decoded_text = self.slm_tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

        print(f"  [SLM Analysis]: {decoded_text}")

        try:
            # Robust JSON extraction
            start_idx = decoded_text.find('{')
            end_idx = decoded_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = decoded_text[start_idx:end_idx]
                parsed = json.loads(json_str)

                # Standardize keys for the rest of the pipeline
                return {
                    "target_noun": parsed.get("target_noun", "object"),
                    "adjectives": parsed.get("adjectives", []),
                    "verbs": parsed.get("verbs", [])
                }
            else:
                raise ValueError("JSON marks not found")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [!] Analysis parsing failed: {e}. Using raw query as noun.")
            # Intelligent fallback: If it's not JSON, assume it's the 'short phrase' requested
            return {
                "target_noun": query.split()[-1].strip("."),  # Simple fallback logic
                "adjectives": [],
                "verbs": []
            }

    def stage_2_exemplar_mining(self, image: Image.Image, noun: str, threshold=0.55) -> List[Tuple]:
        self._swap_to_gpu(self.gd_model)
        inputs = self.gd_processor(images=image, text=noun, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.gd_model(**inputs)

        # ==========================================
        # FOOLPROOF MANUAL PARSING (Version Independent)
        # ==========================================
        # 1. Get raw logits and boxes from the model output
        logits = outputs.logits[0].sigmoid()  # Shape: (num_queries, num_classes)
        boxes = outputs.pred_boxes[0]  # Shape: (num_queries, 4) - format: [cx, cy, w, h]

        # 2. Filter by threshold
        scores = logits.max(dim=-1)[0]
        keep = scores > threshold
        valid_boxes = boxes[keep]

        # 3. Convert from normalized [cx, cy, w, h] to absolute [xmin, ymin, xmax, ymax]
        cx, cy, w, h = valid_boxes.unbind(-1)
        xmin = (cx - 0.5 * w) * image.width
        ymin = (cy - 0.5 * h) * image.height
        xmax = (cx + 0.5 * w) * image.width
        ymax = (cy + 0.5 * h) * image.height

        # 4. Stack and convert to list
        absolute_boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
        results = absolute_boxes.cpu().numpy().tolist()
        # ==========================================

        self._swap_to_cpu(self.gd_model)
        return results

    def stage_3_semantic_search(self, image: Image.Image, seed_boxes: List[Tuple]) -> np.ndarray:
        self._swap_to_gpu(self.dinov2_model)
        inputs = self.dinov2_processor(images=image, return_tensors="pt").to(self.device, dtype=self.dtype)

        with torch.no_grad():
            outputs = self.dinov2_model(**inputs)
            dense_features = outputs.last_hidden_state[:, 1:, :]
        self._swap_to_cpu(self.dinov2_model)

        # Mock heatmap generation (simulating similarity values between 0 and 1)
        h = int(np.sqrt(dense_features.shape[1]))
        sim_map = torch.rand((h, h), dtype=torch.float32).numpy()
        heatmap = cv2.resize(sim_map, image.size)
        return heatmap

    def stage_4_mask_generation(self, image: Image.Image, heatmap: np.ndarray) -> Tuple[List, List]:
        local_max = maximum_filter(heatmap, size=20) == heatmap
        peaks = (heatmap > 0.6) & local_max
        y_coords, x_coords = np.where(peaks)
        points = list(zip(x_coords, y_coords))

        masks, boxes = [], []
        for pt in points:
            b_xmin, b_ymin = max(0, pt[0] - 20), max(0, pt[1] - 20)
            b_xmax, b_ymax = min(image.size[0], pt[0] + 20), min(image.size[1], pt[1] + 20)
            masks.append(None)
            boxes.append([b_xmin, b_ymin, b_xmax, b_ymax])

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            scores_tensor = torch.ones(len(boxes_tensor))
            keep_idx = batched_nms(boxes_tensor, scores_tensor, torch.zeros(len(boxes_tensor)), iou_threshold=0.5)
            boxes = boxes_tensor[keep_idx].tolist()
            masks = [masks[i] for i in keep_idx]

        return boxes, masks

    def stage_5_attribute_filtering(self, image: Image.Image, boxes: List, semantics: dict, margin=0.7) -> List:
        if not boxes: return []
        self._swap_to_gpu(self.clip_model)

        noun = semantics["target_noun"]
        attrs = " ".join(semantics.get("attributes", []))
        verbs = " ".join(semantics.get("verbs", []))

        positive_prompt = f"A {attrs} {noun} {verbs}".strip()
        negative_prompt = f"A clean {noun} or background debris"

        filtered_boxes = []
        for box in boxes:
            crop = image.crop((box[0], box[1], box[2], box[3]))
            inputs = self.clip_processor(
                text=[positive_prompt, negative_prompt], images=crop, return_tensors="pt", padding=True
            ).to(self.device, dtype=self.dtype)

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1).cpu().float().numpy()[0]

            if probs[0] > margin * np.sum(probs):
                filtered_boxes.append(box)

        self._swap_to_cpu(self.clip_model)
        return filtered_boxes

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

    def run(self, dict_entry: dict):
        img_path = dict_entry["save_path"]
        query = dict_entry["question"]
        image = Image.open(img_path).convert("RGB")
        filename_stem = Path(img_path).stem

        # Create a specific directory for this image's intermediate steps
        img_out_dir = self.output_dir / filename_stem
        img_out_dir.mkdir(exist_ok=True)

        print(f"\nProcessing: {query} on {filename_stem}")

        # --- Stage 1 ---
        semantics = self.stage_1_decompose(query)
        with open(img_out_dir / "stage_1_semantics.json", "w") as f:
            json.dump({"query": query, "extracted": semantics}, f, indent=4)

        # --- Stage 2 ---
        seeds = self.stage_2_exemplar_mining(image, semantics["target_noun"])
        self.draw_boxes_and_save(image, seeds, img_out_dir / "stage_2_seeds.jpg", (255, 0, 0), "Seeds")  # Blue

        # --- Stage 3 ---
        heatmap = self.stage_3_semantic_search(image, seeds)
        self.draw_heatmap_and_save(image, heatmap, img_out_dir / "stage_3_heatmap.jpg")

        # --- Stage 4 ---
        boxes, masks = self.stage_4_mask_generation(image, heatmap)
        self.draw_boxes_and_save(image, boxes, img_out_dir / "stage_4_candidates.jpg", (0, 165, 255),
                                 "Candidates")  # Orange

        # --- Stage 5 & 6 ---
        final_boxes = self.stage_5_attribute_filtering(image, boxes, semantics)
        self.draw_boxes_and_save(image, final_boxes, img_out_dir / "stage_6_final.jpg", (0, 255, 0),
                                 "Final Count")  # Green

        print(f"Final Count: {len(final_boxes)}. All intermediates saved to {img_out_dir}/")
        return len(final_boxes)


if __name__ == "__main__":
    dataset = get_img_paths()
    validated_dataset = download_imgs(dataset)

    # Initialize pipeline
    pipeline_app = VRAMOptimizedPipeline()
    for data in validated_dataset:
        pipeline_app.run(data)