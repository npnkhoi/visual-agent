#!/usr/bin/env python3
"""
run_omnicount_pipeline.py

OmniCount-style pipeline runner with strict VRAM management.
- Requires NO arguments to run (uses local directory defaults).
- Automatically generates the default CountBench JSON if missing.
- Strictly flushes VRAM between memory-heavy stages.
"""

import os
import re
import sys
import json
import argparse
import subprocess
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from scipy import ndimage
from skimage.feature import peak_local_max
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

# ---------- Constants ----------
BICUBIC = InterpolationMode.BICUBIC

DEFAULT_JSON_DATA = {
  "database": [
    {
      "question_id": 2,
      "image_url": "http://img0.etsystatic.com/000/0/5304297/il_fullxfull.191730668.jpg",
      "question": "City prints: Set of [MASK] big prints - $150.00 USD",
      "answer": 3
    },
    {
      "question_id": 3,
      "image_url": "https://cdn.vectorstock.com/i/thumb-large/74/37/4657437.jpg",
      "question": "Set of [MASK] arrows in all directions vector",
      "answer": 8
    },
    {
      "question_id": 4,
      "image_url": "https://laurelleaffarm.com/item-photos/vintage-silver-plate-tablespoons-serving-spoon-set-of-six-1847-Rogers-Ambassador-pattern-Laurel-Leaf-Farm-item-no-pw515216-1.jpg",
      "question": "Vintage silver plate tablespoons, serving spoon set of [MASK] 1847 Rogers pattern",
      "answer": 6
    },
    {
      "question_id": 5,
      "image_url": "https://us.123rf.com/450wm/pingpao/pingpao1505/pingpao150500257/40608684-vintage-color-filltered-of-two-cute-little-girls-having-fun-blowing-bubbles-on-beach-in-summer-time.jpg?ver=6",
      "question": "Kids playing beach: Vintage color filltered of [MASK] cute little girls having fun blowing bubbles on beach in summer time Stock Photo",
      "answer": 2
    },
    {
      "question_id": 7,
      "image_url": "https://natashalh.com/wp-content/uploads/2021/05/free-fourth-of-july-dauber-marker-printables-1024x683.webp",
      "question": "A preview of [MASK] printable dot marker coloring pages. Each page has a large 4th of July themed image with dots to color in with a dauber style marker.",
      "answer": 7
    }
  ]
}


# ---------- Memory Management ----------
def flush_vram():
    """
    Forcefully runs Python garbage collection and empties the PyTorch CUDA cache.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ---------- Helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def create_disk(radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(y, x)
    return (xx * xx + yy * yy) <= (radius * radius)


def sanitize_name(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    out = "".join(keep).strip("_")
    return out if out else "item"


# ---------- LLM Query Decomposer ----------
class QueryDecomposer:
    def __init__(self, model_name: str, quantization: str = "bf16"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Use bfloat16 for Llama 3.1 
        dtype = torch.bfloat16 if quantization == "bf16" else torch.float16
        
        self.slm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.slm_tokenizer = AutoTokenizer.from_pretrained(model_name)

    def decompose(self, query: str) -> dict:
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
                    "- If a field is missing, return an empty list [].\n"
                    "- Return ONLY valid JSON."
                ),
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
                ),
            },
        ]

        prompt = self.slm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.slm_tokenizer(prompt, return_tensors="pt").to(self.slm_model.device)

        with torch.no_grad():
            outputs = self.slm_model.generate(
                **inputs,
                max_new_tokens=250,
                pad_token_id=self.slm_tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.0,
            )

        decoded_text = self.slm_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        try:
            start_idx = decoded_text.find("{")
            end_idx = decoded_text.rfind("}") + 1
            parsed_json = json.loads(decoded_text[start_idx:end_idx])

            # Ensure schema adherence 
            target = str(parsed_json.get("target_noun", "object")).strip() or "object"
            adjectives = parsed_json.get("attributes", [])
            verbs = parsed_json.get("interactions", [])
            reasoning = parsed_json.get("reasoning", "")

            if not isinstance(adjectives, list):
                adjectives = []
            if not isinstance(verbs, list):
                verbs = []

            target = re.sub(r"\s+", " ", target).strip()
            adjectives = [re.sub(r"\s+", " ", str(a)).strip() for a in adjectives if str(a).strip()]
            verbs = [re.sub(r"\s+", " ", str(v)).strip() for v in verbs if str(v).strip()]

            return {
                "reasoning": reasoning,
                "target_noun": target if target else "object",
                "attributes": adjectives,
                "interactions": verbs,
            }
        except Exception as e:
            fallback_noun = query.strip("?. ").split()[-1] if query.strip() else "object"
            fallback_noun = re.sub(r"[^a-zA-Z0-9_-]+", "", fallback_noun) or "object"
            return {
                "reasoning": f"Fallback used due to parsing error: {e}",
                "target_noun": fallback_noun, 
                "attributes": [], 
                "interactions": []
            }


def build_san_prompt_from_decomp(decomp: dict) -> str:
    noun = str(decomp.get("target_noun", "object")).strip() or "object"
    attrs = [str(a).strip() for a in decomp.get("attributes", []) if str(a).strip()]
    phrase = " ".join(attrs + [noun]).strip()
    return phrase if phrase else noun


# ---------- Input Loaders ----------
def download_image(url: str, out_path: Path, timeout: int = 30):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; OmniCountPipeline/1.0)"}
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    with out_path.open("wb") as wf:
        wf.write(r.content)


def read_countbench_json_and_prepare(
    json_path: Path,
    image_dir: Path,
    decomposer: Optional[QueryDecomposer] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]], Dict[str, dict]]:
    ensure_dir(image_dir)

    # Auto-generate the fallback JSON if it doesn't exist
    if not json_path.exists():
        print(f"[{json_path.name}] not found. Generating default test file.")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(DEFAULT_JSON_DATA, f, indent=2)

    with json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    rows = obj.get("database", [])
    vocab_map: Dict[str, List[str]] = {}
    gt_map: Dict[str, Dict[str, int]] = {}
    decomp_map: Dict[str, dict] = {}

    for row in rows:
        qid = row["question_id"]
        image_url = row["image_url"]
        question = str(row.get("question", "")).strip()
        answer = int(row["answer"])

        parsed = urlparse(image_url)
        ext = Path(parsed.path).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            ext = ".jpg"

        img_name = f"q{qid}{ext}"
        img_path = image_dir / img_name

        if not img_path.exists():
            print(f"[JSON] downloading: {image_url} -> {img_name}")
            try:
                download_image(image_url, img_path)
            except Exception as e:
                print(f"Failed to download {image_url}: {e}")
                continue

        if decomposer is not None and question:
            decomp = decomposer.decompose(question)
            san_prompt = build_san_prompt_from_decomp(decomp)
        else:
            decomp = {"reasoning": "LLM disabled", "target_noun": "object", "attributes": [], "interactions": []}
            san_prompt = "object"

        vocab_map[img_name] = [san_prompt]
        gt_map[img_name] = {san_prompt: answer}
        decomp_map[img_name] = {
            "question_id": qid,
            "question": question,
            "decomposition": decomp, # Stores the full schema requested
            "san_prompt": san_prompt,
            "answer": answer,
            "image_url": image_url,
        }

    return vocab_map, gt_map, decomp_map


# ---------- Stage 1: Initialization and LLM Decompose ----------
def run_stage_1_llm_init(args, image_dir: Path, outputs_root: Path):
    """
    Downloads images and runs the LLM query decomposition.
    VRAM is flushed strictly at the end of this function.
    """
    decomposer = None
    if args.use_llm_decompose:
        print(f"[Stage 1] Loading {args.llm_model} with {args.llm_quant} quantization")
        decomposer = QueryDecomposer(model_name=args.llm_model, quantization=args.llm_quant)

    vocab_map, gt_map, decomp_map = read_countbench_json_and_prepare(
        json_path=args.countbench_json.resolve(),
        image_dir=image_dir,
        decomposer=decomposer,
    )

    decomp_out = outputs_root / "decomposition.json"
    with decomp_out.open("w", encoding="utf-8") as f:
        json.dump(decomp_map, f, indent=2)
    print(f"[Stage 1] Decomposition trace saved -> {decomp_out}")

    # Explicit deletion of LLM and VRAM flush
    if decomposer is not None:
        del decomposer
        flush_vram()
        print("[Stage 1] LLM unloaded, VRAM flushed.")

    return vocab_map, gt_map


# ---------- Stage 2: SAN masks ----------
def run_stage_2_san(
    repo_root: Path,
    image_dir: Path,
    vocab_map: Dict[str, List[str]],
    output_binary_masks: Path,
    san_config: Path,
    san_model_path: Path,
):
    """
    Runs via subprocess. Python inherently prevents VRAM leaks from subprocesses.
    """
    predict_script = repo_root / "external" / "SAN" / "predict.py"
    if not predict_script.exists():
        print(f"[Stage 2] Skipping SAN: Script not found at {predict_script}")
        return

    for image_name, classes in vocab_map.items():
        img_path = image_dir / image_name
        if not img_path.exists():
            continue

        image_id = Path(image_name).stem
        out_dir = output_binary_masks / image_id
        ensure_dir(out_dir)

        for class_name in classes:
            out_file = out_dir / f"output_mask_{class_name}.jpg"
            if out_file.exists():
                continue

            cmd = [
                sys.executable,
                str(predict_script),
                "--config-file", str(san_config),
                "--model-path", str(san_model_path),
                "--img-path", str(img_path),
                "--vocab", class_name,
                "--output-file", str(out_file),
            ]
            print("[Stage 2]", " ".join(cmd))
            subprocess.run(cmd, check=True)
            
    print("[Stage 2] SAN complete. Subprocess exit ensures VRAM is flushed.")


# ---------- Stage 3: Marigold depth ----------
def run_stage_3_marigold(repo_root: Path, image_dir: Path, depth_out_dir: Path):
    marigold_script = repo_root / "scripts" / "project_wrappers" / "marigold_run_mod.py"
    if not marigold_script.exists():
        print(f"[Stage 3] Skipping Marigold: Script not found at {marigold_script}")
        return

    cmd = [
        sys.executable, str(marigold_script),
        "--input_rgb_dir", str(image_dir),
        "--output_dir", str(depth_out_dir),
        "--denoise_steps", "10",
        "--ensemble_size", "10",
        "--processing_res", "768",
    ]
    print("[Stage 3]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[Stage 3] Marigold complete. Subprocess exit ensures VRAM is flushed.")


# ---------- Stage 4: Geometry-aware refinement ----------
def process_binary_mask(
    binary_mask: np.ndarray,
    other_masks: List[np.ndarray],
    depth_norm: np.ndarray,
    tolerance: float = 0.2,
    win: int = 5,
    radius: int = 5,
) -> np.ndarray:
    refined_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    edge_indices = np.where(binary_mask == 255)
    if len(edge_indices[0]) == 0:
        valid_depth_idx = np.where(depth_norm < 0.5)
        refined_mask[valid_depth_idx] = 255
    else:
        edge_points = zip(edge_indices[0], edge_indices[1])
        mask_indices = np.nonzero(binary_mask)
        mean_depth = float(np.mean(depth_norm[mask_indices])) if len(mask_indices[0]) else 0.0

        h, w = binary_mask.shape
        for x, y in edge_points:
            for k in range(x - win, x + win):
                for j in range(y - win, y + win):
                    if not (0 <= k < h and 0 <= j < w):
                        continue
                    if any(m[k, j] == 255 for m in other_masks):
                        continue
                    if abs(float(depth_norm[k, j]) - mean_depth) < tolerance:
                        refined_mask[k, j] = 255

    strel = create_disk(radius)
    opened = ndimage.binary_opening(refined_mask > 0, structure=strel)
    return opened.astype(np.uint8) * 255


def run_stage_4_refine(binary_masks_root: Path, depth_bw_dir: Path, refined_root: Path):
    ensure_dir(refined_root)

    for image_id_dir in sorted(binary_masks_root.iterdir()):
        if not image_id_dir.is_dir():
            continue
        image_id = image_id_dir.name

        depth_path = depth_bw_dir / f"{image_id}_pred.png"
        if not depth_path.exists():
            continue

        depth = np.array(Image.open(depth_path))
        depth_norm = normalize01(depth)

        out_dir = refined_root / image_id
        ensure_dir(out_dir)

        class_mask_files = [p for p in image_id_dir.iterdir() if p.is_file()]
        loaded_masks = {}
        for p in class_mask_files:
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            _, m_bin = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            loaded_masks[p.name] = m_bin

        for fname, m_bin in loaded_masks.items():
            class_name = Path(fname).stem.split("output_mask_")[-1]
            other = [v for k, v in loaded_masks.items() if k != fname]
            refined = process_binary_mask(m_bin, other, depth_norm)
            out_path = out_dir / f"mask_{class_name}.png"
            cv2.imwrite(str(out_path), refined)
            
    print("[Stage 4] Refinement complete.")


# ---------- Stage 5: Patch extraction ----------
def run_stage_5_patch(image_dir: Path, refined_root: Path, patches_root: Path):
    ensure_dir(patches_root)

    image_lookup = {
        p.stem: p
        for p in image_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    }

    for image_id_dir in sorted(refined_root.iterdir()):
        if not image_id_dir.is_dir():
            continue

        image_id = image_id_dir.name
        if image_id not in image_lookup:
            continue

        rgb = cv2.imread(str(image_lookup[image_id]))
        if rgb is None:
            continue

        out_dir = patches_root / image_id
        ensure_dir(out_dir)

        for mask_file in image_id_dir.iterdir():
            if not mask_file.is_file():
                continue
            class_name = mask_file.stem.split("mask_")[-1]
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            if mask.shape[:2] != rgb.shape[:2]:
                mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

            patch = rgb.copy()
            patch[mask == 0] = 0
            cv2.imwrite(str(out_dir / f"patch_{class_name}.png"), patch)
            
    print("[Stage 5] Patching complete.")


# ---------- Stage 6: SAM + CLIP-guided counting ----------
def run_stage_6_sam_count(
    repo_root: Path,
    patches_root: Path,
    refined_root: Path,
    sam_out_root: Path,
    sam_checkpoint: Path,
    outputs_root: Path,
    min_peak_distance: int = 10,
    peak_threshold_rel: float = 0.9,
    min_mask_area: int = 20,
) -> Dict[str, Dict[str, int]]:
    
    # 1. Load Stage 1 Metadata
    decomp_path = outputs_root / "decomposition.json"
    if not decomp_path.exists():
        print(f"[Stage 6] Error: {decomp_path} not found.")
        return {}
    
    with decomp_path.open("r", encoding="utf-8") as f:
        decomp_db = json.load(f)

    # 2. Environment Setup
    sys.path.insert(0, str(repo_root / "external"))
    sys.path.insert(0, str(repo_root / "external" / "tfoc"))
    from clips import clip
    from tfoc.shi_segment_anything import sam_model_registry
    from tfoc.shi_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3. Model Initialization
    # Loading CLIP for semantic validation
    clip_model, _ = clip.load("CS-ViT-B/16", device=device)
    
    # Loading SAM for geometric proposals
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    pred_counts: Dict[str, Dict[str, int]] = {}
    ensure_dir(sam_out_root)

    # 4. Processing Loop
    for image_id_dir in sorted(patches_root.iterdir()):
        if not image_id_dir.is_dir(): continue
        
        image_id = image_id_dir.name
        # Match image_id to Stage 1 noun (e.g., folder 'q2' matches key 'q2.jpg')
        img_meta = next((v for k, v in decomp_db.items() if Path(k).stem == image_id), None)
        if not img_meta: continue

        target_noun = img_meta["decomposition"]["target_noun"]
        pred_counts[image_id] = {}

        for patch_file in sorted(image_id_dir.glob("patch_*.png")):
            class_name = patch_file.stem.split("patch_")[-1]
            pil_img = Image.open(patch_file).convert("RGB")
            img_np = np.array(pil_img)

            with torch.no_grad():
                # A. CLIP: Generate Similarity Map for the specific target_noun
                # (Replaces the need for a broad 'all_texts' list)
                text_features = clip.encode_text_with_prompt_ensemble(clip_model, [target_noun], device)
                image_features = clip_model.encode_image(preprocess(pil_img).unsqueeze(0).to(device))
                
                similarity = clip.clip_feature_surgery(image_features, text_features)
                sim_map = clip.get_similarity_map(similarity[:, 1:, :], img_np.shape[:2])
                
                # B. SAM: Generate Candidate Masks
                # We use the Sim Map peaks as point prompts for SAM
                coords = peak_local_max(
                    sim_map.cpu().numpy().squeeze(), 
                    min_distance=min_peak_distance, 
                    threshold_rel=peak_threshold_rel
                )
                
                points = [[int(c[1]), int(c[0])] for c in coords]
                masks = mask_generator.generate(img_np, points) if points else []

            # C. Filtering & Counting
            valid_count = 0
            for i, m in enumerate(masks):
                if m['area'] < min_mask_area: continue
                
                # Save individual object segments for inspection
                out_dir = sam_out_root / image_id / class_name
                ensure_dir(out_dir)
                mask_overlay = (m['segmentation'][..., None] * img_np).astype(np.uint8)
                cv2.imwrite(str(out_dir / f"obj_{i}.png"), cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR))
                
                valid_count += 1

            pred_counts[image_id][class_name] = valid_count

    # 5. Cleanup
    del clip_model, sam
    gc.collect()
    torch.cuda.empty_cache()
    
    return pred_counts


# ---------- Evaluation ----------
def evaluate_against_gt(
    pred_counts: Dict[str, Dict[str, int]],
    gt_map: Dict[str, Dict[str, int]],
    outputs_root: Path,
):
    rows = []
    abs_err = []
    sq_err = []

    for img_name, cls_dict in gt_map.items():
        image_id = Path(img_name).stem
        for cls_name, gt in cls_dict.items():
            pred = pred_counts.get(image_id, {}).get(cls_name, 0)
            ae = abs(pred - gt)
            se = (pred - gt) ** 2
            abs_err.append(ae)
            sq_err.append(se)
            rows.append({
                "image": img_name,
                "image_id": image_id,
                "class": cls_name,
                "pred": pred,
                "gt": gt,
                "abs_err": ae,
                "sq_err": se,
            })

    metrics = {
        "num_samples": len(rows),
        "mae": (sum(abs_err) / len(abs_err)) if abs_err else None,
        "rmse": (float(np.sqrt(sum(sq_err) / len(sq_err)))) if sq_err else None,
        "rows": rows,
    }

    eval_path = outputs_root / "countbench_eval.json"
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Eval] saved -> {eval_path}")


def main():
    parser = argparse.ArgumentParser("OmniCount pipeline runner")

    # Paths default to local structure instead of throwing requirements errors
    parser.add_argument("--repo_root", type=Path, default=Path("."), help="Path to OmniCount repo root")
    parser.add_argument("--image_dir", type=Path, default=Path("./images"), help="Input RGB images")
    parser.add_argument("--outputs_root", type=Path, default=Path("./outputs"), help="Output directory")
    parser.add_argument("--countbench_json", type=Path, default=Path("./labels/countbench_test.json"), help="Countbench JSON")

    # LLM Settings
    parser.add_argument("--use_llm_decompose", action="store_true", help="Use LLM Stage 1 on `question`")
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--llm_quant", type=str, default="bf16", choices=["4bit", "8bit", "fp16", "bf16"])

    # Execution Flow
    parser.add_argument("--run_san", action="store_true")
    parser.add_argument("--run_marigold", action="store_true")
    parser.add_argument("--run_refine", action="store_true")
    parser.add_argument("--run_patch", action="store_true")
    parser.add_argument("--run_sam_count", action="store_true")
    parser.add_argument("--run_all", action="store_true")

    parser.add_argument("--pred_json", type=Path, default=None)
    parser.add_argument("--san_config", type=Path, default=None)
    parser.add_argument("--san_model_path", type=Path, default=None)
    parser.add_argument("--sam_checkpoint", type=Path, default=None)

    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    image_dir = args.image_dir.resolve()
    outputs_root = args.outputs_root.resolve()
    
    ensure_dir(outputs_root)
    ensure_dir(image_dir)

    # Resolve deep paths relative to repo_root
    san_config = args.san_config or (repo_root / "external" / "SAN" / "configs" / "san_clip_vit_large_res4_coco.yaml")
    san_model_path = args.san_model_path or (repo_root / "external" / "SAN" / "models" / "san_vit_large_14.pth")
    sam_checkpoint = args.sam_checkpoint or (repo_root / "sam_vit_b_01ec64.pth")

    binary_masks_root = outputs_root / "binary_masks"
    depth_out_root = outputs_root / "depth"
    depth_bw_dir = depth_out_root / "depth_bw"
    refined_root = outputs_root / "refined_bin_masks"
    patches_root = outputs_root / "patches"
    sam_out_root = outputs_root / "sam_patches"

    if args.run_all:
        args.run_san = args.run_marigold = args.run_refine = args.run_patch = args.run_sam_count = True

    # ------------------ STAGE PIPELINE EXECUTION ------------------

    # Stage 1: LLM Initialization
    vocab_map, gt_map = run_stage_1_llm_init(args, image_dir, outputs_root)

    # Stage 2: SAN
    if args.run_san:
        run_stage_2_san(repo_root, image_dir, vocab_map, binary_masks_root, san_config, san_model_path)

    # Stage 3: Marigold
    if args.run_marigold:
        run_stage_3_marigold(repo_root, image_dir, depth_out_root)

    # Stage 4: Refine
    if args.run_refine:
        run_stage_4_refine(binary_masks_root, depth_bw_dir, refined_root)

    # Stage 5: Patch
    if args.run_patch:
        run_stage_5_patch(image_dir, refined_root, patches_root)

    # Stage 6: SAM Count
    if args.run_sam_count:
        pred_counts = run_stage_6_sam_count(
            repo_root, patches_root, refined_root, sam_out_root, sam_checkpoint
        )

        if pred_counts:
            pred_json = args.pred_json or (outputs_root / "pred_counts.json")
            with pred_json.open("w", encoding="utf-8") as f:
                json.dump(pred_counts, f, indent=2)
            print(f"[Done] Saved counts -> {pred_json}")

            if gt_map is not None:
                evaluate_against_gt(pred_counts, gt_map, outputs_root)


if __name__ == "__main__":
    main()