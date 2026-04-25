#!/usr/bin/env python3
"""
run_omnicount_pipeline.py

OmniCount-style pipeline runner with optional LLM query decomposition.

Pipeline:
1) Input loading (TSV or CountBench JSON)
2) (Optional) LLM decomposition from `question` -> target_noun + attributes + interactions
3) SAN mask generation
4) Marigold depth estimation
5) Geometry-aware mask refinement
6) Patch extraction
7) SAM + CLIP-guided counting
8) Optional evaluation (if GT available from JSON)

CountBench JSON keys used:
- question_id
- image_url
- question
- answer
- Dino_prompt (fallback if LLM decomposition disabled)

Example (JSON + LLM):
python run_omnicount_pipeline.py \
  --repo_root /path/to/OmniCount \
  --image_dir /path/to/images \
  --countbench_json CountBench_test.json \
  --outputs_root /path/to/outputs \
  --use_llm_decompose \
  --llm_model meta-llama/Llama-3.1-8B-Instruct \
  --llm_quant 4bit \
  --run_all
"""

import os
import re
import sys
import json
import argparse
import subprocess
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
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        quantization: str = "4bit",
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.model_name = model_name
        self.quantization = quantization

        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.slm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.slm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        elif quantization == "fp16":
            self.slm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            raise ValueError("quantization must be one of: 4bit, 8bit, fp16")

        self.slm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.slm_tokenizer.pad_token is None:
            self.slm_tokenizer.pad_token = self.slm_tokenizer.eos_token

    def stage_1_decompose(self, query: str) -> dict:
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
            parsed = json.loads(decoded_text[start_idx:end_idx])

            target = str(parsed.get("target_noun", "object")).strip() or "object"
            adjectives = parsed.get("attributes", [])
            verbs = parsed.get("interactions", [])

            if not isinstance(adjectives, list):
                adjectives = []
            if not isinstance(verbs, list):
                verbs = []

            target = re.sub(r"\s+", " ", target).strip()
            adjectives = [re.sub(r"\s+", " ", str(a)).strip() for a in adjectives if str(a).strip()]
            verbs = [re.sub(r"\s+", " ", str(v)).strip() for v in verbs if str(v).strip()]

            return {
                "target_noun": target if target else "object",
                "adjectives": adjectives,
                "verbs": verbs,
            }
        except Exception:
            fallback_noun = query.strip("?. ").split()[-1] if query.strip() else "object"
            fallback_noun = re.sub(r"[^a-zA-Z0-9_-]+", "", fallback_noun) or "object"
            return {"target_noun": fallback_noun, "adjectives": [], "verbs": []}


def build_san_prompt_from_decomp(decomp: dict) -> str:
    """
    Build a grounding phrase for SAN.
    Keep it noun + appearance attributes (adjectives).
    Interactions are kept in metadata but not appended by default.
    """
    noun = str(decomp.get("target_noun", "object")).strip() or "object"
    attrs = [str(a).strip() for a in decomp.get("adjectives", []) if str(a).strip()]
    phrase = " ".join(attrs + [noun]).strip()
    return phrase if phrase else noun


# ---------- Input Loaders ----------
def read_vocab_tsv(vocab_tsv: Path) -> Dict[str, List[str]]:
    """
    TSV format:
    image_name.jpg\tclass_a,class_b
    """
    data: Dict[str, List[str]] = {}
    with vocab_tsv.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_name, vocab = line.split("\t")
            classes = [c.strip() for c in vocab.split(",") if c.strip()]
            data[image_name] = classes
    return data


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
    """
    Returns:
      vocab_map: image_filename -> [san_prompt]
      gt_map:    image_filename -> {san_prompt: answer}
      decomp_map:image_filename -> decomposition metadata
    """
    ensure_dir(image_dir)

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
        dino_prompt = str(row.get("Dino_prompt", "object")).strip() or "object"
        answer = int(row["answer"])

        parsed = urlparse(image_url)
        ext = Path(parsed.path).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            ext = ".jpg"

        # Keep filename deterministic per question_id
        img_name = f"q{qid}{ext}"
        img_path = image_dir / img_name

        if not img_path.exists():
            print(f"[JSON] downloading: {image_url} -> {img_name}")
            download_image(image_url, img_path)

        if decomposer is not None and question:
            decomp = decomposer.stage_1_decompose(question)
            san_prompt = build_san_prompt_from_decomp(decomp)
        else:
            decomp = {"target_noun": dino_prompt, "adjectives": [], "verbs": []}
            san_prompt = dino_prompt

        vocab_map[img_name] = [san_prompt]
        gt_map[img_name] = {san_prompt: answer}
        decomp_map[img_name] = {
            "question_id": qid,
            "question": question,
            "dino_prompt": dino_prompt,
            "decomposition": decomp,
            "san_prompt": san_prompt,
            "answer": answer,
            "image_url": image_url,
        }

    return vocab_map, gt_map, decomp_map


# ---------- Step 1: SAN masks ----------
def run_san_predict(
    repo_root: Path,
    image_dir: Path,
    vocab_map: Dict[str, List[str]],
    output_binary_masks: Path,
    san_config: Path,
    san_model_path: Path,
):
    predict_script = repo_root / "external" / "SAN" / "predict.py"

    for image_name, classes in vocab_map.items():
        img_path = image_dir / image_name
        if not img_path.exists():
            print(f"[SAN] missing image: {img_path}")
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
            print("[SAN]", " ".join(cmd))
            subprocess.run(cmd, check=True)


# ---------- Step 2: Marigold depth ----------
def run_marigold_depth(repo_root: Path, image_dir: Path, depth_out_dir: Path):
    marigold_script = repo_root / "scripts" / "project_wrappers" / "marigold_run_mod.py"
    cmd = [
        sys.executable, str(marigold_script),
        "--input_rgb_dir", str(image_dir),
        "--output_dir", str(depth_out_dir),
        "--denoise_steps", "10",
        "--ensemble_size", "10",
        "--processing_res", "768",
    ]
    print("[Marigold]", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------- Step 3: Geometry-aware refinement ----------
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


def refine_all_masks(binary_masks_root: Path, depth_bw_dir: Path, refined_root: Path):
    ensure_dir(refined_root)

    for image_id_dir in sorted(binary_masks_root.iterdir()):
        if not image_id_dir.is_dir():
            continue
        image_id = image_id_dir.name

        depth_path = depth_bw_dir / f"{image_id}_pred.png"
        if not depth_path.exists():
            print(f"[Refine] missing depth for {image_id}")
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


# ---------- Step 4: Patch extraction ----------
def extract_patches(image_dir: Path, refined_root: Path, patches_root: Path):
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
            print(f"[Patch] source image not found for image_id={image_id}")
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


# ---------- Step 5: SAM + CLIP-guided counting ----------
def run_sam_count(
    repo_root: Path,
    patches_root: Path,
    refined_root: Path,
    sam_out_root: Path,
    sam_checkpoint: Path,
    min_peak_distance: int = 10,
    peak_threshold_rel: float = 0.9,
    min_mask_area: int = 20,
) -> Dict[str, Dict[str, int]]:
    sys.path.insert(0, str(repo_root / "external"))
    sys.path.insert(0, str(repo_root / "external" / "tfoc"))

    from clips import clip
    from transformers import AutoProcessor, CLIPSegVisionModel
    from tfoc.shi_segment_anything import sam_model_registry
    from tfoc.shi_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
    from sam.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator as SamAutomaticMaskGeneratorFallback

    device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocess = Compose([
        Resize((512, 512), interpolation=BICUBIC),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    all_texts = [
        'airplane', 'bag', 'strawberries', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle',
        'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup',
        'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard',
        'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'potted plant', 'road', 'rock',
        'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree', 'truck',
        'tv monitor', 'wall', 'water', 'window', 'wood'
    ]

    clip_model, _ = clip.load("CS-ViT-B/16", device=device)
    clip_model.eval()

    clipseg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_vision = CLIPSegVisionModel.from_pretrained("CIDAS/clipseg-rd64-refined")

    sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint))
    sam.to(device=device)

    pred_counts: Dict[str, Dict[str, int]] = {}
    ensure_dir(sam_out_root)

    for image_id_dir in sorted(patches_root.iterdir()):
        if not image_id_dir.is_dir():
            continue
        image_id = image_id_dir.name
        pred_counts[image_id] = {}

        for patch_file in sorted(image_id_dir.glob("patch_*.png")):
            class_name = patch_file.stem.split("patch_")[-1]
            mask_file = refined_root / image_id / f"mask_{class_name}.png"
            if not mask_file.exists():
                pred_counts[image_id][class_name] = 0
                continue

            pil_img = Image.open(patch_file).convert("RGB")

            image_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = clip.encode_text_with_prompt_ensemble(clip_model, all_texts, device)
                similarity = clip.clip_feature_surgery(image_features, text_features)
                similarity_map = clip.get_similarity_map(
                    similarity[:, 1:, :],
                    np.array(pil_img).shape[:2]
                )

            pred = np.array(Image.open(mask_file).convert("L"))

            inputs = clipseg_processor(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                _ = clipseg_vision(**inputs).last_hidden_state  # parity with upstream flow

            new_mask = similarity_map.mean(dim=3).squeeze(0).cpu().numpy()
            exp_mask = pred.astype(np.float32) / 255.0
            hadamard = (1.0 - new_mask) * exp_mask
            hadamard = cv2.GaussianBlur(hadamard, (5, 5), sigmaX=0)

            coords = peak_local_max(
                hadamard,
                min_distance=min_peak_distance,
                threshold_rel=peak_threshold_rel,
            )
            point_prompts = [[int(c[1]), int(c[0])] for c in coords]

            if len(point_prompts) < 1:
                mask_generator = SamAutomaticMaskGeneratorFallback(sam)
                masks = mask_generator.generate(np.asarray(pil_img))
            else:
                mask_generator = SamAutomaticMaskGenerator(sam)
                masks = mask_generator.generate(np.asarray(pil_img), point_prompts)

            out_dir = sam_out_root / image_id / class_name
            ensure_dir(out_dir)

            valid_count = 0
            idx = 0
            rgb_arr = np.asarray(pil_img)
            for m in masks:
                seg = m.get("segmentation", None)
                if seg is None:
                    continue
                seg = seg.astype(np.uint8)
                if seg.sum() < min_mask_area:
                    continue

                overlay = np.where(seg[..., None] > 0, rgb_arr, np.zeros_like(rgb_arr))
                out_file = out_dir / f"{class_name}_{idx}.png"
                cv2.imwrite(str(out_file), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                idx += 1
                valid_count += 1

            pred_counts[image_id][class_name] = valid_count

    return pred_counts


# ---------- Optional evaluation ----------
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
    parser = argparse.ArgumentParser("OmniCount pipeline runner + optional LLM decomposition")

    parser.add_argument("--repo_root", type=Path, required=True, help="Path to OmniCount repo root")
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory with input RGB images")
    parser.add_argument("--outputs_root", type=Path, required=True, help="Output root directory")

    # Input mode
    parser.add_argument("--vocab_tsv", type=Path, default=None, help="TSV mapping image -> class list")
    parser.add_argument("--countbench_json", type=Path, default=None, help="CountBench-style JSON")

    # Optional LLM decompose (JSON mode)
    parser.add_argument("--use_llm_decompose", action="store_true", help="Use Llama decomposition on `question`")
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--llm_quant", type=str, default="4bit", choices=["4bit", "8bit", "fp16"])

    parser.add_argument("--run_san", action="store_true")
    parser.add_argument("--run_marigold", action="store_true")
    parser.add_argument("--run_refine", action="store_true")
    parser.add_argument("--run_patch", action="store_true")
    parser.add_argument("--run_sam_count", action="store_true")
    parser.add_argument("--run_all", action="store_true")

    parser.add_argument("--pred_json", type=Path, default=None)

    parser.add_argument(
        "--san_config",
        type=Path,
        default=None,
        help="Default: <repo_root>/external/SAN/configs/san_clip_vit_large_res4_coco.yaml",
    )
    parser.add_argument(
        "--san_model_path",
        type=Path,
        default=None,
        help="Default: <repo_root>/external/SAN/models/san_vit_large_14.pth",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=Path,
        default=None,
        help="Default: <repo_root>/sam_vit_b_01ec64.pth",
    )

    args = parser.parse_args()

    if (args.vocab_tsv is None) == (args.countbench_json is None):
        raise ValueError("Provide exactly one of --vocab_tsv or --countbench_json")

    if args.use_llm_decompose and args.countbench_json is None:
        raise ValueError("--use_llm_decompose currently applies to --countbench_json mode")

    repo_root = args.repo_root.resolve()
    image_dir = args.image_dir.resolve()
    outputs_root = args.outputs_root.resolve()
    ensure_dir(outputs_root)
    ensure_dir(image_dir)

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

    gt_map = None
    decomp_map = None

    if args.countbench_json is not None:
        decomposer = None
        if args.use_llm_decompose:
            print(f"[LLM] loading {args.llm_model} with {args.llm_quant} quantization")
            decomposer = QueryDecomposer(model_name=args.llm_model, quantization=args.llm_quant)

        vocab_map, gt_map, decomp_map = read_countbench_json_and_prepare(
            json_path=args.countbench_json.resolve(),
            image_dir=image_dir,
            decomposer=decomposer,
        )

        # save decomposition trace
        decomp_out = outputs_root / "decomposition.json"
        with decomp_out.open("w", encoding="utf-8") as f:
            json.dump(decomp_map, f, indent=2)
        print(f"[LLM] decomposition trace saved -> {decomp_out}")

        # release LLM before vision-heavy stages
        if decomposer is not None:
            del decomposer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        vocab_map = read_vocab_tsv(args.vocab_tsv.resolve())

    if args.run_san:
        run_san_predict(
            repo_root=repo_root,
            image_dir=image_dir,
            vocab_map=vocab_map,
            output_binary_masks=binary_masks_root,
            san_config=san_config,
            san_model_path=san_model_path,
        )

    if args.run_marigold:
        run_marigold_depth(
            repo_root=repo_root,
            image_dir=image_dir,
            depth_out_dir=depth_out_root,
        )

    if args.run_refine:
        refine_all_masks(
            binary_masks_root=binary_masks_root,
            depth_bw_dir=depth_bw_dir,
            refined_root=refined_root,
        )

    if args.run_patch:
        extract_patches(
            image_dir=image_dir,
            refined_root=refined_root,
            patches_root=patches_root,
        )

    if args.run_sam_count:
        pred_counts = run_sam_count(
            repo_root=repo_root,
            patches_root=patches_root,
            refined_root=refined_root,
            sam_out_root=sam_out_root,
            sam_checkpoint=sam_checkpoint,
        )

        pred_json = args.pred_json or (outputs_root / "pred_counts.json")
        with pred_json.open("w", encoding="utf-8") as f:
            json.dump(pred_counts, f, indent=2)
        print(f"[Done] saved counts -> {pred_json}")

        if gt_map is not None:
            evaluate_against_gt(pred_counts, gt_map, outputs_root)


if __name__ == "__main__":
    main()