"""CLIP verification stage for SAM masks.

Crops each SAM mask region from the image, scores it against the target noun
via CLIP cosine similarity, then applies mask-IoU NMS to deduplicate overlapping
regions of the same object.

Input:  Image, SAMMasks (from SAMAutoProcessor), TargetNoun
Output: DetectionResult
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from agentflow.processors.base import Processor
from ..types import DetectionResult, SAMMasks, decode_rle

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
CLIP_SIM_THRESHOLD = 0.22
MASK_NMS_IOU = 0.30

_clip_model = None
_clip_processor = None


def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
        _clip_model.eval()
    return _clip_model, _clip_processor


def _crop_bbox(image: Image.Image, bbox: list[float]) -> Image.Image:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return image.crop((x1, y1, x2 + 1, y2 + 1))


def _clip_similarity(crops: list[Image.Image], text: str) -> list[float]:
    model, processor = _get_clip()
    device = next(model.parameters()).device
    inputs = processor(
        text=[text], images=crops,
        return_tensors="pt", padding=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    return (img_emb @ txt_emb.T)[:, 0].cpu().tolist()


def _mask_iou_from_rle(rle1, rle2) -> float:
    m1 = decode_rle(rle1)
    m2 = decode_rle(rle2)
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _nms_by_score(indices: list[int], scores: list[float], rles, iou_threshold: float) -> list[int]:
    order = sorted(indices, key=lambda i: scores[i], reverse=True)
    keep, suppressed = [], set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in order:
            if j <= i or j in suppressed:
                continue
            if _mask_iou_from_rle(rles[i], rles[j]) > iou_threshold:
                suppressed.add(j)
    return keep


def verify_masks(
    image_path: Path,
    sam_masks: SAMMasks,
    target_noun: str,
    sim_threshold: float = CLIP_SIM_THRESHOLD,
    nms_iou: float = MASK_NMS_IOU,
) -> tuple[DetectionResult, list[float]]:
    if not sam_masks.masks:
        return DetectionResult(num_boxes=0, labels=[], boxes=[], scores=[]), []

    img = Image.open(image_path).convert("RGB")
    crops = [_crop_bbox(img, m.bbox) for m in sam_masks.masks]
    text_prompt = f"a photo of a {target_noun}"
    sims = _clip_similarity(crops, text_prompt)

    # Filter by threshold
    passing = [i for i, s in enumerate(sims) if s > sim_threshold]
    if not passing:
        return DetectionResult(num_boxes=0, labels=[], boxes=[], scores=[]), sims

    # Mask-IoU NMS on passing masks
    rles = [m.rle for m in sam_masks.masks]
    kept = _nms_by_score(passing, sims, rles, nms_iou)

    return DetectionResult(
        num_boxes=len(kept),
        labels=[target_noun] * len(kept),
        boxes=[sam_masks.masks[i].bbox for i in kept],
        scores=[sims[i] for i in kept],
    ), sims


class CLIPVerifyProcessor(Processor):
    def __init__(self, pipeline, stage_config):
        super().__init__(pipeline, stage_config)
        kw = stage_config.kwargs or {}
        self._sim_threshold = float(kw.get("sim_threshold", CLIP_SIM_THRESHOLD))
        self._nms_iou = float(kw.get("nms_iou", MASK_NMS_IOU))

    def __call__(self, inputs: dict, logger=None, output_dir=None) -> DetectionResult:
        image_path = Path(inputs[self._input_names_snake[0]])
        sam_masks: SAMMasks = inputs[self._input_names_snake[1]]
        target_noun = inputs[self._input_names_snake[2]]
        result, all_sims = verify_masks(
            image_path, sam_masks, target_noun,
            self._sim_threshold, self._nms_iou,
        )
        if logger:
            print(f"target_noun: {target_noun}", file=logger, flush=True)
            print(f"masks_in: {len(sam_masks.masks)}", file=logger, flush=True)
            print(f"passing_threshold: {sum(1 for s in all_sims if s > self._sim_threshold)}", file=logger, flush=True)
            print(f"count_after_nms: {result.num_boxes}", file=logger, flush=True)
            print(f"scores: {[round(s, 3) for s in result.scores]}", file=logger, flush=True)
        return result
