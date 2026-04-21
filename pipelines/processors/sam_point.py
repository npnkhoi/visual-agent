"""SAM point-prompted segmentation stage.

Uses VLM-predicted (x, y) center coordinates as SAM foreground point prompts,
one per object instance. Mask-IoU NMS deduplicates overlapping segments
(handles cases where two nearby points hit the same object).

Input:  Image, ObjectCoordinates
Output: DetectionResult
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from agentflow.processors.base import Processor
from .grounding_dino import _resize
from ..types import DetectionResult, ObjectCoordinates

SAM_CHECKPOINT = str(Path(__file__).parent.parent.parent / "models" / "sam_vit_b.pth")
SAM_MODEL_TYPE = "vit_b"
MASK_IOU_THRESHOLD = 0.5

_sam_predictor = None


def _get_sam():
    global _sam_predictor
    if _sam_predictor is None:
        from segment_anything import sam_model_registry, SamPredictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to(device)
        _sam_predictor = SamPredictor(sam)
    return _sam_predictor


def _mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _nms_masks(masks: list[np.ndarray], scores: list[float], iou_threshold: float) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep, suppressed = [], set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in order:
            if j <= i or j in suppressed:
                continue
            if _mask_iou(masks[i], masks[j]) > iou_threshold:
                suppressed.add(j)
    return keep


def segment_at_points(
    image_path: Path,
    coords: ObjectCoordinates,
    iou_threshold: float = MASK_IOU_THRESHOLD,
) -> DetectionResult:
    if not coords.points:
        return DetectionResult(num_boxes=0, labels=[], boxes=[], scores=[])

    predictor = _get_sam()
    raw = Image.open(image_path).convert("RGB")
    resized = _resize(raw)
    orig_w, orig_h = raw.size
    res_w, res_h = resized.size

    img_array = np.array(resized)
    predictor.set_image(img_array)

    # Scale VLM coordinates (original image space) to resized image space
    scale_x = res_w / orig_w
    scale_y = res_h / orig_h

    masks_list, scores_list, boxes_list = [], [], []
    for x, y in coords.points:
        px = float(x) * scale_x
        py = float(y) * scale_y
        point_coords = np.array([[px, py]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)   # 1 = foreground

        sam_masks, sam_scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        # Pick highest-confidence mask among the three SAM candidates
        best = int(np.argmax(sam_scores))
        masks_list.append(sam_masks[best].astype(bool))
        scores_list.append(float(sam_scores[best]))

        # Derive bbox from mask
        rows = np.any(sam_masks[best], axis=1)
        cols = np.any(sam_masks[best], axis=0)
        y_idx = np.where(rows)[0]
        x_idx = np.where(cols)[0]
        if len(y_idx) == 0 or len(x_idx) == 0:
            continue
        y1, y2 = int(y_idx[0]), int(y_idx[-1])
        x1, x2 = int(x_idx[0]), int(x_idx[-1])
        boxes_list.append([float(x1), float(y1), float(x2), float(y2)])

    # Deduplicate overlapping masks (two VLM points on the same object)
    keep = _nms_masks(masks_list, scores_list, iou_threshold)
    return DetectionResult(
        num_boxes=len(keep),
        labels=["object"] * len(keep),
        boxes=[boxes_list[i] for i in keep],
        scores=[scores_list[i] for i in keep],
    )


class SAMPointProcessor(Processor):
    def __init__(self, pipeline, stage_config):
        super().__init__(pipeline, stage_config)
        kw = stage_config.kwargs or {}
        self._iou_threshold = float(kw.get("iou_threshold", MASK_IOU_THRESHOLD))

    def __call__(self, inputs: dict, logger=None, output_dir=None) -> DetectionResult:
        image_path = Path(inputs[self._input_names_snake[0]])
        coords: ObjectCoordinates = inputs[self._input_names_snake[1]]
        result = segment_at_points(image_path, coords, self._iou_threshold)
        if logger:
            print(f"vlm_points_in: {len(coords.points)}", file=logger, flush=True)
            print(f"masks_after_nms: {result.num_boxes}", file=logger, flush=True)
            print(f"scores: {[round(s, 3) for s in result.scores]}", file=logger, flush=True)
        return result
