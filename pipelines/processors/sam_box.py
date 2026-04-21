"""SAM box-prompted segmentation stage.

Takes GDINO DetectionResult (bounding boxes) and refines counts via
per-box SAM masks + mask-IoU NMS.

Input:  Image, DetectionResult (from GroundingDinoProcessor)
Output: DetectionResult (mask-NMS refined count)
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from agentflow.processors.base import Processor
from .grounding_dino import _resize
from ..types import DetectionResult

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


def refine_with_sam(
    image_path: Path,
    detection: DetectionResult,
    iou_threshold: float = MASK_IOU_THRESHOLD,
) -> DetectionResult:
    if not detection.boxes:
        return detection

    predictor = _get_sam()
    raw = Image.open(image_path).convert("RGB")
    resized = _resize(raw)
    img_array = np.array(resized)
    predictor.set_image(img_array)

    masks_list, valid_scores, valid_labels, valid_boxes = [], [], [], []
    for box, score, label in zip(detection.boxes, detection.scores, detection.labels):
        sam_masks, sam_scores, _ = predictor.predict(
            box=np.array(box), multimask_output=False
        )
        if len(sam_masks) > 0:
            masks_list.append(sam_masks[0].astype(bool))
            valid_scores.append(float(sam_scores[0]))
            valid_labels.append(label)
            valid_boxes.append(box)

    if not masks_list:
        return DetectionResult(num_boxes=0, labels=[], boxes=[], scores=[])

    keep = _nms_masks(masks_list, valid_scores, iou_threshold)
    return DetectionResult(
        num_boxes=len(keep),
        labels=[valid_labels[i] for i in keep],
        boxes=[valid_boxes[i] for i in keep],
        scores=[valid_scores[i] for i in keep],
    )


class SAMBoxProcessor(Processor):
    def __init__(self, pipeline, stage_config):
        super().__init__(pipeline, stage_config)
        kw = stage_config.kwargs or {}
        self._iou_threshold = float(kw.get("iou_threshold", MASK_IOU_THRESHOLD))

    def __call__(self, inputs: dict, logger=None, output_dir=None) -> DetectionResult:
        image_path = Path(inputs[self._input_names_snake[0]])
        detection: DetectionResult = inputs[self._input_names_snake[1]]
        result = refine_with_sam(image_path, detection, self._iou_threshold)
        if logger:
            print(f"boxes_in: {detection.num_boxes}", file=logger, flush=True)
            print(f"boxes_after_mask_nms: {result.num_boxes}", file=logger, flush=True)
        return result
