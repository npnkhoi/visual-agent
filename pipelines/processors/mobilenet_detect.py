"""Object detection using SSDLite320 with MobileNetV3-Large backbone.

No text prompt required — detects all COCO objects above a score threshold.
"""
from pathlib import Path

import torch
from PIL import Image

from agentflow.processors.base import Processor
from ..types import DetectionResult

SCORE_THRESHOLD = 0.3

_model_cache: dict = {}


def _get_model():
    if "model" not in _model_cache:
        from torchvision.models.detection import (
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights,
        )
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ssdlite320_mobilenet_v3_large(weights=weights).to(device)
        model.eval()
        transform = weights.transforms()
        _model_cache["model"] = model
        _model_cache["transform"] = transform
        _model_cache["device"] = device
        _model_cache["labels"] = weights.meta["categories"]
    return _model_cache


def _detect(image_path: Path, score_threshold: float = SCORE_THRESHOLD) -> DetectionResult:
    cache = _get_model()
    model = cache["model"]
    transform = cache["transform"]
    device = cache["device"]
    label_names = cache["labels"]

    raw = Image.open(image_path).convert("RGB")
    tensor = transform(raw).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)[0]

    keep = outputs["scores"] >= score_threshold
    boxes = outputs["boxes"][keep].cpu().tolist()
    scores = outputs["scores"][keep].cpu().tolist()
    label_ids = outputs["labels"][keep].cpu().tolist()
    labels = [label_names[i] for i in label_ids]

    return DetectionResult(
        num_boxes=len(boxes),
        labels=labels,
        boxes=boxes,
        scores=scores,
    )


class MobileNetDetectProcessor(Processor):
    def __init__(self, pipeline, stage_config):
        super().__init__(pipeline, stage_config)
        kw = stage_config.kwargs or {}
        self._score_threshold = float(kw.get("score_threshold", SCORE_THRESHOLD))

    def __call__(self, inputs: dict, logger=None, output_dir=None) -> DetectionResult:
        image_path = Path(inputs[self._input_names_snake[0]])
        result = _detect(image_path, self._score_threshold)
        if logger:
            print(f"score_threshold: {self._score_threshold}", file=logger, flush=True)
            print(f"num_boxes: {result.num_boxes}", file=logger, flush=True)
            from collections import Counter
            print(f"labels: {dict(Counter(result.labels))}", file=logger, flush=True)
        return result
