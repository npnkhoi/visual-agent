"""Grounding DINO zero-shot object detection.

A single processor handles both tiny and base variants — pass model_id via
stage kwargs in the YAML (defaults to grounding-dino-tiny).

    processor: GroundingDinoProcessor
    kwargs:
      model_id: "IDEA-Research/grounding-dino-base"
"""
from pathlib import Path

import torch
from PIL import Image

from agentflow.processors.base import Processor
from ..types import DetectionResult

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
MAX_DINO_DIM = 1333
NMS_IOU_THRESHOLD = 0.5
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# model_id -> (processor, model, device)
_model_cache: dict[str, tuple] = {}


def _get_models(model_id: str):
    if model_id not in _model_cache:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        model.eval()
        _model_cache[model_id] = (proc, model, device)
    return _model_cache[model_id]


def _resize(image: Image.Image, max_size: int = MAX_DINO_DIM) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    if w >= h:
        return image.resize((max_size, int(h * max_size / w)), Image.Resampling.LANCZOS)
    return image.resize((int(w * max_size / h), max_size), Image.Resampling.LANCZOS)


def _apply_nms(boxes, scores, labels, iou_threshold: float = NMS_IOU_THRESHOLD):
    from torchvision.ops import nms as tv_nms
    if len(boxes) == 0:
        return boxes, scores, labels
    keep = tv_nms(boxes.float(), scores.float(), iou_threshold)
    return boxes[keep], scores[keep], [labels[i] for i in keep.tolist()]


def _detect(
    image_path: Path,
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    box_threshold: float = BOX_THRESHOLD,
) -> tuple[int, list[str], list[list[float]], list[float]]:
    proc, model, device = _get_models(model_id)
    raw = Image.open(image_path).convert("RGB")
    resized = _resize(raw)
    text = prompt if prompt.endswith(".") else f"{prompt}."
    inputs = proc(
        images=resized, text=text,
        return_tensors="pt", padding=True, truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=box_threshold, text_threshold=TEXT_THRESHOLD,
        target_sizes=[resized.size[::-1]],
    )[0]
    boxes, scores, labels = _apply_nms(results["boxes"], results["scores"], results["labels"])
    boxes_list = [b.tolist() for b in boxes] if len(boxes) > 0 else []
    scores_list = scores.tolist() if len(scores) > 0 else []
    return len(boxes_list), labels, boxes_list, scores_list


class GroundingDinoProcessor(Processor):
    def __init__(self, pipeline, stage_config):
        super().__init__(pipeline, stage_config)
        kw = stage_config.kwargs or {}
        self._model_id = kw.get("model_id", DEFAULT_MODEL_ID)
        self._box_threshold = float(kw.get("box_threshold", BOX_THRESHOLD))

    def __call__(self, inputs: dict, logger=None, output_dir=None) -> DetectionResult:
        image_path = Path(inputs[self._input_names_snake[0]])
        dino_prompt_obj = inputs[self._input_names_snake[1]]
        prompt = dino_prompt_obj.prompt if hasattr(dino_prompt_obj, "prompt") else str(dino_prompt_obj)
        num_boxes, labels, boxes, scores = _detect(
            image_path, prompt, self._model_id, self._box_threshold
        )
        if logger:
            print(f"model: {self._model_id}", file=logger, flush=True)
            print(f"prompt: {prompt}", file=logger, flush=True)
            print(f"num_boxes: {num_boxes}", file=logger, flush=True)
        return DetectionResult(num_boxes=num_boxes, labels=labels, boxes=boxes, scores=scores)
