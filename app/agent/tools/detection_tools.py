"""Grounding DINO detection tool."""
import json
import os
import uuid
from typing import Optional

import torch
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from .model_registry import registry

MAX_DIM = 1333


def _resize_if_needed(image: Image.Image):
    """Resize image if largest dimension exceeds MAX_DIM, return (image, scale)."""
    w, h = image.size
    max_dim = max(w, h)
    if max_dim <= MAX_DIM:
        return image, 1.0
    scale = MAX_DIM / max_dim
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS), scale


def run_grounding_dino(
    image_path: str,
    text_prompt: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    output_dir: str = "tmp",
) -> str:
    """
    Run Grounding DINO zero-shot detection on an image.

    Returns JSON with detection results including crop paths and annotated image path.
    """
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    resized, scale = _resize_if_needed(image)
    res_w, res_h = resized.size

    processor = registry.gdino_processor
    model = registry.gdino_model
    device = registry.device

    inputs = processor(images=resized, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(res_h, res_w)],
    )[0]

    boxes = results["boxes"].cpu().tolist()
    scores = results["scores"].cpu().tolist()
    labels = results["labels"]

    # Scale boxes back to original image coordinates
    inv_scale = 1.0 / scale
    boxes_orig = [
        [
            b[0] * inv_scale,
            b[1] * inv_scale,
            b[2] * inv_scale,
            b[3] * inv_scale,
        ]
        for b in boxes
    ]

    # Crop each detection from the original image
    crop_paths = []
    for i, box in enumerate(boxes_orig):
        x0, y0, x1, y1 = [max(0, int(v)) for v in box]
        x1 = min(x1, orig_w)
        y1 = min(y1, orig_h)
        crop = image.crop((x0, y0, x1, y1))
        crop_path = os.path.join(output_dir, f"crop_{uuid.uuid4().hex}.png")
        crop.save(crop_path)
        crop_paths.append(crop_path)

    # Draw numbered boxes on a copy
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]
    for i, (box, score, label) in enumerate(zip(boxes_orig, scores, labels)):
        x0, y0, x1, y1 = box
        color = colors[i % len(colors)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0 + 4, y0 + 2), f"{i}: {label} {score:.2f}", fill=color, font=font)

    annotated_path = os.path.join(output_dir, f"annotated_{uuid.uuid4().hex}.png")
    annotated.save(annotated_path)

    return json.dumps({
        "num_detections": len(boxes),
        "boxes_xyxy": boxes_orig,
        "scores": scores,
        "labels": labels,
        "crop_paths": crop_paths,
        "annotated_image_path": annotated_path,
    })


class GroundingDINOInput(BaseModel):
    image_path: str = Field(description="Absolute path to the input image file")
    text_prompt: str = Field(
        description="Text prompt describing objects to detect, e.g. 'cat . dog . person'"
    )
    output_dir: str = Field(
        description="Absolute path to directory where crops and annotated image will be saved"
    )


grounding_dino_tool = StructuredTool.from_function(
    func=run_grounding_dino,
    name="grounding_dino_detect",
    description=(
        "Detect objects in an image using Grounding DINO zero-shot detection. "
        "Returns JSON with detected boxes, scores, labels, crop image paths, and annotated image path. "
        "Use dot-separated prompts like 'cat . dog' to detect multiple classes."
    ),
    args_schema=GroundingDINOInput,
)
