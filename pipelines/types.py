from __future__ import annotations

import numpy as np
from pydantic import BaseModel


class DinoPrompt(BaseModel):
    prompt: str


class DetectionResult(BaseModel):
    num_boxes: int
    labels: list[str]
    boxes: list[list[float]]   # [[x1, y1, x2, y2], ...] XYXY pixels
    scores: list[float]


class SAMMaskRLE(BaseModel):
    """Run-length encoding of a boolean mask (row-major, starting with zeros count)."""
    counts: list[int]
    size: list[int]   # [H, W]


class SAMMask(BaseModel):
    rle: SAMMaskRLE
    bbox: list[float]   # [x1, y1, x2, y2] XYXY pixels
    area: int
    score: float        # SAM predicted_iou


class SAMMasks(BaseModel):
    masks: list[SAMMask]


class ObjectCoordinates(BaseModel):
    """VLM-predicted (x, y) pixel centers for each detected object instance."""
    points: list[list[float]]   # [[x1, y1], [x2, y2], ...]


class VLMCount(BaseModel):
    count: int


class MaskScores(BaseModel):
    scores: dict[str, float]   # number word → CLIP cosine similarity


class LabeledImage(BaseModel):
    path: str
    num_points: int
    points: list[list[float]] = []       # NMS-filtered pixel coords [[x,y], ...]
    image_size: list[int] = []           # [W, H]
    original_image_path: str = ""


class VLMGroupCount(BaseModel):
    count: int
    groups: list[list[int]]  # each inner list = point IDs belonging to one object


class EvalResult(BaseModel):
    predicted: int
    ground_truth: int
    is_correct: bool


# ---------------------------------------------------------------------------
# RLE helpers
# ---------------------------------------------------------------------------

def encode_rle(mask: np.ndarray) -> SAMMaskRLE:
    """Encode a boolean HxW mask as run-length encoding (starts with zeros count)."""
    H, W = mask.shape
    flat = mask.flatten().astype(np.uint8)
    counts: list[int] = []
    if len(flat) == 0:
        return SAMMaskRLE(counts=[], size=[H, W])
    current = 0  # start counting zeros
    run = 0
    for px in flat:
        if px == current:
            run += 1
        else:
            counts.append(run)
            current ^= 1
            run = 1
    counts.append(run)
    return SAMMaskRLE(counts=counts, size=[H, W])


def decode_rle(rle: SAMMaskRLE) -> np.ndarray:
    """Decode RLE back to a boolean HxW mask."""
    H, W = rle.size
    flat = np.zeros(H * W, dtype=bool)
    idx = 0
    val = False  # starts with zeros
    for count in rle.counts:
        flat[idx: idx + count] = val
        idx += count
        val = not val
    return flat.reshape(H, W)
