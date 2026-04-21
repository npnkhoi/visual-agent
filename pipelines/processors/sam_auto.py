"""SAM full-image automatic mask generation stage.

Segments the entire image into instance masks and returns them as RLE-encoded
SAMMasks for downstream processing (e.g. CLIPVerifyProcessor).

Input:  Image
Output: SAMMasks (RLE-encoded masks with bbox, area, score)
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from agentflow.processors.base import Processor
from ..types import SAMMask, SAMMasks, encode_rle

SAM_CHECKPOINT = str(Path(__file__).parent.parent.parent / "models" / "sam_vit_b.pth")
SAM_MODEL_TYPE = "vit_b"
MIN_MASK_AREA_RATIO = 0.001   # ignore masks < 0.1% of image area
MAX_MASKS = 200

_sam_generator = None


def _get_generator():
    global _sam_generator
    if _sam_generator is None:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to(device)
        _sam_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=100,
        )
    return _sam_generator


def generate_masks(image_path: Path) -> SAMMasks:
    generator = _get_generator()
    img = Image.open(image_path).convert("RGB")
    img_area = img.width * img.height
    img_array = np.array(img)

    raw_masks = generator.generate(img_array)

    # Filter tiny masks and cap total
    raw_masks = [m for m in raw_masks if m["area"] / img_area >= MIN_MASK_AREA_RATIO]
    raw_masks = sorted(raw_masks, key=lambda m: m["predicted_iou"], reverse=True)[:MAX_MASKS]

    sam_masks = []
    for m in raw_masks:
        seg: np.ndarray = m["segmentation"]  # bool H×W
        x, y, w, h = m["bbox"]               # SAM returns XYWH
        sam_masks.append(SAMMask(
            rle=encode_rle(seg),
            bbox=[x, y, x + w, y + h],       # convert to XYXY
            area=int(m["area"]),
            score=float(m["predicted_iou"]),
        ))

    return SAMMasks(masks=sam_masks)


class SAMAutoProcessor(Processor):
    def __call__(self, inputs: dict, logger=None, output_dir=None) -> SAMMasks:
        image_path = Path(inputs[self._input_names_snake[0]])
        result = generate_masks(image_path)
        if logger:
            print(f"image: {image_path}", file=logger, flush=True)
            print(f"total_masks_generated: {len(result.masks)}", file=logger, flush=True)
        return result
