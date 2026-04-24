"""Annotate image with numbered center points and save to output_dir."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from agentflow.processors.base import Processor
from ..types import ObjectCoordinates, LabeledImage


def _nms(points: list[list[float]], min_dist: float) -> list[int]:
    """Greedy NMS: keep a point only if no already-kept point is within min_dist."""
    kept = []
    pts = np.array(points)
    for i, p in enumerate(pts):
        if all(np.linalg.norm(p - pts[j]) >= min_dist for j in kept):
            kept.append(i)
    return kept


class LabeledImageProcessor(Processor):
    def __init__(self, pipeline, stage_config):
        super().__init__(pipeline, stage_config)
        kw = stage_config.kwargs or {}
        self._nms_radius_frac = float(kw.get("nms_radius_frac", 0.06))

    def __call__(self, inputs: dict, logger=None, output_dir=None) -> LabeledImage:
        image_path = Path(inputs[self._input_names_snake[0]])
        coords: ObjectCoordinates = inputs[self._input_names_snake[1]]

        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        radius = max(10, int(min(W, H) * 0.03))
        font_size = max(8, int(radius * 1.0))

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
            )
        except Exception:
            font = ImageFont.load_default()

        nms_min_dist = max(radius * 2.0, min(W, H) * self._nms_radius_frac)
        kept_indices = _nms(coords.points, min_dist=nms_min_dist)

        draw = ImageDraw.Draw(img)
        for label, idx in enumerate(kept_indices):
            x, y = int(coords.points[idx][0]), int(coords.points[idx][1])
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill="red", outline="white", width=max(2, radius // 5),
            )
            txt = str(label)
            bbox = draw.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x - tw / 2, y - th / 2), txt, fill="white", font=font)

        out_path = Path(output_dir) / "labeled_points.jpg"
        img.save(out_path)

        kept_points = [coords.points[i] for i in kept_indices]
        n = len(kept_indices)
        if logger:
            print(f"original_points: {len(coords.points)}", file=logger, flush=True)
            print(f"after_nms: {n}", file=logger, flush=True)
            print(f"radius: {radius}", file=logger, flush=True)
            print(f"saved: {out_path}", file=logger, flush=True)

        return LabeledImage(
            path=str(out_path), num_points=n,
            points=kept_points, image_size=[W, H],
            original_image_path=str(image_path),
        )
