"""Extract centroid of each SAM mask → ObjectCoordinates.

Filters by predicted_iou score and area before computing centroids.
"""
import numpy as np
from agentflow.processors.base import Processor
from ..types import SAMMasks, ObjectCoordinates, decode_rle

MIN_IOU = 0.88
MIN_AREA_FRAC = 0.001   # 0.1% of image
MAX_AREA_FRAC = 0.20    # 20% of image
MAX_POINTS = 20


class MaskCenterProcessor(Processor):
    def __init__(self, pipeline, stage_config):
        super().__init__(pipeline, stage_config)
        kw = stage_config.kwargs or {}
        self._min_iou = float(kw.get("min_iou", MIN_IOU))
        self._min_area_frac = float(kw.get("min_area_frac", MIN_AREA_FRAC))
        self._max_area_frac = float(kw.get("max_area_frac", MAX_AREA_FRAC))
        self._max_points = int(kw.get("max_points", MAX_POINTS))

    def __call__(self, inputs: dict, logger=None, output_dir=None) -> ObjectCoordinates:
        sam_masks: SAMMasks = inputs[self._input_names_snake[0]]

        if not sam_masks.masks:
            return ObjectCoordinates(points=[])

        # Infer image area from first mask's RLE size
        h, w = sam_masks.masks[0].rle.size
        img_area = h * w
        min_area = self._min_area_frac * img_area
        max_area = self._max_area_frac * img_area

        # Filter by score and area, then sort by score descending
        filtered = [
            m for m in sam_masks.masks
            if m.score >= self._min_iou and min_area <= m.area <= max_area
        ]
        filtered.sort(key=lambda m: m.score, reverse=True)
        filtered = filtered[:self._max_points]

        points = []
        for mask in filtered:
            arr = decode_rle(mask.rle)
            ys, xs = np.nonzero(arr)
            if len(xs) == 0:
                continue
            points.append([float(xs.mean()), float(ys.mean())])

        if logger:
            print(f"total_masks: {len(sam_masks.masks)}", file=logger, flush=True)
            print(f"after_filter: {len(filtered)}", file=logger, flush=True)
            print(f"num_centers: {len(points)}", file=logger, flush=True)

        return ObjectCoordinates(points=points)
