"""VLM-based grouping: given a labeled image with numbered points, ask the VLM
to return groups of point IDs, each group exclusively belonging to one object."""
import json
import re
from pathlib import Path

from agentflow.processors.base import Processor
from ..types import LabeledImage, ObjectCoordinates, VLMGroupCount
from .vlm_backend import load_vlm, run_vlm

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT_GROUPS = """\
The image shows {n} candidate points labeled 0 to {last}, each marking the \
center of a detected region.

Your task: group the points by the {target_noun} instance they belong to.

Instructions:
- Each inner list must contain point IDs that all belong to the SAME {target_noun}.
- Every point ID must appear in at most one group.
- Omit points that do not correspond to any {target_noun}.
- Return ONLY a JSON list of lists, e.g.: [[0,3],[1,4,7],[2]]
"""

PROMPT_FLAT = """\
The image shows {n} candidate points labeled 0 to {last}, each marking the \
center of a detected region.

Your task: for each {target_noun} you can see, pick ONE point ID that best marks its center.

Instructions:
- Return exactly one ID per {target_noun} instance.
- Omit IDs that are not on a {target_noun}.
- Return ONLY a JSON list of integers, e.g.: [0, 3, 7]
"""

PROMPT_FLAT_V2 = """\
The image shows {n} candidate points labeled 0 to {last}.

Step 1: Count the distinct {target_noun} objects you see in the image.
Step 2: For each one, choose the single best-centered point ID.

Rules:
- Select EXACTLY one point per distinct {target_noun}. No more, no fewer.
- If a region has multiple nearby points, pick only the best one and skip the rest.
- Omit all points not on a {target_noun}.
- Return ONLY a JSON list of integers with length equal to your count, e.g.: [0, 3, 7]
"""

PROMPT_DIRECT = """\
How many {target_noun} can you see in this image?
Reply with a single integer and nothing else.
"""

PROMPT_PICK_N = """\
The image has {n} labeled dots (0 to {last}).
There are exactly {count} {target_noun} objects in this image.
Pick exactly {count} dot IDs, one representative per object.
Return ONLY a JSON list of integers, e.g.: [0, 3, 7]
"""

PROMPT_TWO_STEP = """\
The image shows {n} candidate points labeled 0 to {last}.

First, count the distinct {target_noun} objects you see in the image.
Then, select exactly that many point IDs — one representative point per object.
Multiple nearby points often belong to the same object; pick only one per object.

Return ONLY a JSON object: {{"count": N, "ids": [id1, id2, ...]}}
"""

PROMPT_TWO_STEP_V2 = """\
This image has {n} labeled dots (0 to {last}) overlaid on detected regions.
Important: there may be MULTIPLE dots on the same physical object.

Your task:
1. Count the distinct {target_noun} objects in the scene (ignore the dots while counting).
2. Then pick ONE dot per object that best marks its center.

Return ONLY a JSON object: {{"count": N, "ids": [id1, id2, ...]}}
"""

PROMPT_TWO_STEP_V3 = """\
The image has {n} labeled dots (0 to {last}) marking detected regions.
Note: multiple dots may appear on the SAME physical object.

Task:
1. Count the distinct {target_noun} objects — use the dots as location hints, \
but count the actual objects, not the dots.
2. Pick one representative dot ID per object.

Return ONLY: {{"count": N, "ids": [id1, id2, ...]}}
"""

PROMPT_TWO_STEP_COORDS = """\
The image shows {n} candidate points labeled 0 to {last}.
Point positions (x, y as fraction of image width/height):
{coords_str}

Step 1: Count the distinct {target_noun} objects in the image.
Step 2: Select one representative point ID per object.

Key rule: points that are spatially close (similar x or y fractions) almost always \
belong to the same object — select only ONE of them.

Return ONLY a JSON object: {{"count": N, "ids": [id1, id2, ...]}}
"""


def _parse_groups(response: str, n: int) -> list[list[int]]:
    """Extract the first valid list-of-lists from the VLM response."""
    match = re.search(r'\[\s*\[.*?\]\s*\]', response, re.DOTALL)
    if match:
        try:
            raw = json.loads(match.group())
            groups = []
            seen = set()
            for group in raw:
                if not isinstance(group, list):
                    continue
                ids = [int(x) for x in group if isinstance(x, (int, float)) and int(x) < n and int(x) not in seen]
                if ids:
                    seen.update(ids)
                    groups.append(ids)
            return groups
        except Exception:
            pass
    return []


class VLMGroupCountProcessor(Processor):
    def __init__(self, pipeline, stage_config):
        super().__init__(pipeline, stage_config)
        kw = stage_config.kwargs or {}
        self._model_id = kw.get("model_id", DEFAULT_MODEL_ID)
        self._mode = kw.get("mode", "groups")  # groups | flat | direct

    def __call__(self, inputs: dict, logger=None, output_dir=None) -> VLMGroupCount:
        labeled: LabeledImage = inputs[self._input_names_snake[0]]
        target_noun: str = inputs[self._input_names_snake[1]]
        coords: ObjectCoordinates = inputs[self._input_names_snake[2]]

        n = labeled.num_points

        if self._mode == "count_then_pick":
            # Step 1: count on original image
            orig_path = Path(labeled.original_image_path) if labeled.original_image_path else Path(labeled.path)
            model, processor_obj = load_vlm(self._model_id)
            count_resp = run_vlm(orig_path, PROMPT_DIRECT.format(target_noun=target_noun),
                                 model, processor_obj, self._model_id, max_new_tokens=16)
            try:
                count = int(re.search(r'\d+', count_resp).group())
            except Exception:
                count = 0
            # Step 2: pick IDs on labeled image
            if count > 0:
                pick_q = PROMPT_PICK_N.format(n=n, last=n - 1, count=count, target_noun=target_noun)
                pick_resp = run_vlm(Path(labeled.path), pick_q, model, processor_obj,
                                    self._model_id, max_new_tokens=256)
                match = re.search(r'\[[\d\s,]+\]', pick_resp)
                if match:
                    try:
                        ids = [int(x) for x in json.loads(match.group()) if isinstance(x, (int, float)) and int(x) < n]
                        ids = list(dict.fromkeys(ids))[:count]
                    except Exception:
                        ids = []
                else:
                    ids = []
            else:
                pick_resp, ids = "", []
            groups = [[i] for i in ids]
            if logger:
                print(f"model: {self._model_id}", file=logger, flush=True)
                print(f"mode: {self._mode}", file=logger, flush=True)
                print(f"target_noun: {target_noun}", file=logger, flush=True)
                print(f"count_response: {count_resp}", file=logger, flush=True)
                print(f"count: {count}", file=logger, flush=True)
                print(f"pick_response: {pick_resp}", file=logger, flush=True)
                print(f"ids: {ids}", file=logger, flush=True)
            return VLMGroupCount(count=count, groups=groups)
        elif self._mode == "two_step_v3":
            question = PROMPT_TWO_STEP_V3.format(n=n, last=n - 1, target_noun=target_noun)
        elif self._mode == "two_step_v2":
            question = PROMPT_TWO_STEP_V2.format(n=n, last=n - 1, target_noun=target_noun)
        elif self._mode == "two_step_coords":
            W, H = (labeled.image_size + [1, 1])[:2]
            coords_str = "  ".join(
                f"{i}: ({labeled.points[i][0]/W:.2f},{labeled.points[i][1]/H:.2f})"
                for i in range(n)
            )
            question = PROMPT_TWO_STEP_COORDS.format(
                n=n, last=n - 1, target_noun=target_noun, coords_str=coords_str
            )
        elif self._mode == "two_step":
            question = PROMPT_TWO_STEP.format(n=n, last=n - 1, target_noun=target_noun)
        elif self._mode == "flat_v2":
            question = PROMPT_FLAT_V2.format(n=n, last=n - 1, target_noun=target_noun)
        elif self._mode == "flat":
            question = PROMPT_FLAT.format(n=n, last=n - 1, target_noun=target_noun)
        elif self._mode == "direct":
            question = PROMPT_DIRECT.format(target_noun=target_noun)
        else:
            question = PROMPT_GROUPS.format(n=n, last=n - 1, target_noun=target_noun)

        model, processor = load_vlm(self._model_id)
        response = run_vlm(
            Path(labeled.path), question, model, processor,
            self._model_id, max_new_tokens=256,
        )

        if self._mode in ("two_step", "two_step_coords", "two_step_v2", "two_step_v3"):
            try:
                obj = json.loads(re.search(r'\{.*\}', response, re.DOTALL).group())
                count = int(obj.get("count", 0))
                ids = [int(x) for x in obj.get("ids", []) if int(x) < n]
                ids = list(dict.fromkeys(ids))
                if len(ids) != count:
                    count = len(ids)
            except Exception:
                count, ids = 0, []
            groups = [[i] for i in ids]
        elif self._mode == "direct":
            try:
                count = int(re.search(r'\d+', response).group())
            except Exception:
                count = 0
            groups = [[i] for i in range(count)]
        elif self._mode in ("flat", "flat_v2"):
            match = re.search(r'\[[\d\s,]+\]', response)
            if match:
                try:
                    ids = [int(x) for x in json.loads(match.group()) if isinstance(x, (int, float)) and int(x) < n]
                    ids = list(dict.fromkeys(ids))  # dedup while preserving order
                except Exception:
                    ids = []
            else:
                ids = []
            count = len(ids)
            groups = [[i] for i in ids]
        else:
            groups = _parse_groups(response, n)
            count = len(groups)

        if logger:
            print(f"model: {self._model_id}", file=logger, flush=True)
            print(f"mode: {self._mode}", file=logger, flush=True)
            print(f"target_noun: {target_noun}", file=logger, flush=True)
            print(f"response: {response}", file=logger, flush=True)
            print(f"groups: {groups}", file=logger, flush=True)
            print(f"count: {count}", file=logger, flush=True)

        return VLMGroupCount(count=count, groups=groups)
