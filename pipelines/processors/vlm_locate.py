"""VLM coordinate localization stage.

Asks the VLM to output pixel (x, y) centers for each instance of the target
noun, rather than a scalar count. Prompt is loaded from the pipeline's
prompt_dir (ObjectCoordinates__Image_TargetNoun.md).

Input:  Image, TargetNoun
Output: ObjectCoordinates { points: [[x, y], ...] }
"""
import json
import re
from pathlib import Path

from agentflow.processors.base import Processor
from ..types import ObjectCoordinates
from .vlm_backend import load_vlm, run_vlm

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def _parse_points(text: str) -> list[list[float]]:
    """Extract [[x, y], ...] from model response; robust to minor formatting issues."""
    try:
        candidate = re.search(r'\[[\s\S]*\]', text)
        if candidate:
            parsed = json.loads(candidate.group())
            if isinstance(parsed, list):
                points = []
                for item in parsed:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        points.append([float(item[0]), float(item[1])])
                return points
    except (json.JSONDecodeError, ValueError):
        pass
    pairs = re.findall(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', text)
    return [[float(x), float(y)] for x, y in pairs]


def _locate(image_path: Path, target_noun: str, model_id: str, prompt_template: str) -> list[list[float]]:
    model, processor = load_vlm(model_id)
    question = prompt_template.format(target_noun=target_noun).strip()
    response = run_vlm(image_path, question, model, processor, model_id, max_new_tokens=256)
    return _parse_points(response)


class VLMLocateProcessor(Processor):
    def __init__(self, pipeline, stage_config):
        super().__init__(pipeline, stage_config)
        kw = stage_config.kwargs or {}
        self._model_id = kw.get("model_id", DEFAULT_MODEL_ID)
        prompt_id = f"{stage_config.output}__{'_'.join(self._input_names_camel)}"
        prompt_path = pipeline.prompt_dir / f"{prompt_id}.md"
        self._prompt_template = prompt_path.read_text(encoding="utf-8")

    def __call__(self, inputs: dict, logger=None, output_dir=None) -> ObjectCoordinates:
        image_path = Path(inputs[self._input_names_snake[0]])
        target_noun = inputs[self._input_names_snake[1]]
        points = _locate(image_path, target_noun, self._model_id, self._prompt_template)
        if logger:
            print(f"model: {self._model_id}", file=logger, flush=True)
            print(f"target_noun: {target_noun}", file=logger, flush=True)
            print(f"vlm_points ({len(points)}): {points}", file=logger, flush=True)
        return ObjectCoordinates(points=points)
