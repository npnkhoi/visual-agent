"""VLM-based scoring for [MASK] fill-in: score each number-word candidate
independently with a separate VLM call, then pick the highest score."""
import re
from pathlib import Path

from agentflow.processors.base import Processor
from ..types import MaskScores
from .vlm_backend import load_vlm, run_vlm
from .clip_mask_score import NUMBER_WORDS

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

SCORE_PROMPT = """\
Caption: "{caption}"

How accurately does this caption describe the image?
Reply with a single integer from 0 (completely wrong) to 10 (perfectly accurate).
"""


def _score_all_words(image_path: Path, question: str, model_id: str) -> dict[str, float]:
    model, processor = load_vlm(model_id)
    scores = {}
    for word in NUMBER_WORDS:
        caption = question.replace("[MASK]", word)
        prompt = SCORE_PROMPT.format(caption=caption)
        response = run_vlm(image_path, prompt, model, processor, model_id, max_new_tokens=4)
        match = re.search(r"\d+", response)
        scores[word] = float(match.group()) if match else 0.0
    return scores


class VLMMaskScoreProcessor(Processor):
    def __init__(self, pipeline, stage_config):
        super().__init__(pipeline, stage_config)
        kw = stage_config.kwargs or {}
        self._model_id = kw.get("model_id", DEFAULT_MODEL_ID)

    def __call__(self, inputs: dict, logger=None, output_dir=None) -> MaskScores:
        image_path = Path(inputs[self._input_names_snake[0]])
        question: str = inputs[self._input_names_snake[1]]

        scores = _score_all_words(image_path, question, self._model_id)

        if logger:
            print(f"model: {self._model_id}", file=logger, flush=True)
            for w, s in scores.items():
                print(f"{w}: {s}", file=logger, flush=True)
            best = max(scores, key=scores.__getitem__)
            print(f"best: {best}", file=logger, flush=True)

        return MaskScores(scores=scores)
