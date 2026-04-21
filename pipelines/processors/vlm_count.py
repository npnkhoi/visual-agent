"""Local VLM counting processor.

Runs a vision-language model on the image and asks it to count the target noun.
The model_id is configurable via stage kwargs (defaults to Qwen2.5-VL-7B-Instruct).

Prompt is loaded from the pipeline's prompt_dir using agentflow's naming convention:
    {output}__{Input1}_{Input2}.md
e.g. prompts/VLMCount__Image_TargetNoun.md

    processor: VLMCountProcessor
    kwargs:
      model_id: "Qwen/Qwen2.5-VL-7B-Instruct"
"""
import re
from pathlib import Path

from agentflow.processors.base import Processor
from ..types import VLMCount
from .vlm_backend import load_vlm, run_vlm

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def _count(image_path: Path, target_noun: str, model_id: str, prompt_template: str) -> int:
    model, processor = load_vlm(model_id)
    question = prompt_template.format(target_noun=target_noun).strip()
    response = run_vlm(image_path, question, model, processor, model_id, max_new_tokens=8)
    match = re.search(r"\d+", response)
    return int(match.group()) if match else 0


class VLMCountProcessor(Processor):
    def __init__(self, pipeline, stage_config):
        super().__init__(pipeline, stage_config)
        kw = stage_config.kwargs or {}
        self._model_id = kw.get("model_id", DEFAULT_MODEL_ID)

        # Resolve prompt file: {output}__{Input1}_{Input2}.md
        prompt_id = f"{stage_config.output}__{'_'.join(self._input_names_camel)}"
        prompt_path = pipeline.prompt_dir / f"{prompt_id}.md"
        self._prompt_template = prompt_path.read_text(encoding="utf-8")

    def __call__(self, inputs: dict, logger=None, output_dir=None) -> VLMCount:
        image_path = Path(inputs[self._input_names_snake[0]])
        target_noun = inputs[self._input_names_snake[1]]
        count = _count(image_path, target_noun, self._model_id, self._prompt_template)
        if logger:
            print(f"model: {self._model_id}", file=logger, flush=True)
            print(f"target_noun: {target_noun}", file=logger, flush=True)
            print(f"vlm_count: {count}", file=logger, flush=True)
        return VLMCount(count=count)
