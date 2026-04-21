"""Stage 1 (Oracle): Use target_noun from labels directly — skips LLM extraction."""
from agentflow.processors.base import Processor
from ..types import DinoPrompt


class OraclePromptProcessor(Processor):
    def __call__(self, inputs: dict, logger=None, output_dir=None) -> DinoPrompt:
        target_noun = inputs[self._input_names_snake[0]]
        prompt = target_noun.strip().rstrip(".") + " ."
        if logger:
            print(f"target_noun: {target_noun}", file=logger, flush=True)
            print(f"prompt: {prompt}", file=logger, flush=True)
        return DinoPrompt(prompt=prompt)
