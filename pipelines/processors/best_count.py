"""Pick the number word with the highest CLIP score and return it as VLMCount."""
from agentflow.processors.base import Processor
from ..types import MaskScores, VLMCount
from .clip_mask_score import NUMBER_VALUES


class BestCountProcessor(Processor):
    def __call__(self, inputs: dict, logger=None, output_dir=None) -> VLMCount:
        mask_scores: MaskScores = inputs[self._input_names_snake[0]]

        best_word = max(mask_scores.scores, key=mask_scores.scores.__getitem__)
        count = NUMBER_VALUES[best_word]

        if logger:
            ranked = sorted(mask_scores.scores.items(), key=lambda x: x[1], reverse=True)
            for w, s in ranked:
                print(f"{w}: {s:.4f}", file=logger, flush=True)
            print(f"best_word: {best_word}  count: {count}", file=logger, flush=True)

        return VLMCount(count=count)
