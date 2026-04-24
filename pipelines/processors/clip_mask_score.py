"""CLIP scoring for [MASK] fill-in: replace [MASK] with each number word two–ten,
score each filled sentence against the image via CLIP cosine similarity."""
from pathlib import Path

import torch
from PIL import Image as PILImage

from agentflow.processors.base import Processor
from ..types import MaskScores
from .clip_verify import _get_clip

NUMBER_WORDS = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
NUMBER_VALUES = {w: i + 2 for i, w in enumerate(NUMBER_WORDS)}


def score_mask_candidates(image_path: Path, question: str) -> dict[str, float]:
    model, processor = _get_clip()
    device = next(model.parameters()).device

    img = PILImage.open(image_path).convert("RGB")
    texts = [question.replace("[MASK]", w) for w in NUMBER_WORDS]

    # Repeat image once per text — use combined forward like clip_verify does
    inputs = processor(
        text=texts, images=[img] * len(texts),
        return_tensors="pt", padding=True, truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    # diagonal: image[i] paired with text[i] (all images are the same)
    sims = (img_emb * txt_emb).sum(dim=-1).cpu().tolist()

    return {w: float(s) for w, s in zip(NUMBER_WORDS, sims)}


class CLIPMaskScoreProcessor(Processor):
    def __call__(self, inputs: dict, logger=None, output_dir=None) -> MaskScores:
        image_path = Path(inputs[self._input_names_snake[0]])
        question: str = inputs[self._input_names_snake[1]]

        scores = score_mask_candidates(image_path, question)

        if logger:
            for w, s in scores.items():
                print(f"{w}: {s:.4f}", file=logger, flush=True)
            best = max(scores, key=scores.__getitem__)
            print(f"best: {best} ({scores[best]:.4f})", file=logger, flush=True)

        return MaskScores(scores=scores)
