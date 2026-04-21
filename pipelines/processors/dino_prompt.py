"""Stage 1: Extract a Grounding DINO-ready noun phrase from a CountBench question.

Uses Qwen2.5-1.5B-Instruct (local, lazy-loaded singleton) to pull the countable
object out of the masked question, then singularizes it via spaCy.
"""
import re
import torch
import spacy
from transformers import pipeline as hf_pipeline, GenerationConfig

from agentflow.processors.base import Processor
from agentflow.pipeline import Pipeline
from agentflow.typing.config import StageConfig

from ..types import DinoPrompt

# Module-level lazy singletons so the models load once per process.
_extractor = None
_nlp = None


def _get_models():
    global _extractor, _nlp
    if _extractor is None or _nlp is None:
        device = 0 if torch.cuda.is_available() else -1
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _extractor = hf_pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-1.5B-Instruct",
            device=device,
            torch_dtype=dtype,
        )
        _nlp = spacy.load("en_core_web_trf")
    return _extractor, _nlp


def _clean_question(text: str) -> str:
    return re.sub(
        r"fill in a number.*?into the mask\.?\s*", "", text, flags=re.IGNORECASE
    ).strip()


def _singularize(phrase: str, nlp) -> str:
    doc = nlp(phrase)
    tokens = list(doc)
    head_idx = next(
        (i for i in range(len(tokens) - 1, -1, -1) if tokens[i].pos_ in ("NOUN", "PROPN")),
        None,
    )
    if head_idx is None:
        return phrase.lower()
    result = []
    for i, token in enumerate(tokens):
        if token.pos_ in ("DET", "NUM") or token.text.lower() in ("a", "an", "the"):
            continue
        result.append(token.lemma_.lower() if i == head_idx else token.text.lower())
    return " ".join(result).strip()


def _extract_dino_prompt(question: str) -> str:
    extractor, nlp = _get_models()
    context = _clean_question(question)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise data extraction assistant for computer vision. "
                "Your task is to identify the primary, countable physical object in a caption. "
                "You must ignore abstract concepts, actions, and background elements"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Caption: {context}\n"
                "Extract the main countable object.\n\n"
                "Rules:\n"
                "- Return ONLY a short noun phrase\n"
                "- Include adjectives that describe the noun phrase\n"
                "- No articles or numbers\n"
                "- Only the object that is being counted\n"
                "Examples:\n"
                "- Caption: Three large red apples on a table. -> large red apple\n"
                "- Caption: A person wearing a blue waterproof jacket. -> blue waterproof jacket\n\n"
                "Object:\n"
            ),
        },
    ]
    gen_config = GenerationConfig(max_new_tokens=10, do_sample=False)
    result = extractor(messages, generation_config=gen_config)
    output = result[0]["generated_text"][-1]["content"].strip()
    output = re.sub(r"[^a-zA-Z0-9 ]", "", output).strip()
    if not output:
        doc = nlp(context.replace("[MASK]", "some"))
        output = next((chunk.text for chunk in doc.noun_chunks), "object")
    return _singularize(output, nlp) + " ."


class DinoPromptProcessor(Processor):
    def __call__(self, inputs: dict, logger=None, output_dir=None) -> DinoPrompt:
        question = inputs[self._input_names_snake[0]]
        prompt = _extract_dino_prompt(question)
        if logger:
            print(f"question: {question}", file=logger, flush=True)
            print(f"dino_prompt: {prompt}", file=logger, flush=True)
        return DinoPrompt(prompt=prompt)
