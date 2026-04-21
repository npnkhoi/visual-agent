"""Generic VLM inference backend.

Supports multiple model families via a unified interface:
  - Qwen2.5-VL  (uses qwen_vl_utils for image preprocessing)
  - Gemma 4 / any other HF vision model (uses AutoModelForImageTextToText + PIL)

Usage:
    model, processor = load_vlm(model_id)
    response = run_vlm(image_path, question, model, processor, model_id, max_new_tokens=32)
"""
from __future__ import annotations

import torch
from pathlib import Path

# model_id -> (model, processor)
_model_cache: dict[str, tuple] = {}


def _is_qwen(model_id: str) -> bool:
    return "qwen" in model_id.lower()


def load_vlm(model_id: str) -> tuple:
    if model_id in _model_cache:
        return _model_cache[model_id]

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if _is_qwen(model_id):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        proc = AutoProcessor.from_pretrained(model_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )

    model.eval()
    _model_cache[model_id] = (model, proc)
    return model, proc


def run_vlm(
    image_path: Path,
    question: str,
    model,
    processor,
    model_id: str,
    max_new_tokens: int = 32,
) -> str:
    """Run a single image+text query and return the model's text response."""
    if _is_qwen(model_id):
        from qwen_vl_utils import process_vision_info
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": question},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            return_tensors="pt", padding=True,
        ).to(model.device)
    else:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        # Use chat template if available, otherwise fall back to Gemma instruction format
        if getattr(processor, "chat_template", None):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }]
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            # Gemma-style raw instruction format with the correct image placeholder token
            img_token = getattr(processor.tokenizer, "special_tokens_map", {}).get("image_token", "<image>")
            text = (
                f"<bos><start_of_turn>user\n"
                f"{img_token}\n"
                f"{question}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
        inputs = processor(
            images=[img], text=text, return_tensors="pt",
        ).to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    output_ids = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
