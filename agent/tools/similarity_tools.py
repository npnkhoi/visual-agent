"""CLIP-based similarity tools for verification and ranking."""
import json
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from .model_registry import registry


def _to_tensor(features) -> torch.Tensor:
    """Unwrap BaseModelOutputWithPooling → tensor if needed (transformers 5.x)."""
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "last_hidden_state"):
        return features.last_hidden_state[:, 0, :]
    raise TypeError(f"Cannot extract tensor from {type(features)}")


def clip_verify_crops(
    crop_paths_json: str,
    text_query: str = "object",
    threshold: float = 0.15,
) -> str:
    """
    Verify crop images against a text query using CLIP cosine similarity.

    Returns JSON with verified indices, similarities, and count.
    """
    crop_paths: List[str] = json.loads(crop_paths_json)

    model = registry.clip_model
    processor = registry.clip_processor
    device = registry.device

    # Encode text
    text_inputs = processor(text=[text_query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = F.normalize(_to_tensor(model.get_text_features(**text_inputs)), dim=-1)

    similarities = []
    for crop_path in crop_paths:
        try:
            image = Image.open(crop_path).convert("RGB")
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = F.normalize(_to_tensor(model.get_image_features(**image_inputs)), dim=-1)
            sim = (image_features @ text_features.T).item()
        except Exception:
            sim = 0.0
        similarities.append(sim)

    verified_indices = [i for i, s in enumerate(similarities) if s >= threshold]

    return json.dumps({
        "verified_indices": verified_indices,
        "similarities": similarities,
        "verified_count": len(verified_indices),
    })


def clip_rank_by_pattern(
    crop_paths_json: str,
    pattern_image_path: str,
    top_k: int = 5,
) -> str:
    """
    Rank crop images by visual similarity to a pattern image using CLIP.

    Returns JSON with ranked results and top-k paths.
    """
    crop_paths: List[str] = json.loads(crop_paths_json)

    model = registry.clip_model
    processor = registry.clip_processor
    device = registry.device

    # Encode pattern image
    pattern = Image.open(pattern_image_path).convert("RGB")
    pattern_inputs = processor(images=pattern, return_tensors="pt").to(device)
    with torch.no_grad():
        pattern_features = F.normalize(_to_tensor(model.get_image_features(**pattern_inputs)), dim=-1)

    # Encode all crops
    results = []
    for i, crop_path in enumerate(crop_paths):
        try:
            crop = Image.open(crop_path).convert("RGB")
            crop_inputs = processor(images=crop, return_tensors="pt").to(device)
            with torch.no_grad():
                crop_features = F.normalize(_to_tensor(model.get_image_features(**crop_inputs)), dim=-1)
            sim = (crop_features @ pattern_features.T).item()
        except Exception:
            sim = 0.0
        results.append({"index": i, "crop_path": crop_path, "similarity": sim})

    # Sort descending by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)

    k = min(top_k, len(results))
    ranked = [
        {"rank": rank + 1, "index": r["index"], "crop_path": r["crop_path"], "similarity": r["similarity"]}
        for rank, r in enumerate(results[:k])
    ]

    return json.dumps({
        "ranked": ranked,
        "top_k_paths": [r["crop_path"] for r in ranked],
    })


class CLIPVerifyInput(BaseModel):
    crop_paths_json: str = Field(
        description="JSON array of absolute paths to crop image files to verify"
    )
    text_query: str = Field(
        default="object",
        description="Text description to match against (e.g., 'a cat', 'a red car'). Required — use the object name from the user's question.",
    )


class CLIPRankInput(BaseModel):
    crop_paths_json: str = Field(
        description="JSON array of absolute paths to crop image files to rank"
    )
    pattern_image_path: str = Field(
        description="Absolute path to the reference/pattern image to match against"
    )
    top_k: int = Field(
        default=5,
        description="Number of top matching crops to return",
    )


clip_verify_tool = StructuredTool.from_function(
    func=clip_verify_crops,
    name="clip_verify_crops",
    description=(
        "Verify detected crop images against a text description using CLIP. "
        "Filters out false positives from object detection. "
        "Returns verified indices, similarity scores, and count. "
        "Similarity >0.25 means the crop matches the text description."
    ),
    args_schema=CLIPVerifyInput,
)

clip_rank_tool = StructuredTool.from_function(
    func=clip_rank_by_pattern,
    name="clip_rank_by_pattern",
    description=(
        "Rank detected crop images by visual similarity to a pattern/reference image using CLIP. "
        "Used for person/object search tasks. "
        "Returns crops sorted by similarity score. "
        "Similarity >0.75 = high confidence match, 0.50-0.75 = possible match, <0.50 = not found."
    ),
    args_schema=CLIPRankInput,
)
