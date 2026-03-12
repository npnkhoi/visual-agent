"""PIL-based image utility tools: grid creation and box annotation."""
import json
import math
import os
import uuid
from typing import List

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


def make_image_grid(
    image_paths_json: str,
    output_dir: str = "tmp",
    columns: int = 5,
    cell_size: int = 224,
) -> str:
    """
    Create a grid image from a list of image paths.

    Returns JSON with the grid image path.
    """
    image_paths: List[str] = json.loads(image_paths_json)

    if not image_paths:
        return json.dumps({"error": "No image paths provided", "grid_image_path": None})

    cols = min(columns, len(image_paths))
    rows = math.ceil(len(image_paths) / cols)

    grid = Image.new("RGB", (cols * cell_size, rows * cell_size), color=(30, 30, 30))

    for idx, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((cell_size, cell_size), Image.LANCZOS)
        except Exception:
            img = Image.new("RGB", (cell_size, cell_size), color=(80, 80, 80))

        row = idx // cols
        col = idx % cols
        grid.paste(img, (col * cell_size, row * cell_size))

    grid_path = os.path.join(output_dir, f"grid_{uuid.uuid4().hex}.png")
    grid.save(grid_path)

    return json.dumps({"grid_image_path": grid_path, "num_images": len(image_paths)})


def annotate_boxes(
    image_path: str,
    indices_json: str,
    boxes_xyxy_json: str,
    output_dir: str = "tmp",
    color: str = "lime",
) -> str:
    """
    Draw colored bounding boxes for specified indices on an image copy.

    Returns JSON with the annotated image path.
    """
    indices: List[int] = json.loads(indices_json)
    all_boxes: List[List[float]] = json.loads(boxes_xyxy_json)

    image = Image.open(image_path).convert("RGB")
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for i in indices:
        if i < len(all_boxes):
            x0, y0, x1, y1 = all_boxes[i]
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            draw.text((x0 + 4, y0 + 2), str(i), fill=color, font=font)

    annotated_path = os.path.join(output_dir, f"annotated_{uuid.uuid4().hex}.png")
    annotated.save(annotated_path)

    return json.dumps({
        "annotated_image_path": annotated_path,
        "num_boxes_drawn": len(indices),
    })


class ImageGridInput(BaseModel):
    image_paths_json: str = Field(
        description="JSON array of absolute paths to images to tile into a grid"
    )
    output_dir: str = Field(
        description="Absolute path to directory where the grid image will be saved"
    )
    columns: int = Field(default=5, description="Number of columns in the grid")
    cell_size: int = Field(
        default=224, description="Width and height of each cell in pixels"
    )


class AnnotateBoxesInput(BaseModel):
    image_path: str = Field(description="Absolute path to the input image")
    indices_json: str = Field(
        description="JSON array of integer indices specifying which boxes to draw"
    )
    boxes_xyxy_json: str = Field(
        description="JSON array of [x0, y0, x1, y1] bounding boxes (all detections)"
    )
    output_dir: str = Field(
        description="Absolute path to directory where annotated image will be saved"
    )
    color: str = Field(
        default="lime",
        description="Color for the bounding boxes (e.g. 'lime', 'red', 'blue')",
    )


image_grid_tool = StructuredTool.from_function(
    func=make_image_grid,
    name="make_image_grid",
    description=(
        "Create a visual grid from multiple image files. "
        "Useful for displaying top search results side by side. "
        "Returns the path to the saved grid image."
    ),
    args_schema=ImageGridInput,
)

annotate_boxes_tool = StructuredTool.from_function(
    func=annotate_boxes,
    name="annotate_boxes",
    description=(
        "Draw colored bounding boxes on an image for specific detection indices. "
        "Use after CLIP verification to highlight only the verified/matched objects. "
        "Returns the path to the annotated image."
    ),
    args_schema=AnnotateBoxesInput,
)
