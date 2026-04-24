"""Pipeline output viewer — run with:

    streamlit run pipelines/viewer.py

Auto-scans pipelines/configs/ and lets you choose a pipeline interactively.
"""
from __future__ import annotations

import colorsys
import json
from pathlib import Path

import numpy as np
import streamlit as st
import yaml
from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_CONFIGS = _HERE / "configs"


# ---------------------------------------------------------------------------
# Helpers (standalone — no agentflow imports)
# ---------------------------------------------------------------------------

def _camel_to_snake(name: str) -> str:
    import re
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _decode_rle(counts: list[int], size: list[int]) -> np.ndarray:
    """Decode run-length encoding to a boolean H×W mask."""
    H, W = size
    flat = np.zeros(H * W, dtype=bool)
    idx, val = 0, False
    for c in counts:
        flat[idx: idx + c] = val
        idx += c
        val = not val
    return flat.reshape(H, W)


def _palette(n: int) -> list[tuple[int, int, int]]:
    """Generate n visually distinct RGB colours."""
    return [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / max(n, 1), 0.75, 0.92))
        for i in range(n)
    ]


def _visual_type(data: dict) -> str | None:
    """Detect whether a JSON blob contains renderable visual data."""
    if (
        isinstance(data.get("boxes"), list)
        and data["boxes"]
        and isinstance(data["boxes"][0], list)
        and len(data["boxes"][0]) == 4
    ):
        return "boxes"
    if (
        isinstance(data.get("masks"), list)
        and data["masks"]
        and isinstance(data["masks"][0], dict)
        and "rle" in data["masks"][0]
    ):
        return "masks"
    if (
        isinstance(data.get("points"), list)
        and data["points"]
        and isinstance(data["points"][0], list)
        and len(data["points"][0]) == 2
    ):
        return "points"
    if isinstance(data.get("path"), str) and Path(data["path"]).is_file():
        return "labeled_image"
    return None


def _render_boxes(img_path: Path, data: dict) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    boxes = data["boxes"]
    labels = data.get("labels") or [""] * len(boxes)
    scores = data.get("scores") or [None] * len(boxes)
    colors = _palette(len(boxes))
    for box, label, score, color in zip(boxes, labels, scores, colors):
        x1, y1, x2, y2 = (int(v) for v in box)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        txt = label or ""
        if score is not None:
            txt += f" {score:.2f}"
        if txt.strip():
            # Small background for readability
            tw, th = draw.textlength(txt), 12
            draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - th - 2), txt, fill="white")
    return img


def _render_masks(img_path: Path, data: dict) -> Image.Image:
    img = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    masks = data["masks"]
    colors = _palette(len(masks))
    for mask_item, color in zip(masks, colors):
        rle = mask_item["rle"]
        arr = _decode_rle(rle["counts"], rle["size"])  # bool H×W
        h, w = arr.shape
        mask_pil = Image.fromarray((arr.astype(np.uint8) * 255))
        if (w, h) != img.size:
            mask_pil = mask_pil.resize(img.size, Image.NEAREST)
        rgba = (*color, 140)  # semi-transparent
        colored = Image.new("RGBA", img.size, rgba)
        overlay.paste(colored, mask=mask_pil)
    return Image.alpha_composite(img, overlay).convert("RGB")


def _render_points(img_path: Path, data: dict) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    colors = _palette(len(data["points"]))
    r = max(6, min(img.width, img.height) // 80)
    for (x, y), color in zip(data["points"], colors):
        x, y = int(x), int(y)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="white", width=2)
    return img


# ---------------------------------------------------------------------------
# Dialog definitions (must be at module level for st.dialog)
# ---------------------------------------------------------------------------

@st.dialog("Overlay", width="large")
def _show_overlay(title: str, rendered: Image.Image) -> None:
    st.subheader(title)
    st.image(rendered, use_container_width=True)


# ---------------------------------------------------------------------------
# Main viewer logic
# ---------------------------------------------------------------------------

def _scan_configs() -> list[Path]:
    return sorted(_CONFIGS.glob("*.yaml"))


def _build_meta(config_path: Path) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    pipeline_name = raw["name"]
    return {
        "name": pipeline_name,
        "output_root": str(_ROOT / "output" / pipeline_name),
        "loader_source": str(_ROOT / raw["loader"]["source"]),
        "image_dir": str(_ROOT / raw["loader"]["kwargs"]["image_dir"]),
        "stages": [
            {
                "name": s["output"],
                "processor": s.get("processor", ""),
                "inputs": [{"type": t, "source": src} for t, src in s["inputs"]],
            }
            for s in raw["stages"]
        ],
    }


def _show_visual_buttons(
    data: dict,
    vtype: str,
    img_path: Path | None,
    key_prefix: str,
) -> None:
    """Render the overlay button(s) for a JSON blob that has visual data."""
    if vtype == "labeled_image":
        st.image(data["path"], use_container_width=True)
        return

    if img_path is None or not img_path.exists():
        st.caption("_(image not found for overlay)_")
        return

    if vtype == "boxes":
        if st.button("Show Boxes", key=f"{key_prefix}_boxes"):
            _show_overlay("Bounding Boxes", _render_boxes(img_path, data))

    elif vtype == "masks":
        if st.button("Show Masks", key=f"{key_prefix}_masks"):
            _show_overlay("Segmentation Masks", _render_masks(img_path, data))

    elif vtype == "points":
        if st.button("Show Points", key=f"{key_prefix}_points"):
            _show_overlay("Object Centers", _render_points(img_path, data))


def _display_json_with_overlay(
    data: dict,
    img_path: Path | None,
    key_prefix: str,
    expanded: bool = True,
) -> None:
    """Show st.json and, if the data is visual, an overlay button below it."""
    st.json(data, expanded=expanded)
    vtype = _visual_type(data)
    if vtype:
        _show_visual_buttons(data, vtype, img_path, key_prefix)


def _pipeline_dot(meta: dict, output_root: Path, item_ids: list, total: int) -> str:
    stages = meta["stages"]
    human_types: set[str] = set()
    model_types: set[str] = set()
    edges: list[tuple[str, str]] = []

    for s in stages:
        out = s["name"]
        model_types.add(out)
        for inp in s["inputs"]:
            t, src = inp["type"], inp["source"]
            edges.append((t, out))
            if src == "human":
                human_types.add(t)
            else:
                model_types.add(t)

    lines = ['digraph { rankdir=LR node [fontname="Helvetica" fontsize=12]']

    for t in human_types:
        lines.append(f'  "{t}" [shape=ellipse style=filled fillcolor="#AED6F1"]')

    processor_for: dict[str, str] = {s["name"]: s.get("processor", "") for s in stages}

    for t in model_types:
        n_done = sum(
            1 for iid in item_ids
            if (output_root / t / iid / "output.json").exists()
        ) if item_ids else 0
        proc = processor_for.get(t, "")
        label = f"{t}\\n{proc}\\n{n_done}/{total}"
        lines.append(f'  "{t}" [shape=box style=filled fillcolor="#A9DFBF" label="{label}"]')

    for src, dst in edges:
        lines.append(f'  "{src}" -> "{dst}"')

    lines.append("}")
    return "\n".join(lines)


def _overview_view(configs: list[Path]) -> None:
    st.title("Pipelines")
    for cfg_path in configs:
        meta = _build_meta(cfg_path)
        output_root = Path(meta["output_root"])
        loader_source = Path(meta["loader_source"])

        try:
            items = json.loads(loader_source.read_text(encoding="utf-8"))
            total = len(items)
            item_ids = [it["id"] for it in items]
        except Exception:
            total = "?"
            item_ids = []

        with st.container(border=True):
            st.markdown(f"### {meta['name']}")
            dot = _pipeline_dot(meta, output_root, item_ids, total)
            st.graphviz_chart(dot, use_container_width=True)


def main() -> None:
    st.set_page_config(layout="wide", page_title="Pipeline Viewer")

    configs = _scan_configs()
    if not configs:
        st.error(f"No pipeline configs found in {_CONFIGS}")
        st.stop()

    tab_overview, tab_inspect = st.tabs(["Overview", "Inspect"])

    with tab_overview:
        _overview_view(configs)

    with tab_inspect:
        config_names = [p.name for p in configs]
        chosen = st.selectbox("Pipeline", config_names)
        meta = _build_meta(_CONFIGS / chosen)

        output_root = Path(meta["output_root"])
        loader_source = Path(meta["loader_source"])
        image_dir = Path(meta["image_dir"])
        stages = meta["stages"]

        st.title(f"Pipeline: {meta['name']}")

        if not loader_source.exists():
            st.error(f"Loader source not found: {loader_source}")
            st.stop()

        items = json.loads(loader_source.read_text(encoding="utf-8"))
        item_ids = [item["id"] for item in items]
        item_data = {item["id"]: item["data"] for item in items}

        # --- Stage / Item selectors ---
        col_stage, col_item = st.columns(2)

        with col_stage:
            stage_idx = int(st.number_input(
                "Stage", min_value=0, max_value=len(stages) - 1, step=1, key="stage_idx",
            ))
            stage = stages[stage_idx]
            stage_name = stage["name"]
            stage_output_dir = output_root / stage_name
            n_done = sum(1 for iid in item_ids if (stage_output_dir / iid / "output.json").exists())
            st.caption(f"**{stage_name}** · {n_done}/{len(item_ids)} done")

        with col_item:
            item_idx = int(st.number_input(
                "Item", min_value=0, max_value=len(item_ids) - 1, step=1, key="item_idx",
            ))
            item_id = item_ids[item_idx]
            st.caption(f"**Item {item_idx}** · ID: `{item_id}`")
        st.divider()

        # Resolve the image path from local image_dir only — strip any URL to filename
        data_for_item = item_data.get(item_id, {})
        img_val = data_for_item.get("image")
        if img_val:
            filename = Path(str(img_val).split("/")[-1])
            img_path: Path | None = image_dir / filename
        else:
            img_path = None

        # --- Two-column display ---
        left, right = st.columns(2)

        with left:
            st.subheader("Inputs")
            for inp in stage["inputs"]:
                input_type = inp["type"]
                input_source = inp["source"]
                st.markdown(f"**{input_type}** *(from {input_source})*")

                if input_source == "human":
                    field = _camel_to_snake(input_type)
                    val = data_for_item.get(field)
                    if val is None:
                        st.warning("Not found in loader data.")
                    elif input_type == "Image":
                        if img_path and img_path.exists():
                            st.image(str(img_path), use_container_width=True)
                        else:
                            st.warning(f"Image not found locally: {img_path}")
                    elif isinstance(val, (dict, list)):
                        st.json(val, expanded=True)
                    else:
                        st.text(str(val))
                else:
                    cache_file = output_root / input_type / item_id / "output.json"
                    if cache_file.exists():
                        data = json.loads(cache_file.read_text(encoding="utf-8"))
                        key = f"inp_{stage_name}_{item_id}_{input_type}"
                        _display_json_with_overlay(data, img_path, key)
                    else:
                        st.warning(f"Stage output not found: {cache_file}")

        with right:
            st.subheader(f"Output: {stage_name}")
            output_file = stage_output_dir / item_id / "output.json"
            if output_file.exists():
                data = json.loads(output_file.read_text(encoding="utf-8"))
                key = f"out_{stage_name}_{item_id}"
                _display_json_with_overlay(data, img_path, key)
            else:
                st.warning("Not executed, failed, or cached from a previous run.")

            log_file = stage_output_dir / item_id / "run.log"
            if log_file.exists():
                log_text = log_file.read_text(encoding="utf-8").strip()
                if log_text:
                    with st.expander("run.log"):
                        st.code(log_text, language="text", wrap_lines=True)


if __name__ == "__main__":
    main()
