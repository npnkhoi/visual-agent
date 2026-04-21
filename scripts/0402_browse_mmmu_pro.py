"""
MMMU-Pro Dataset Browser
Browse examples from the MMMU-Pro benchmark: https://huggingface.co/datasets/MMMU/MMMU_Pro
"""

import ast
import streamlit as st
from datasets import load_dataset
from PIL import Image

st.set_page_config(
    page_title="MMMU-Pro Browser",
    page_icon="📚",
    layout="wide",
)

SUBSETS = [
    "standard (10 options)",
    "standard (4 options)",
    "vision",
]

ANSWER_LABELS = list("ABCDEFGHIJ")


@st.cache_resource(show_spinner="Loading dataset…")
def load_subset(name: str):
    return load_dataset("MMMU/MMMU_Pro", name=name)["test"]


@st.cache_data
def get_subjects(_ds) -> list[str]:
    return ["All"] + sorted(set(_ds["subject"]))


@st.cache_data
def get_difficulties(_ds) -> list[str]:
    if "topic_difficulty" not in _ds.column_names:
        return []
    return ["All"] + sorted(set(_ds["topic_difficulty"]))


@st.cache_data
def get_metadata(_ds):
    """Fetch only the lightweight metadata columns in one bulk pass."""
    cols = {"subject": _ds["subject"]}
    if "topic_difficulty" in _ds.column_names:
        cols["topic_difficulty"] = _ds["topic_difficulty"]
    return cols


@st.cache_data
def filter_dataset(_ds, subject: str, difficulty: str):
    meta = get_metadata(_ds)
    subjects_col = meta["subject"]
    difficulties_col = meta.get("topic_difficulty")
    indices = []
    for i in range(len(subjects_col)):
        if subject != "All" and subjects_col[i] != subject:
            continue
        if difficulty and difficulty != "All" and difficulties_col and difficulties_col[i] != difficulty:
            continue
        indices.append(i)
    return indices


def render_example(example: dict, subset: str, show_answer: bool):
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # ---- Images ----
        if subset == "vision":
            img = example.get("image")
            if img and isinstance(img, Image.Image):
                st.image(img, use_container_width=True)
        else:
            images = []
            for i in range(1, 8):
                img = example.get(f"image_{i}")
                if img and isinstance(img, Image.Image):
                    images.append(img)
            if images:
                img_cols = st.columns(min(len(images), 2))
                for j, img in enumerate(images):
                    img_cols[j % 2].image(img, use_container_width=True)

        # ---- Metadata ----
        st.markdown("---")
        st.write(f"**ID:** `{example.get('id', 'N/A')}`")
        st.write(f"**Subject:** {example.get('subject', 'N/A')}")
        if "topic_difficulty" in example and example["topic_difficulty"]:
            difficulty = example["topic_difficulty"]
            color = {"Easy": "green", "Medium": "orange", "Hard": "red"}.get(difficulty, "gray")
            st.markdown(f"**Difficulty:** :{color}[{difficulty}]")
        if "img_type" in example and example["img_type"]:
            st.write(f"**Image type:** {example['img_type']}")

    with col_right:
        # ---- Question ----
        st.markdown("### Question")
        st.markdown(example.get("question", ""))

        # ---- Options ----
        raw_options = example.get("options", [])
        if isinstance(raw_options, str):
            try:
                raw_options = ast.literal_eval(raw_options)
            except Exception:
                raw_options = [raw_options]
        options = raw_options if isinstance(raw_options, list) else []
        if options:
            st.markdown("### Options")
            answer = example.get("answer", "")
            for idx, opt in enumerate(options):
                label = ANSWER_LABELS[idx] if idx < len(ANSWER_LABELS) else str(idx)
                if show_answer and label == answer:
                    st.markdown(f"**:green[{label}. {opt}] ✓**")
                else:
                    st.markdown(f"{label}. {opt}")

        # ---- Answer / Explanation ----
        if show_answer:
            answer = example.get("answer", "?")
            st.markdown(f"---\n**Answer: {answer}**")
            explanation = example.get("explanation") or ""
            if explanation:
                st.markdown(f"**Explanation:**\n\n{explanation}")


# ---- Sidebar ----
st.sidebar.title("MMMU-Pro Browser")

subset = st.sidebar.selectbox("Subset", SUBSETS)
ds = load_subset(subset)

subjects = get_subjects(ds)
subject = st.sidebar.selectbox("Subject", subjects)

difficulties = get_difficulties(ds)
difficulty = ""
if difficulties:
    difficulty = st.sidebar.selectbox("Difficulty", difficulties)

filtered_indices = filter_dataset(ds, subject, difficulty)
total = len(filtered_indices)

st.sidebar.markdown(f"**{total}** examples match")

show_answer = st.sidebar.toggle("Show answer", value=False)

# ---- Navigation ----
if total == 0:
    st.warning("No examples match the current filters.")
    st.stop()

if "example_idx" not in st.session_state:
    st.session_state.example_idx = 0

# Clamp index after filter change
if st.session_state.example_idx >= total:
    st.session_state.example_idx = 0

st.sidebar.markdown("---")
nav_col1, nav_col2, nav_col3 = st.sidebar.columns([1, 2, 1])
with nav_col1:
    if st.button("◀", use_container_width=True):
        st.session_state.example_idx = (st.session_state.example_idx - 1) % total
with nav_col3:
    if st.button("▶", use_container_width=True):
        st.session_state.example_idx = (st.session_state.example_idx + 1) % total
with nav_col2:
    new_idx = st.number_input(
        "Go to",
        min_value=1,
        max_value=total,
        value=st.session_state.example_idx + 1,
        step=1,
        label_visibility="collapsed",
    )
    st.session_state.example_idx = int(new_idx) - 1

st.sidebar.markdown(f"Showing **{st.session_state.example_idx + 1}** / {total}")

# ---- Main ----
st.title("MMMU-Pro Benchmark Browser")

actual_idx = filtered_indices[st.session_state.example_idx]
example = ds[actual_idx]

render_example(example, subset, show_answer)
