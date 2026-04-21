import json
import re
from pathlib import Path
import streamlit as st

NUM_WORDS = {
    2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten",
}

DATASETS = {
    "CountBench Nouns (540)": "labels/CountBench_nouns.json",
    "CountBench Filtered (401)": "labels/CountBench_filtered.json",
    "CountBench Subset-18": "labels/CountBench_subset18.json",
}

def highlight_number_word(text, number):
    word = NUM_WORDS.get(number)
    if not word:
        return text
    return re.sub(rf"\b({re.escape(word)})\b", r"`\1`", text, flags=re.IGNORECASE)

def normalize(item):
    """Return a unified dict with keys: image_url, text, number, target_noun."""
    if "answer" in item:
        return {
            "image_url": item["image_url"],
            "text": item.get("question", ""),
            "number": item["answer"],
            "target_noun": item.get("target_noun", "—"),
            "question_id": item.get("question_id"),
        }
    return {
        "image_url": item["image_url"],
        "text": item.get("text", ""),
        "number": item["number"],
        "target_noun": item.get("target_noun", "—"),
        "question_id": None,
    }

st.set_page_config(page_title="CountBench Browser", layout="wide")

@st.cache_data
def load_data(path):
    with open(path) as f:
        raw = json.load(f)
    return [normalize(item) for item in raw]

@st.cache_data
def load_local_image_indices():
    data_dir = Path("data")
    if not data_dir.exists():
        return set()
    return {int(p.stem) for p in data_dir.iterdir() if p.is_file() and p.stem.isdigit()}

# Sidebar — dataset selector first
with st.sidebar:
    st.header("Dataset")
    dataset_name = st.selectbox("Choose dataset", list(DATASETS.keys()))

data = load_data(DATASETS[dataset_name])
local_image_indices = load_local_image_indices()

st.title("CountBench Dataset Browser")
st.caption(f"**{dataset_name}** — {len(data)} entries")

with st.sidebar:
    st.header("Filters")
    count_filter = st.multiselect(
        "Count value",
        options=list(range(2, 11)),
        default=list(range(2, 11)),
    )
    search = st.text_input("Search text", placeholder="e.g. dog, chair...")
    only_local_images = st.checkbox("Only show tasks with local image", value=False)
    cols_per_row = st.slider("Columns", 1, 5, 3)
    st.divider()
    st.header("Jump to index")
    jump_idx = st.number_input("Index", min_value=0, max_value=len(data)-1, value=0, step=1)

# Filter
filtered = [
    (i, d) for i, d in enumerate(data)
    if d["number"] in count_filter
    and (not search or search.lower() in d["text"].lower())
    and (not only_local_images or i in local_image_indices)
]

st.write(f"**{len(filtered)}** matching entries")

if not filtered:
    st.warning("No entries match the current filters.")
    st.stop()

# Pagination
page_size = cols_per_row * 4
total_pages = max(1, (len(filtered) + page_size - 1) // page_size)

if "page" not in st.session_state:
    st.session_state.page = 0
st.session_state.page = min(max(int(st.session_state.page), 0), total_pages - 1)

if "page_input" not in st.session_state:
    st.session_state.page_input = st.session_state.page + 1
st.session_state.page_input = min(max(int(st.session_state.page_input), 1), total_pages)

st.number_input("Page", min_value=1, max_value=total_pages, step=1, key="page_input")
st.session_state.page = st.session_state.page_input - 1

page = st.session_state.page
start = page * page_size
batch = filtered[start: start + page_size]

# Grid
rows = [batch[i: i + cols_per_row] for i in range(0, len(batch), cols_per_row)]
for row in rows:
    cols = st.columns(cols_per_row)
    for col, (orig_idx, item) in zip(cols, row):
        with col:
            try:
                st.image(item["image_url"], use_container_width=True)
            except Exception:
                st.error("Image unavailable")
            qid = f" · QID `{item['question_id']}`" if item["question_id"] is not None else ""
            st.markdown(
                f"**Index:** `#{orig_idx}`{qid}  \n"
                f"**Text:** {highlight_number_word(item['text'], item['number'])}  \n"
                f"**Count:** `{item['number']}`  \n"
                f"**Target noun:** `{item['target_noun']}`",
                unsafe_allow_html=True,
            )

# Detail view
st.divider()
st.subheader("Detail view")
detail_idx = st.number_input(
    "Entry index (0-based)", min_value=0, max_value=len(data)-1,
    value=int(jump_idx), step=1, key="detail"
)
item = data[detail_idx]
c1, c2 = st.columns([2, 1])
with c1:
    try:
        st.image(item["image_url"], use_container_width=True)
    except Exception:
        st.error("Image unavailable")
with c2:
    st.write("**Index:**", f"`#{detail_idx}`")
    if item["question_id"] is not None:
        st.write("**Question ID:**", f"`{item['question_id']}`")
    st.write("**Text:**", highlight_number_word(item["text"], item["number"]))
    st.write("**Count:**", f"`{item['number']}`")
    st.write("**Target noun:**", f"`{item['target_noun']}`")
    st.write("**Image URL:**")
    st.code(item["image_url"], language=None)
