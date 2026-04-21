import json
import os
import re
import glob
import streamlit as st

st.set_page_config(page_title="CountBench Results Browser", layout="wide")

RESULTS_DIR = "results"
LABELS_PATH = "labels/CountBench_nouns.json"
DATA_DIR = "data"

NUM_WORDS = {
    2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten",
}

SYSTEMS = {
    "system1": "S1: Agent (MiniMax)",
    "system2": "S2: GDINO + CLIP",
    "system3": "S3: GDINO only",
}


def highlight_number_word(text, number):
    word = NUM_WORDS.get(number)
    if not word:
        return text
    return re.sub(rf"\b({re.escape(word)})\b", r"`\1`", text, flags=re.IGNORECASE)


@st.cache_data
def load_dataset():
    with open(LABELS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_results():
    """Load all per-item result JSONs into a dict: {sys_key: {idx: record}}"""
    results = {}
    for sys_key in SYSTEMS:
        sys_dir = os.path.join(RESULTS_DIR, sys_key)
        sys_results = {}
        for path in glob.glob(os.path.join(sys_dir, "*.json")):
            idx = int(os.path.splitext(os.path.basename(path))[0])
            with open(path) as f:
                sys_results[idx] = json.load(f)
        results[sys_key] = sys_results
    return results


@st.cache_data
def load_metrics():
    path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def find_image(idx):
    matches = glob.glob(os.path.join(DATA_DIR, f"{idx:04d}.*"))
    return matches[0] if matches else None


def verdict(predicted, ground_truth):
    if predicted == -1:
        return "error", "⚠ error"
    if predicted == ground_truth:
        return "correct", f"✓ {predicted}"
    return "wrong", f"✗ {predicted}"


dataset = load_dataset()
all_results = load_results()
metrics = load_metrics()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    count_filter = st.multiselect("Count value", list(range(2, 11)), default=list(range(2, 11)))

    verdict_filter = st.multiselect(
        "Result",
        ["correct", "wrong", "error", "no result"],
        default=["correct", "wrong", "error", "no result"],
    )

    system_for_filter = st.selectbox(
        "Apply result filter using", list(SYSTEMS.keys()),
        format_func=lambda k: SYSTEMS[k],
    )

    search = st.text_input("Search text / noun", placeholder="e.g. dog")

    st.divider()
    st.header("Summary metrics")
    for sys_key, label in SYSTEMS.items():
        m = metrics.get(sys_key)
        if m:
            st.markdown(f"**{label}**")
            col_a, col_b = st.columns(2)
            col_a.metric("Exact acc", f"{m['exact_acc']}%")
            col_b.metric("MAE", f"{m['MAE']}")
            st.caption(f"n={m['n']}  parse_failures={m['parse_failures']}")
        else:
            st.markdown(f"**{label}** — no metrics")

# ── Build index of entries that have any result ────────────────────────────────
# Collect all indices that have a result in any system
all_indices = sorted(
    set().union(*[set(r.keys()) for r in all_results.values()])
)

# Apply filters
def get_verdict_for(idx):
    rec = all_results[system_for_filter].get(idx)
    if rec is None:
        return "no result"
    return verdict(rec["predicted"], rec["ground_truth"])[0]

filtered_indices = []
for idx in all_indices:
    if idx >= len(dataset):
        continue
    item = dataset[idx]
    if item["number"] not in count_filter:
        continue
    if search and search.lower() not in item["text"].lower() and search.lower() not in item.get("target_noun", "").lower():
        continue
    v = get_verdict_for(idx)
    if v not in verdict_filter:
        continue
    filtered_indices.append(idx)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("CountBench Results Browser")
st.caption(f"{len(filtered_indices)} matching entries (of {len(all_indices)} evaluated)")

if not filtered_indices:
    st.warning("No entries match the current filters.")
    st.stop()

# ── Pagination ─────────────────────────────────────────────────────────────────
PAGE_SIZE = 10
total_pages = max(1, (len(filtered_indices) + PAGE_SIZE - 1) // PAGE_SIZE)

col_prev, col_page, col_next = st.columns([1, 3, 1])
with col_prev:
    if st.button("← Prev"):
        st.session_state.page = max(0, st.session_state.get("page", 0) - 1)
with col_next:
    if st.button("Next →"):
        st.session_state.page = min(total_pages - 1, st.session_state.get("page", 0) + 1)
with col_page:
    page = st.number_input("Page", min_value=1, max_value=total_pages,
                           value=st.session_state.get("page", 0) + 1, step=1) - 1
    st.session_state.page = page

batch = filtered_indices[page * PAGE_SIZE: (page + 1) * PAGE_SIZE]

# ── Per-entry rows ─────────────────────────────────────────────────────────────
for idx in batch:
    item = dataset[idx]
    image_path = find_image(idx)

    with st.container(border=True):
        img_col, info_col = st.columns([1, 2])

        with img_col:
            if image_path:
                st.image(image_path, use_container_width=True)
            else:
                try:
                    st.image(item["image_url"], use_container_width=True)
                except Exception:
                    st.error("Image unavailable")

        with info_col:
            st.markdown(
                f"**#{idx}** &nbsp;|&nbsp; GT count: **{item['number']}**  \n"
                f"{highlight_number_word(item['text'], item['number'])}  \n"
                f"Noun: `{item.get('target_noun', '—')}`"
            )

            st.divider()

            sys_cols = st.columns(len(SYSTEMS))
            for col, (sys_key, label) in zip(sys_cols, SYSTEMS.items()):
                rec = all_results[sys_key].get(idx)
                with col:
                    st.markdown(f"**{label}**")
                    if rec is None:
                        st.caption("no result")
                        continue

                    kind, text = verdict(rec["predicted"], rec["ground_truth"])
                    color = {"correct": "green", "wrong": "red", "error": "orange"}[kind]
                    st.markdown(f":{color}[{text}] / GT {rec['ground_truth']}")

                    # System-specific detail
                    if sys_key == "system1":
                        with st.expander("response"):
                            st.write(rec.get("agent_response", "—"))
                    elif sys_key in ("system2", "system3"):
                        gdino = rec.get("gdino_output", {})
                        st.caption(f"GDINO detections: {gdino.get('num_detections', '?')}")
                        if sys_key == "system2":
                            clip = rec.get("clip_output", {})
                            st.caption(f"CLIP verified: {clip.get('verified_count', '?')}")
                        scores = gdino.get("scores", [])
                        if scores:
                            with st.expander("scores"):
                                st.write([round(s, 3) for s in scores])
