"""Streamlit UI for the visual agent system."""
import json
import logging
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

load_dotenv()

_log = logging.getLogger(__name__)

st.set_page_config(
    page_title="Visual Agent",
    page_icon="👁️",
    layout="wide",
)


# ── Event system ───────────────────────────────────────────────────────────────

def _tool_input_summary(tool_name: str, inputs: dict) -> str:
    if tool_name == "grounding_dino_detect":
        return inputs.get("text_prompt", "")
    if tool_name == "clip_verify_crops":
        return f"'{inputs.get('text_query', '')}'"
    if tool_name == "clip_rank_by_pattern":
        return f"top {inputs.get('top_k', 5)}"
    if tool_name == "make_image_grid":
        try:
            n = len(json.loads(inputs.get("image_paths_json", "[]")))
        except Exception:
            n = "?"
        return f"{n} images"
    if tool_name == "annotate_boxes":
        try:
            n = len(json.loads(inputs.get("indices_json", "[]")))
        except Exception:
            n = "?"
        return f"{n} boxes"
    return ""


def _tool_result_summary(tool_name: str, result: str) -> str:
    try:
        data = json.loads(result)
        if tool_name == "grounding_dino_detect":
            return f"{data.get('num_detections', '?')} detections"
        if tool_name == "clip_verify_crops":
            total = len(data.get("similarities", []))
            return f"{data.get('verified_count', '?')}/{total} verified"
        if tool_name == "clip_rank_by_pattern":
            return f"{len(data.get('ranked', []))} ranked"
        if tool_name == "make_image_grid":
            return "grid created"
        if tool_name == "annotate_boxes":
            return f"{data.get('num_boxes_drawn', '?')} boxes drawn"
    except Exception:
        pass
    return "done"


TOOL_IMAGE_KEYS = {
    "grounding_dino_detect": ("annotated_image_path", "All detections (Grounding DINO)"),
    "annotate_boxes":        ("annotated_image_path", "Verified detections (CLIP)"),
    "make_image_grid":       ("grid_image_path",      "Top matches (CLIP similarity)"),
}


def render_event(ev: dict):
    """Render a single stored event as a Streamlit widget."""
    t = ev["type"]
    if t == "llm_think":
        st.caption(f"🤔 `{ev['model']}` thinking...")
    elif t == "model_try":
        st.caption(f"⏳ Trying `{ev['model']}`...")
    elif t == "model_fail":
        st.caption(f"❌ `{ev['model']}` — {ev['reason']}")
    elif t == "model_ok":
        st.caption(f"✅ `{ev['model']}`")
    elif t == "tool_call":
        summary = ev.get("summary", "")
        label = f"🔧 `{ev['tool']}`" + (f" — {summary}" if summary else "")
        with st.expander(label, expanded=False):
            st.code(ev.get("args", ""), language="json")
    elif t == "tool_result":
        summary = ev.get("summary", "")
        with st.expander(f"📤 `{ev['tool']}` → {summary}", expanded=False):
            st.code(ev.get("result", ""), language="json")
    elif t == "answer":
        st.markdown(ev["text"])
    elif t == "image":
        path = ev.get("path", "")
        if path and os.path.exists(path):
            st.caption(ev.get("label", ""))
            st.image(path, width=480)


class AgentLogger:
    """Emits events to Streamlit live and stores them for persistent replay."""

    def __init__(self):
        self.events: list[dict] = []
        self._container = st.container()

    def _emit(self, ev: dict):
        self.events.append(ev)
        with self._container:
            render_event(ev)

    def model_trying(self, model_id: str):
        self._emit({"type": "model_try", "model": model_id})

    def model_failed(self, model_id: str, reason: str):
        self._emit({"type": "model_fail", "model": model_id, "reason": reason})

    def model_ok(self, model_id: str):
        self._emit({"type": "model_ok", "model": model_id})

    def answer(self, text: str):
        self._emit({"type": "answer", "text": text})

    def image(self, path: str, label: str):
        self._emit({"type": "image", "path": path, "label": label})


class ToolEventCallback(BaseCallbackHandler):
    """LangChain callback that logs tool calls and results into AgentLogger."""

    def __init__(self, logger: AgentLogger):
        super().__init__()
        self._logger = logger
        self._tool_stack: list[str] = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        model = serialized.get("name") or serialized.get("id", ["", ""])[-1]
        self._logger._emit({"type": "llm_think", "model": model})

    def on_tool_start(self, serialized, input_str, *, inputs=None, **kwargs):
        tool_name = serialized.get("name", "tool")
        self._tool_stack.append(tool_name)
        inputs_dict = inputs or {}
        if not inputs_dict and input_str:
            try:
                inputs_dict = json.loads(input_str)
            except Exception:
                inputs_dict = {"input": str(input_str)}
        summary = _tool_input_summary(tool_name, inputs_dict)
        self._logger._emit({
            "type": "tool_call",
            "tool": tool_name,
            "summary": summary,
            "args": json.dumps(inputs_dict, indent=2),
        })

    def on_tool_end(self, output, **kwargs):
        tool_name = self._tool_stack.pop() if self._tool_stack else "tool"
        result_str = output if isinstance(output, str) else json.dumps(output)
        summary = _tool_result_summary(tool_name, result_str)
        self._logger._emit({
            "type": "tool_result",
            "tool": tool_name,
            "summary": summary,
            "result": result_str[:2000],
        })
        # Emit image if the tool produced one
        if tool_name in TOOL_IMAGE_KEYS:
            img_key, img_label = TOOL_IMAGE_KEYS[tool_name]
            try:
                path = json.loads(result_str).get(img_key)
                if path and os.path.exists(path):
                    self._logger.image(path, img_label)
            except Exception:
                pass

    def on_tool_error(self, error, **kwargs):
        tool_name = self._tool_stack.pop() if self._tool_stack else "tool"
        self._logger._emit({
            "type": "tool_result",
            "tool": tool_name,
            "summary": "ERROR",
            "result": str(error),
        })


# ── Model loading ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading vision models (first run may take a few minutes)...")
def load_models():
    from agent.tools.model_registry import registry
    registry.ensure_all()
    return registry


def save_upload(uploaded_file, output_dir: str) -> str:
    dest = os.path.join(output_dir, uploaded_file.name)
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


# ── Session state ──────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp(prefix="visual_agent_", dir=TMP_DIR)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    task_mode = st.radio(
        "Task mode",
        options=["counting", "search"],
        format_func=lambda x: "🔢 Object Counting" if x == "counting" else "🔍 Object Search",
    )

    st.divider()
    st.subheader("Upload Images")

    scene_file = st.file_uploader(
        "Scene image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        key="scene_uploader",
        help="The main image to analyze",
    )
    if scene_file:
        st.image(scene_file, caption="Scene image", use_container_width=True)

    pattern_file = None
    if task_mode == "search":
        pattern_file = st.file_uploader(
            "Reference / pattern image",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key="pattern_uploader",
            help="The person or object to search for",
        )
        if pattern_file:
            st.image(pattern_file, caption="Reference image", use_container_width=True)

    st.divider()
    if st.button("🗑️ Clear session", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.temp_dir = tempfile.mkdtemp(prefix="visual_agent_", dir=TMP_DIR)
        st.rerun()

    api_key = os.environ.get("MINIMAX_API_KEY", "")
    if api_key and api_key != "your-minimax-api-key":
        st.success("✅ API key loaded")
    else:
        st.error("❌ MINIMAX_API_KEY not set — copy .env.example → .env and fill it in")

# ── Model loading ──────────────────────────────────────────────────────────────
load_models()

# ── Main area ──────────────────────────────────────────────────────────────────
st.title("👁️ Visual Agent")
from agent.agent import FALLBACK_MODELS as _FALLBACK_MODELS
_default_model = os.environ.get("MODEL_ID", _FALLBACK_MODELS[0])
st.caption(
    f"Mode: **{'Object Counting' if task_mode == 'counting' else 'Object Search'}** | "
    f"LLM: `{_default_model}` | Detection: Grounding DINO | Verification: CLIP ViT-L/14"
)

# ── Replay chat history ────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            for ev in msg.get("events", []):
                render_event(ev)

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    placeholder=(
        "e.g. 'How many cats are in the image?'"
        if task_mode == "counting"
        else "e.g. 'Find the person in the red jacket'"
    )
)

if user_input:
    if scene_file is None:
        st.error("Please upload a scene image first.")
        st.stop()
    if task_mode == "search" and pattern_file is None:
        st.error("Please upload a reference/pattern image for search mode.")
        st.stop()

    temp_dir = st.session_state.temp_dir
    scene_path = save_upload(scene_file, temp_dir)
    pattern_path = save_upload(pattern_file, temp_dir) if pattern_file else None

    if task_mode == "counting":
        agent_input = f"{user_input}\n\nimage_path: {scene_path}\noutput_dir: {temp_dir}"
    else:
        agent_input = (
            f"{user_input}\n\n"
            f"image_path: {scene_path}\n"
            f"pattern_image_path: {pattern_path}\n"
            f"output_dir: {temp_dir}"
        )

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Run agent
    with st.chat_message("assistant"):
        from agent.agent import build_agent, FALLBACK_MODELS
        from openai import RateLimitError, NotFoundError, APIStatusError, BadRequestError

        env_model = os.environ.get("MODEL_ID", "")
        models_to_try = (
            [env_model] + [m for m in FALLBACK_MODELS if m != env_model]
            if env_model else FALLBACK_MODELS
        )

        logger = AgentLogger()
        tool_cb = ToolEventCallback(logger)
        result = None
        last_error = None

        for model_id in models_to_try:
            logger.model_trying(model_id)
            try:
                executor = build_agent(task_mode, model=model_id)
                result = executor.invoke(
                    {"input": agent_input},
                    config={"callbacks": [tool_cb]},
                )
                logger.model_ok(model_id)
                _log.info("[%s] succeeded", model_id)
                break
            except (RateLimitError, NotFoundError, BadRequestError) as e:
                reason = type(e).__name__
                logger.model_failed(model_id, reason)
                _log.warning("[%s] FAILED (%s): %s", model_id, reason, e)
                last_error = e
                continue
            except APIStatusError as e:
                last_error = e
                if e.status_code in (400, 402, 429, 503):
                    reason = f"HTTP {e.status_code}"
                    logger.model_failed(model_id, reason)
                    _log.warning("[%s] FAILED (%s): %s", model_id, reason, e.message)
                    continue
                raise
            except Exception as e:
                logger.model_failed(model_id, type(e).__name__)
                _log.error("[%s] ERROR: %s", model_id, e)
                last_error = e
                break

        if result is not None:
            output = result.get("output", "No answer returned.")
            if isinstance(output, list):
                output = " ".join(
                    block.get("text", "") for block in output
                    if isinstance(block, dict) and block.get("type") == "text"
                ) or "No answer returned."
            logger.answer(output)
        else:
            _log.error("All models failed. Last: %s", last_error)
            logger.answer("Sorry, all models are currently unavailable. Please try again in a moment.")

        st.session_state.chat_history.append({
            "role": "assistant",
            "events": logger.events,
        })
