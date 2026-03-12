"""LangChain agent factory for counting and search tasks."""
import os
from typing import Optional

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Ordered list of models to try when the primary is unavailable.
# Free models first, then cheap paid fallbacks.
FALLBACK_MODELS = [
    # Free tier
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "qwen/qwen3-coder:free",
    "openai/gpt-oss-120b:free",
    "google/gemma-3-27b-it:free",
    # Cheap paid (~$0.01–0.15 / 1M tokens)
    "google/gemini-2.0-flash-lite-001",
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-small-3.2-24b-instruct",
]

from .tools.detection_tools import grounding_dino_tool
from .tools.similarity_tools import clip_verify_tool, clip_rank_tool
from .tools.image_tools import image_grid_tool, annotate_boxes_tool

ALL_TOOLS = [
    grounding_dino_tool,
    clip_verify_tool,
    clip_rank_tool,
    image_grid_tool,
    annotate_boxes_tool,
]

COUNTING_SYSTEM_PROMPT = """You are a precise computer vision assistant specialized in counting objects in images.

Your workflow for counting tasks:
1. Use `grounding_dino_detect` to find candidate objects. ALWAYS use dot-separated prompts with synonyms (e.g., "person . people . human" for people, "cat . kitten" for cats). A single word without dots often returns 0 detections.
2. Use `clip_verify_crops` to filter out false positives. You MUST pass both `crop_paths_json` (from step 1) AND `text_query` (e.g., "a cat"). Never omit text_query. Use threshold=0.15 for people/persons, 0.25 for other objects.
3. Use `annotate_boxes` to draw lime-colored boxes around only the verified detections on the original image.
4. Report the final count clearly: how many were detected, how many were verified.

Always provide image_path, text_prompt, and output_dir to detection. Pass crop_paths as JSON string to CLIP.
Be precise and methodical. If detection returns 0 results, try a different/broader text prompt.
Never mention file paths or image paths in your final answer."""

SEARCH_SYSTEM_PROMPT = """You are a computer vision assistant specialized in finding specific persons or objects in images.

Your workflow for search tasks:
1. Use `grounding_dino_detect` on the scene image to detect all relevant objects (e.g., "person" for people searches).
2. Use `clip_rank_by_pattern` with the pattern/reference image to rank detections by visual similarity.
3. Use `make_image_grid` to create a visual grid of the top-K matches for display.
4. Interpret results: similarity >0.75 = high confidence, 0.50-0.75 = possible match, <0.50 = not found.
5. Report the best match index/rank and its similarity score.

Always pass crop_paths as JSON string to CLIP tools. Use top_k=5 unless asked otherwise.
Never mention file paths or image paths in your final answer."""


def build_agent(task_mode: str = "counting", model: Optional[str] = None) -> AgentExecutor:
    """Build and return a LangChain AgentExecutor for the given task mode."""
    system_prompt = (
        COUNTING_SYSTEM_PROMPT if task_mode == "counting" else SEARCH_SYSTEM_PROMPT
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    resolved_model = model or os.environ.get("MODEL_ID", FALLBACK_MODELS[0])

    llm = ChatOpenAI(
        model=resolved_model,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        openai_api_base=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0,
        streaming=True,
    )

    agent = create_tool_calling_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        max_iterations=15,
        return_intermediate_steps=True,
        verbose=True,
    )
