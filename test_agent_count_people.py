"""End-to-end agent test: count people via LLM + tools."""
import json
import os
import subprocess
import tempfile

_TMP = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(_TMP, exist_ok=True)

IMAGE_URL = "https://www.utdallas.edu/files/2026/02/thermal-energy-hero.jpg"
IMAGE_PATH = os.path.join(_TMP, "test_people.jpg")
OUTPUT_DIR = tempfile.mkdtemp(prefix="va_agent_test_", dir=_TMP)

if not os.path.exists(IMAGE_PATH):
    print(f"Downloading {IMAGE_URL}...")
    subprocess.run(["curl", "-L", "-A", "Mozilla/5.0", IMAGE_URL, "-o", IMAGE_PATH], check=True)

from dotenv import load_dotenv
load_dotenv()

from agent.agent import build_agent, FALLBACK_MODELS
from openai import RateLimitError, NotFoundError, APIStatusError

prompt = (
    f"How many people are in this image?\n\n"
    f"image_path: {IMAGE_PATH}\n"
    f"output_dir: {OUTPUT_DIR}"
)

print(f"Prompt : {prompt}")
print(f"Output : {OUTPUT_DIR}")
print()

env_model = os.environ.get("MODEL_ID", "")
models_to_try = (
    [env_model] + [m for m in FALLBACK_MODELS if m != env_model]
    if env_model else FALLBACK_MODELS
)

result = None
used_model = None

for model_id in models_to_try:
    try:
        print(f"Trying model: {model_id}")
        executor = build_agent("counting", model=model_id)
        result = executor.invoke({"input": prompt})
        used_model = model_id
        break
    except (RateLimitError, NotFoundError) as e:
        print(f"  FAILED ({type(e).__name__}): {e}")
    except APIStatusError as e:
        if e.status_code in (402, 429, 503):
            print(f"  FAILED (HTTP {e.status_code}): {e.message}")
        else:
            raise

if result is None:
    print("All models failed.")
else:
    print()
    print(f"Model used: {used_model}")
    print()
    print("=" * 50)
    print(result["output"])
    print("=" * 50)
