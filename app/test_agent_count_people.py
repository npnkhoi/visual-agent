"""End-to-end agent test: count people via LLM + tools."""
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

from app.agent.agent import build_agent, DEFAULT_MODEL

prompt = (
    f"How many people are in this image?\n\n"
    f"image_path: {IMAGE_PATH}\n"
    f"output_dir: {OUTPUT_DIR}"
)

model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL)
print(f"Model  : {model_id}")
print(f"Prompt : {prompt}")
print(f"Output : {OUTPUT_DIR}")
print()

executor = build_agent("counting")
result = executor.invoke({"input": prompt})

output = result.get("output", "")
if isinstance(output, list):
    output = " ".join(
        block.get("text", "") for block in output
        if isinstance(block, dict) and block.get("type") == "text"
    )

print("=" * 50)
print(output)
print("=" * 50)
