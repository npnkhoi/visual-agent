"""Test: count people in a given image using Grounding DINO + CLIP."""
import json
import os
import sys
import tempfile

_TMP = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(_TMP, exist_ok=True)

IMAGE_URL = "https://www.utdallas.edu/files/2026/02/thermal-energy-hero.jpg"
IMAGE_PATH = os.path.join(_TMP, "test_people.jpg")
OUTPUT_DIR = tempfile.mkdtemp(prefix="va_test_", dir=_TMP)

if not os.path.exists(IMAGE_PATH):
    import subprocess
    print(f"Downloading {IMAGE_URL}...")
    subprocess.run(["curl", "-L", "-A", "Mozilla/5.0", IMAGE_URL, "-o", IMAGE_PATH], check=True)

print(f"Image : {IMAGE_PATH}")
print(f"Output: {OUTPUT_DIR}")
print()

# ── Step 1: Grounding DINO detection ──────────────────────────────────────────
print("Step 1 — Grounding DINO: detecting 'person . people'...")
from agent.tools.detection_tools import run_grounding_dino

det_json = run_grounding_dino(
    image_path=IMAGE_PATH,
    text_prompt="person . people",
    box_threshold=0.3,
    text_threshold=0.25,
    output_dir=OUTPUT_DIR,
)
det = json.loads(det_json)
print(f"  Detected : {det['num_detections']}")
print(f"  Annotated: {det['annotated_image_path']}")
print()

if det["num_detections"] == 0:
    print("No detections — try lowering box_threshold.")
    sys.exit(0)

# ── Step 2: CLIP verification ─────────────────────────────────────────────────
print("Step 2 — CLIP: verifying crops against 'a person'...")
from agent.tools.similarity_tools import clip_verify_crops

ver_json = clip_verify_crops(
    crop_paths_json=json.dumps(det["crop_paths"]),
    text_query="a person",
    threshold=0.15,  # person crops score ~0.18-0.19 with CLIP ViT-L/14
)
ver = json.loads(ver_json)
sims = [f"{s:.3f}" for s in ver["similarities"]]
print(f"  Similarities : {sims}")
print(f"  Verified     : {ver['verified_count']} / {det['num_detections']}")
print()

# ── Step 3: Annotate verified boxes ──────────────────────────────────────────
print("Step 3 — Annotating verified detections...")
from agent.tools.image_tools import annotate_boxes

ann_json = annotate_boxes(
    image_path=IMAGE_PATH,
    indices_json=json.dumps(ver["verified_indices"]),
    boxes_xyxy_json=json.dumps(det["boxes_xyxy"]),
    output_dir=OUTPUT_DIR,
    color="lime",
)
ann = json.loads(ann_json)
print(f"  Annotated image: {ann['annotated_image_path']}")
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 50)
print(f"RESULT: {ver['verified_count']} people found")
print(f"        ({det['num_detections']} detected, {ver['verified_count']} verified by CLIP)")
print("=" * 50)
