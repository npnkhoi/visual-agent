"""CountBench evaluation: three counting systems.

Usage:
    python scripts/0330_benchmark.py [--system {1,2,3}] [--limit N]

    --system  Which system(s) to run (default: all three)
    --limit   Only process first N items (for smoke tests)
"""
import argparse
import glob
import json
import logging
import math
import os
import re
import shutil
import sys

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_PATH = os.path.join(ROOT, "labels", "CountBench_nouns.json")
CLOZE_PATH = os.path.join(ROOT, "labels", "CountBench_cloze.json")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
TMP_DIR = os.path.join(ROOT, "tmp")


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_image(idx: int) -> str | None:
    """Return the image path for a given dataset index, or None if not found."""
    pattern = os.path.join(DATA_DIR, f"{idx:04d}.*")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return matches[0]


def load_dataset(limit: int | None = None) -> list[dict]:
    with open(LABELS_PATH, "r") as f:
        items = json.load(f)
    if limit is not None:
        items = items[:limit]
    return items


def compute_metrics(records: list[dict]) -> dict:
    """
    Compute exact_acc, MAE, RMSE from a list of {predicted, ground_truth} dicts.
    Items where predicted == -1 are excluded from all metrics but counted separately.
    """
    valid = [r for r in records if r["predicted"] != -1]
    failed = len(records) - len(valid)

    if not valid:
        return {"exact_acc": None, "MAE": None, "RMSE": None, "n": 0, "parse_failures": failed}

    exact = sum(1 for r in valid if r["predicted"] == r["ground_truth"]) / len(valid)
    mae = sum(abs(r["predicted"] - r["ground_truth"]) for r in valid) / len(valid)
    mse = sum((r["predicted"] - r["ground_truth"]) ** 2 for r in valid) / len(valid)
    rmse = math.sqrt(mse)

    return {
        "exact_acc": round(exact * 100, 2),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "n": len(valid),
        "parse_failures": failed,
    }


def save_metrics(all_metrics: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Metrics saved → %s", path)


def print_metrics(name: str, m: dict):
    if m["exact_acc"] is None:
        log.info("[%s] No valid predictions.", name)
        return
    log.info(
        "[%s] n=%d  exact_acc=%.2f%%  MAE=%.4f  RMSE=%.4f  parse_failures=%d",
        name, m["n"], m["exact_acc"], m["MAE"], m["RMSE"], m["parse_failures"],
    )


# ── System 1: Agent (MiniMax-orchestrated) ────────────────────────────────────

def load_cloze(limit: int | None = None) -> list[dict]:
    """Load cloze items, keeping only valid ones."""
    with open(CLOZE_PATH) as f:
        items = json.load(f)
    if limit is not None:
        items = items[:limit]
    valid = [it for it in items if it["valid"]]
    skipped = len(items) - len(valid)
    if skipped:
        log.info("[S1] Skipping %d invalid cloze items (ambiguous or missing span)", skipped)
    return valid


def extract_count_from_text(text: str) -> int:
    """Extract last integer from text; return -1 if none found."""
    matches = re.findall(r"\b(\d+)\b", text)
    if not matches:
        return -1
    return int(matches[-1])


def run_system1(limit: int | None = None) -> list[dict]:
    from agent.agent import build_agent

    out_dir = os.path.join(RESULTS_DIR, "system1")
    os.makedirs(out_dir, exist_ok=True)

    cloze_items = load_cloze(limit)

    # Build agent once (models cached via ModelRegistry)
    log.info("[S1] Building agent...")
    executor = build_agent("counting")
    # Disable streaming for batch use
    try:
        executor.agent.runnable.steps[0].bound.streaming = False
    except Exception:
        pass

    records = []
    for item in cloze_items:
        idx = item["idx"]
        ground_truth = item["number"]
        cloze_text = item["cloze_text"]

        image_path = find_image(idx)
        if image_path is None:
            log.warning("[S1] idx=%04d: image not found, skipping", idx)
            continue

        tmp_item_dir = os.path.join(TMP_DIR, "bench_s1", f"{idx:04d}")
        os.makedirs(tmp_item_dir, exist_ok=True)

        prompt = (
            f'What number fills in the [MASK] in this sentence, based on what you see in the image?\n'
            f'"{cloze_text}"\n'
            f"Reply with a single integer.\n\n"
            f"image_path: {os.path.abspath(image_path)}\n"
            f"output_dir: {tmp_item_dir}"
        )

        log.info("[S1] idx=%04d  cloze=%r", idx, cloze_text)
        try:
            result = executor.invoke({"input": prompt})
            raw_output = result.get("output", "")
            # Handle list output (some models return list of content blocks)
            if isinstance(raw_output, list):
                raw_output = " ".join(
                    b.get("text", "") for b in raw_output
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            predicted = extract_count_from_text(str(raw_output))
        except Exception as e:
            log.error("[S1] idx=%04d ERROR: %s", idx, e)
            raw_output = f"ERROR: {e}"
            predicted = -1

        record = {
            "idx": idx,
            "cloze_text": cloze_text,
            "prompt": prompt,
            "agent_response": raw_output,
            "predicted": predicted,
            "ground_truth": ground_truth,
        }
        records.append(record)

        result_path = os.path.join(out_dir, f"{idx:04d}.json")
        with open(result_path, "w") as f:
            json.dump(record, f, indent=2)

        # Clean up tmp files
        try:
            shutil.rmtree(tmp_item_dir)
        except Exception:
            pass

        log.info("[S1] idx=%04d  predicted=%s  gt=%d", idx, predicted, ground_truth)

    return records


# ── System 2: Fixed Pipeline (GDINO → CLIP) ───────────────────────────────────

def run_system2(items: list[dict]) -> list[dict]:
    from agent.tools.detection_tools import run_grounding_dino
    from agent.tools.similarity_tools import clip_verify_crops

    out_dir = os.path.join(RESULTS_DIR, "system2")
    os.makedirs(out_dir, exist_ok=True)

    records = []
    for idx, item in enumerate(items):
        target_noun = item["target_noun"]
        ground_truth = item["number"]

        image_path = find_image(idx)
        if image_path is None:
            log.warning("[S2] idx=%04d: image not found, skipping", idx)
            continue

        abs_image_path = os.path.abspath(image_path)
        tmp_item_dir = os.path.join(TMP_DIR, "bench_s2", f"{idx:04d}")
        os.makedirs(tmp_item_dir, exist_ok=True)

        # Determine CLIP threshold
        clip_threshold = 0.15 if any(w in target_noun.lower() for w in ("person", "people")) else 0.25

        gdino_input = {
            "image_path": abs_image_path,
            "text_prompt": target_noun,
            "box_threshold": 0.3,
            "text_threshold": 0.25,
        }

        log.info("[S2] idx=%04d  noun=%r", idx, target_noun)
        try:
            gdino_raw = run_grounding_dino(
                image_path=abs_image_path,
                text_prompt=target_noun,
                box_threshold=0.3,
                text_threshold=0.25,
                output_dir=tmp_item_dir,
            )
            gdino_data = json.loads(gdino_raw)
        except Exception as e:
            log.error("[S2] idx=%04d GDINO ERROR: %s", idx, e)
            records.append({
                "idx": idx,
                "ground_truth": ground_truth,
                "gdino_input": gdino_input,
                "gdino_output": {"error": str(e)},
                "clip_input": {},
                "clip_output": {"error": "gdino failed"},
                "predicted": -1,
            })
            continue

        crop_paths = gdino_data.get("crop_paths", [])
        num_detections = gdino_data.get("num_detections", 0)

        gdino_output = {
            "num_detections": num_detections,
            "boxes_xyxy": gdino_data.get("boxes_xyxy", []),
            "scores": gdino_data.get("scores", []),
            "labels": gdino_data.get("labels", []),
        }

        clip_input = {
            "text_query": target_noun,
            "threshold": clip_threshold,
            "num_crops": len(crop_paths),
        }

        if crop_paths:
            try:
                clip_raw = clip_verify_crops(
                    crop_paths_json=json.dumps(crop_paths),
                    text_query=target_noun,
                    threshold=clip_threshold,
                )
                clip_data = json.loads(clip_raw)
            except Exception as e:
                log.error("[S2] idx=%04d CLIP ERROR: %s", idx, e)
                clip_data = {"error": str(e), "verified_count": num_detections}
        else:
            clip_data = {"verified_indices": [], "similarities": [], "verified_count": 0}

        clip_output = {
            "verified_indices": clip_data.get("verified_indices", []),
            "similarities": clip_data.get("similarities", []),
            "verified_count": clip_data.get("verified_count", 0),
        }
        predicted = clip_output["verified_count"]

        record = {
            "idx": idx,
            "ground_truth": ground_truth,
            "gdino_input": gdino_input,
            "gdino_output": gdino_output,
            "clip_input": clip_input,
            "clip_output": clip_output,
            "predicted": predicted,
        }
        records.append(record)

        result_path = os.path.join(out_dir, f"{idx:04d}.json")
        with open(result_path, "w") as f:
            json.dump(record, f, indent=2)

        # Clean up tmp files
        try:
            shutil.rmtree(tmp_item_dir)
        except Exception:
            pass

        log.info("[S2] idx=%04d  predicted=%d  gt=%d", idx, predicted, ground_truth)

    return records


# ── System 3: GDINO only ───────────────────────────────────────────────────────

def run_system3(items: list[dict]) -> list[dict]:
    from agent.tools.detection_tools import run_grounding_dino

    out_dir = os.path.join(RESULTS_DIR, "system3")
    os.makedirs(out_dir, exist_ok=True)

    records = []
    for idx, item in enumerate(items):
        target_noun = item["target_noun"]
        ground_truth = item["number"]

        image_path = find_image(idx)
        if image_path is None:
            log.warning("[S3] idx=%04d: image not found, skipping", idx)
            continue

        abs_image_path = os.path.abspath(image_path)
        tmp_item_dir = os.path.join(TMP_DIR, "bench_s3", f"{idx:04d}")
        os.makedirs(tmp_item_dir, exist_ok=True)

        gdino_input = {
            "image_path": abs_image_path,
            "text_prompt": target_noun,
            "box_threshold": 0.3,
            "text_threshold": 0.25,
        }

        log.info("[S3] idx=%04d  noun=%r", idx, target_noun)
        try:
            gdino_raw = run_grounding_dino(
                image_path=abs_image_path,
                text_prompt=target_noun,
                box_threshold=0.3,
                text_threshold=0.25,
                output_dir=tmp_item_dir,
            )
            gdino_data = json.loads(gdino_raw)
            predicted = gdino_data.get("num_detections", -1)
        except Exception as e:
            log.error("[S3] idx=%04d GDINO ERROR: %s", idx, e)
            gdino_data = {"error": str(e)}
            predicted = -1

        gdino_output = {
            "num_detections": gdino_data.get("num_detections", -1),
            "scores": gdino_data.get("scores", []),
            "labels": gdino_data.get("labels", []),
        }

        record = {
            "idx": idx,
            "ground_truth": ground_truth,
            "gdino_input": gdino_input,
            "gdino_output": gdino_output,
            "predicted": predicted,
        }
        records.append(record)

        result_path = os.path.join(out_dir, f"{idx:04d}.json")
        with open(result_path, "w") as f:
            json.dump(record, f, indent=2)

        # Clean up tmp files
        try:
            shutil.rmtree(tmp_item_dir)
        except Exception:
            pass

        log.info("[S3] idx=%04d  predicted=%d  gt=%d", idx, predicted, ground_truth)

    return records


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CountBench evaluation")
    parser.add_argument(
        "--system",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Which system to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N items",
    )
    args = parser.parse_args()

    systems_to_run = [args.system] if args.system else [1, 2, 3]

    log.info("Loading dataset from %s", LABELS_PATH)
    items = load_dataset(args.limit)
    log.info("Dataset: %d items", len(items))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    all_metrics: dict = {}

    # Load existing metrics if present (to avoid overwriting other systems)
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            all_metrics = json.load(f)

    if 3 in systems_to_run:
        log.info("=" * 60)
        log.info("Running System 3: Grounding DINO only")
        log.info("=" * 60)
        records3 = run_system3(items)
        m3 = compute_metrics(records3)
        all_metrics["system3"] = m3
        print_metrics("S3", m3)
        save_metrics(all_metrics)

    if 2 in systems_to_run:
        log.info("=" * 60)
        log.info("Running System 2: GDINO → CLIP")
        log.info("=" * 60)
        records2 = run_system2(items)
        m2 = compute_metrics(records2)
        all_metrics["system2"] = m2
        print_metrics("S2", m2)
        save_metrics(all_metrics)

    if 1 in systems_to_run:
        log.info("=" * 60)
        log.info("Running System 1: Agent (MiniMax)")
        log.info("=" * 60)
        records1 = run_system1(args.limit)
        m1 = compute_metrics(records1)
        all_metrics["system1"] = m1
        print_metrics("S1", m1)
        save_metrics(all_metrics)

    log.info("=" * 60)
    log.info("Final metrics:")
    for name, m in all_metrics.items():
        print_metrics(name, m)
    log.info("Results saved to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
