"""Convert CountBench_subset18.json to agentflow format and download images.

Run once from visual-agent/:
    python pipelines/pipelines/prepare_data.py

Output:
    pipelines/pipelines/data/subset18.json   — agentflow-formatted dataset
    pipelines/pipelines/data/images/          — downloaded images
"""
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
from tqdm import tqdm

HERE = Path(__file__).parent
LABELS_PATH = HERE.parent.parent / "labels" / "CountBench_subset18.json"
DATA_DIR = HERE / "data"
IMAGES_DIR = DATA_DIR / "images"
OUTPUT_PATH = DATA_DIR / "subset18.json"


def filename_from_url(url: str) -> str:
    return Path(urlparse(url).path).name


def download_image(item: dict, retries: int = 3) -> bool:
    url = item["image_url"]
    fname = filename_from_url(url)
    dest = IMAGES_DIR / fname
    item["_filename"] = fname
    if dest.exists():
        return True
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=20, stream=True)
            r.raise_for_status()
            if not r.headers.get("Content-Type", "").startswith("image/"):
                return False
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"  attempt {attempt+1}/{retries} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return False


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(LABELS_PATH) as f:
        subset = json.load(f)

    print(f"Downloading {len(subset)} images...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download_image, item): item for item in subset}
        for future in tqdm(as_completed(futures), total=len(subset), unit="img"):
            future.result()

    agentflow_data = []
    skipped = 0
    for item in subset:
        fname = item.get("_filename")
        if not fname or not (IMAGES_DIR / fname).exists():
            print(f"  skipping {item['question_id']}: image not downloaded")
            skipped += 1
            continue
        agentflow_data.append({
            "id": str(item["question_id"]),
            "data": {
                "image": fname,
                "question": item["question"],
                "answer": item["answer"],
            },
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(agentflow_data, f, indent=2)

    print(f"Saved {len(agentflow_data)} items to {OUTPUT_PATH} ({skipped} skipped)")


if __name__ == "__main__":
    main()
