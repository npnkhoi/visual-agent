import json
import requests
import re
import spacy
import torch
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, pipeline, GenerationConfig
from tqdm import tqdm
import httpx

# --- Configuration ---
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_DINO_DIM = 1333  # GroundingDINO architectural constraint
NMS_IOU_THRESHOLD = 0.5  # IoU threshold for Non-Maximum Suppression

# --- Model Initialization ---
print(f"Initializing Grounding DINO on {DEVICE}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
nlp = spacy.load("en_core_web_sm")  # for singularization only

# Generative extractor for Dino prompts
print("Initializing Qwen2.5 prompt extractor...")
extractor = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device=DEVICE,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)


def get_img_paths():
    """Load the dataset metadata."""
    path = Path("labels/CountBench_test.json")
    if not path.exists():
        raise FileNotFoundError(f"Label file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        img_path = json.load(f)
    return img_path["database"]


import httpx

def download_single_image(data, dir_path, retries=3):
    """Worker function for parallel image downloading using httpx."""
    img_url = data.get("image_url")
    if not img_url:
        data["save_path"] = None
        return False

    filename = img_url.split("/")[-1].split("?")[0]
    # Ensure filename isn't empty or weirdly formatted
    if not filename or len(filename) < 3:
        filename = f"img_{hash(img_url)}.jpg"
        
    save_path = Path(dir_path) / filename
    data["save_path"] = str(save_path)

    if save_path.exists():
        return True

    # Standard browser-like headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }

    # Use a context manager for the client to handle connection pooling
    with httpx.Client(headers=headers, follow_redirects=True, timeout=20.0) as client:
        for attempt in range(retries):
            try:
                response = client.get(img_url)
                response.raise_for_status()

                # Verify we actually got an image
                content_type = response.headers.get("Content-Type", "")
                if "image" not in content_type:
                    print(f"Skipping: URL did not return an image ({content_type})")
                    return False

                with open(save_path, "wb") as f:
                    f.write(response.content)

                return True

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    print(f"Image not found (404): {img_url}")
                    break # Don't retry 404s
                print(f"Attempt {attempt + 1} failed for {img_url}: {e}")
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {img_url}: {e}")
            
            if attempt < retries - 1:
                time.sleep(2)

    return False


def download_imgs(dataset, max_workers=8):
    """Parallel download utility with progress bar."""
    dir_path = Path("images")
    dir_path.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single_image, d, dir_path): d for d in dataset}
        for future in tqdm(as_completed(futures), total=len(dataset), desc="[1/3] Downloading Images", unit="img"):
            results.append(future.result())
    return dataset

def clean_question(text: str) -> str:
    """Only remove the task instruction prefix. Nothing else."""
    return re.sub(
        r"fill in a number.*?into the mask\.?\s*",
        "", text, flags=re.IGNORECASE
    ).strip()


def singularize(phrase: str) -> str:
    """Singularize the head noun of a phrase via spaCy lemmatizer."""
    doc = nlp(phrase)
    tokens = list(doc)

    head_idx = next(
        (i for i in range(len(tokens) - 1, -1, -1)
         if tokens[i].pos_ in ("NOUN", "PROPN")),
        None
    )
    if head_idx is None:
        return phrase.lower()

    result = []
    for i, token in enumerate(tokens):
        if token.pos_ in ("DET", "NUM") or token.text.lower() in ("a", "an", "the"):
            continue
        result.append(token.lemma_.lower() if i == head_idx else token.text.lower())

    return " ".join(result).strip()


def extract_dino_prompt(question: str) -> str:
    """Use Qwen2.5-Instruct to extract the main countable visual object from the caption."""
    context = clean_question(question)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise data extraction assistant for computer vision. Your task is to identify the primary, "
                "countable physical object in a caption. You must ignore abstract "
                "concepts, actions, and background elements"
            )
        },
        {
            "role": "user",
            "content": f"Caption: {context} \n"
                       "Extract the main countable object. \n"
                       "\n"
                       "Rules: \n"
                       "- Return ONLY a short noun phrase \n"
                       "- Include adjectives that describe the noun phrase \n"
                       "- No articles or numbers \n"
                       "- Only the object that is being counted \n"
                       "Examples:\n"
                        "- Caption: Three large red apples on a table. -> large red apple\n"
                        "- Caption: A person wearing a blue waterproof jacket. -> blue waterproof jacket\n"
                        "- Caption: The fluffy white dog is running. -> fluffy white dog\n\n"
                        "\n"
                        "Object: \n"
        }
    ]

    gen_config = GenerationConfig(max_new_tokens=10, do_sample=False)
    result = extractor(messages, generation_config=gen_config)
    output = result[0]["generated_text"][-1]["content"].strip()
    # Clean up any stray punctuation
    output = re.sub(r"[^a-zA-Z0-9 ]", "", output).strip()

    if not output:
        # Fallback: use spaCy to grab the first noun chunk
        doc = nlp(context.replace("[MASK]", "some"))
        output = next((chunk.text for chunk in doc.noun_chunks), "object")

    return singularize(output) + " ."


def preprocess_text(database):
    for data in tqdm(database, desc="[2/3] Preprocessing Text", unit="query"):
        if "Dino_prompt" in data:
            continue
        data["Dino_prompt"] = extract_dino_prompt(data["question"])
    return database

def resize_image(image, max_size=MAX_DINO_DIM):
    """Resizes image maintaining aspect ratio for model constraints."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image

    if w > h:
        new_w = max_size
        new_h = int(h * (max_size / w))
    else:
        new_h = max_size
        new_w = int(w * (max_size / h))

    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def apply_nms(result, iou_threshold=NMS_IOU_THRESHOLD):
    """Remove overlapping boxes using torchvision NMS, keeping highest-score boxes."""
    from torchvision.ops import nms as tv_nms

    boxes = result["boxes"]   # (N, 4) float tensor
    scores = result["scores"] # (N,) float tensor

    if len(boxes) == 0:
        return result

    keep = tv_nms(boxes.float(), scores.float(), iou_threshold)
    return {
        "boxes":         boxes[keep],
        "scores":        scores[keep],
        "labels":        [result["labels"][i] for i in keep.tolist()],
        "original_size": result["original_size"],
        "resized_size":  result["resized_size"],
    }


def run_groundingDINO_batch(batch_data):
    """Performs detection on a batch with corrected processor handling."""
    raw_images = [Image.open(d["save_path"]).convert("RGB") for d in batch_data]
    resized_images = [resize_image(img) for img in raw_images]

    # Ensure prompts are valid strings and not empty
    prompts = [d.get("Dino_prompt", "").strip() for d in batch_data]
    prompts = [p if p else "object" for p in prompts]

    # Grounding DINO requires specific prompt formatting (ending with a dot)
    # The processor usually handles this, but explicit check prevents errors
    formatted_prompts = [p if p.endswith(".") else f"{p}." for p in prompts]

    inputs = processor(
        images=resized_images,
        text=formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process using the resized dimensions
    resized_sizes = [img.size[::-1] for img in resized_images]

    # post_process_grounded_object_detection is specifically designed for this model
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.35,
        text_threshold=0.25,
        target_sizes=resized_sizes
    )

    # Attach metadata for inverse scaling
    for i, res in enumerate(results):
        res["original_size"] = raw_images[i].size
        res["resized_size"] = resized_images[i].size

    # Apply NMS to remove duplicate/overlapping boxes across labels
    results = [apply_nms(res) for res in results]

    return results


def save_processed_image(data, results, output_folder="processed"):
    """Annotates original high-res image by inverting scaling factors."""
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load original for final output
    original_image = Image.open(data["save_path"]).convert("RGB")
    draw = ImageDraw.Draw(original_image)

    orig_w, orig_h = results["original_size"]
    res_w, res_h = results["resized_size"]

    # Calculate ratios for coordinate inversion
    ratio_w = orig_w / res_w
    ratio_h = orig_h / res_h

    # Dynamic styling based on image resolution
    font_size = max(15, int(orig_h * 0.02))
    line_width = max(3, int(orig_w * 0.004))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        box_coords = box.tolist()

        # Apply inverse scaling
        inverted_box = [
            box_coords[0] * ratio_w,
            box_coords[1] * ratio_h,
            box_coords[2] * ratio_w,
            box_coords[3] * ratio_h
        ]

        draw.rectangle(inverted_box, outline="red", width=line_width)
        draw.text((inverted_box[0], inverted_box[1] - font_size - 5),
                  f"{label}: {score:.2f}", fill="red", font=font)

    save_name = Path(data["save_path"]).name
    original_image.save(out_path / save_name)


def print_detection_stats(database):
    """Prints detection statistics comparing num_boxes vs ground truth (answer)."""
    # Per-label: list of (num_boxes, ground_truth) tuples
    label_stats = defaultdict(list)

    for entry in database:
        if "num_boxes" not in entry:
            continue
        prompt = entry.get("Dino_prompt", "unknown")
        gt = entry.get("answer", None)
        label_stats[prompt].append((entry["num_boxes"], gt))

    # Aggregate totals
    all_pairs = [(nb, gt) for pairs in label_stats.values() for nb, gt in pairs if gt is not None]
    total_entries = len(all_pairs)
    total_boxes = sum(nb for nb, _ in all_pairs)
    total_gt = sum(gt for _, gt in all_pairs)
    exact_matches = sum(1 for nb, gt in all_pairs if nb == gt)
    zero_det = sum(1 for nb, _ in all_pairs if nb == 0)

    print("\n--- Detection Statistics (vs Ground Truth) ---")
    print(f"{'Label':<30} {'N':>5} {'Avg Det':>8} {'Avg GT':>8} {'Exact%':>8} {'Zero Det':>9}")
    print("-" * 72)
    for label, pairs in sorted(label_stats.items()):
        valid = [(nb, gt) for nb, gt in pairs if gt is not None]
        if not valid:
            continue
        n = len(valid)
        avg_det = sum(nb for nb, _ in valid) / n
        avg_gt  = sum(gt for _, gt in valid) / n
        exact   = sum(1 for nb, gt in valid if nb == gt) / n * 100
        zeros   = sum(1 for nb, _ in valid if nb == 0)
        print(f"{label:<30} {n:>5} {avg_det:>8.2f} {avg_gt:>8.2f} {exact:>7.1f}% {zeros:>9}")
    print("-" * 72)
    overall_exact = exact_matches / max(total_entries, 1) * 100
    print(f"{'TOTAL':<30} {total_entries:>5} {total_boxes/max(total_entries,1):>8.2f} "
          f"{total_gt/max(total_entries,1):>8.2f} {overall_exact:>7.1f}% {zero_det:>9}")


def save_database_json(database, folder="processed", filename="metadata_results.json"):
    """
    Serializes the processed database into a JSON file for data persistence.
    """
    out_path = Path(folder)
    out_path.mkdir(parents=True, exist_ok=True)

    output_file = out_path / filename

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            # indent=4 ensures the file is human-readable for academic review
            json.dump(database, f, indent=4, ensure_ascii=False)
        print(f"Database successfully saved to {output_file}")
    except Exception as e:
        print(f"Failed to save database JSON: {e}")

if __name__ == '__main__':
    print("\n--- Starting Data Pipeline ---")
    db = get_img_paths()
    db = download_imgs(db)
    db = preprocess_text(db)

    print("\n--- Starting Model Inference ---")
    for i in tqdm(range(0, len(db), BATCH_SIZE), desc="[3/3] Inference & Saving", unit="batch"):
        batch = db[i: i + BATCH_SIZE]
        valid_batch = [d for d in batch if d.get("save_path") and Path(d["save_path"]).exists()]

        if not valid_batch:
            continue

        try:
            batch_results = run_groundingDINO_batch(valid_batch)
            for entry, result in zip(valid_batch, batch_results):
                save_processed_image(entry, result)
                # Collect per-entry detection statistics
                labels = result.get("labels", [])
                entry["num_boxes"] = len(labels)
                entry["ground_labels"] = labels
        except Exception as e:
            print(f"Error in batch {i}: {e}")

    save_database_json(db)

    print_detection_stats(db)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nAll tasks completed. Check the 'processed/' folder for results.")