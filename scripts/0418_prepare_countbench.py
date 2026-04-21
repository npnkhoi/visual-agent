"""Build pipelines/data/countbench.json from labels/CountBench_original.json.

Steps:
  1. Filter to items whose image exists locally in data/
  2. Extract target_noun with spaCy dependency parsing
  3. Build masked question (cloze) by replacing the number word with [MASK]
  4. Save as agentflow format: [{"id": str, "data": {...}}, ...]
"""
import json
import re
from pathlib import Path

import spacy

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SRC = ROOT / "labels" / "CountBench_original.json"
OUT = ROOT / "pipelines" / "data" / "countbench.json"

NUM_TO_WORD = {
    2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten",
}

nlp = spacy.load("en_core_web_sm")


def extract_noun(text: str, number: int) -> str | None:
    word = NUM_TO_WORD.get(number)
    if word is None:
        return None
    doc = nlp(text)

    def chunk_for(head_token):
        for chunk in doc.noun_chunks:
            if head_token in chunk:
                tokens = [t for t in chunk if not (t.lower_ == word and t.dep_ == "nummod")]
                return " ".join(t.text for t in tokens)
        return head_token.text

    for token in doc:
        if token.lower_ == word and token.dep_ == "nummod":
            return chunk_for(token.head)
    for token in doc:
        if token.lower_ == word:
            return chunk_for(token.head)
    return None


def make_question(text: str, number: int) -> str | None:
    word = NUM_TO_WORD.get(number)
    if word is None:
        return None
    forms = [word, str(number)]
    for form in forms:
        matches = re.findall(rf"\b{re.escape(form)}\b", text, flags=re.IGNORECASE)
        if len(matches) == 1:
            return re.sub(rf"\b{re.escape(form)}\b", "[MASK]", text, count=1, flags=re.IGNORECASE)
    return None


raw = json.loads(SRC.read_text())
items = []
skipped_no_image = 0
skipped_no_noun = 0
skipped_no_question = 0

for i, entry in enumerate(raw):
    matches = list(DATA_DIR.glob(f"{i:04d}.*"))
    if not matches:
        skipped_no_image += 1
        continue
    try:
        from PIL import Image as _Image
        _Image.open(matches[0]).verify()
    except Exception:
        skipped_no_image += 1
        continue

    noun = extract_noun(entry["text"], entry["number"])
    if noun is None:
        skipped_no_noun += 1
        continue

    question = make_question(entry["text"], entry["number"])
    if question is None:
        skipped_no_question += 1
        continue

    items.append({
        "id": str(i),
        "data": {
            "image": matches[0].name,
            "text": entry["text"],
            "question": question,
            "answer": entry["number"],
            "target_noun": noun,
        },
    })

OUT.write_text(json.dumps(items, indent=2))
print(f"Written {len(items)} items to {OUT}")
print(f"Skipped: {skipped_no_image} no image, {skipped_no_noun} no noun, {skipped_no_question} ambiguous number")
