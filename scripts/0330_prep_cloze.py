"""Preprocess CountBench items into cloze sentences.

For each item, find the number span in the text, verify it is unique,
mask it with [MASK], and save results to labels/CountBench_cloze.json.

A span is valid iff exactly one occurrence of the number (as word or digit)
appears in the text (case-insensitive, word-boundary matched).
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_PATH = os.path.join(ROOT, "labels", "CountBench_nouns.json")
OUT_PATH = os.path.join(ROOT, "labels", "CountBench_cloze.json")

NUMBER_WORDS = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def find_spans(text: str, number: int) -> list[tuple[str, int]]:
    """Return all (form, count) pairs where the number appears in text."""
    forms = [NUMBER_WORDS[number], str(number)]
    results = []
    for form in forms:
        matches = re.findall(rf"\b{re.escape(form)}\b", text, flags=re.IGNORECASE)
        if matches:
            results.append((form, len(matches)))
    return results


def make_cloze(text: str, form: str) -> str:
    return re.sub(rf"\b{re.escape(form)}\b", "[MASK]", text, count=1, flags=re.IGNORECASE)


def process(items: list[dict]) -> list[dict]:
    out = []
    n_valid = 0
    n_invalid = 0

    for idx, item in enumerate(items):
        number = item["number"]
        text = item["text"]
        spans = find_spans(text, number)

        total_occurrences = sum(c for _, c in spans)
        distinct_forms = [f for f, _ in spans]

        if total_occurrences == 0:
            issue = f"number {number!r} not found in text"
            entry = {
                "idx": idx,
                "number": number,
                "original_text": text,
                "masked_span": None,
                "cloze_text": None,
                "valid": False,
                "issue": issue,
            }
            n_invalid += 1
            print(f"  INVALID [{idx:04d}] {issue!r}: {text!r}")
        elif total_occurrences > 1:
            issue = f"{total_occurrences} occurrences of {distinct_forms} in text"
            entry = {
                "idx": idx,
                "number": number,
                "original_text": text,
                "masked_span": None,
                "cloze_text": None,
                "valid": False,
                "issue": issue,
            }
            n_invalid += 1
            print(f"  INVALID [{idx:04d}] {issue!r}: {text!r}")
        else:
            # Exactly one occurrence across all forms
            masked_form = distinct_forms[0]
            cloze_text = make_cloze(text, masked_form)
            entry = {
                "idx": idx,
                "number": number,
                "original_text": text,
                "masked_span": masked_form,
                "cloze_text": cloze_text,
                "valid": True,
            }
            n_valid += 1

        out.append(entry)

    print(f"\nTotal: {len(items)}  valid: {n_valid}  invalid: {n_invalid}")
    return out


def main():
    with open(LABELS_PATH) as f:
        items = json.load(f)

    print(f"Loaded {len(items)} items from {LABELS_PATH}\n")
    results = process(items)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
