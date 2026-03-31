"""Extract the noun/noun phrase corresponding to the number field in CountBench.json.

Strategy: use spaCy dependency parsing to find the token with a `nummod` dependency
whose text matches the number word, then return that token's head noun (expanded to
its full noun chunk).
"""
import json
import re
import spacy

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

    def chunk_without_nummod(head_token):
        """Return the noun chunk containing head_token, minus any leading nummod."""
        for chunk in doc.noun_chunks:
            if head_token in chunk:
                # Strip leading number word if present
                tokens = [t for t in chunk if not (t.lower_ == word and t.dep_ == "nummod")]
                return " ".join(t.text for t in tokens)
        return head_token.text

    # Find token matching the number word (whole word, case-insensitive)
    for token in doc:
        if token.lower_ == word and token.dep_ == "nummod":
            return chunk_without_nummod(token.head)

    # Fallback: find any token matching the word and return its head
    for token in doc:
        if token.lower_ == word:
            return chunk_without_nummod(token.head)

    return None


if __name__ == "__main__":
    with open("labels/CountBench.json") as f:
        data = json.load(f)

    failures = []
    for i, item in enumerate(data):
        text = item["text"]
        number = item["number"]
        noun = extract_noun(text, number)
        item["target_noun"] = noun
        if noun is None:
            failures.append((i, number, text))
        else:
            print(f"[{i:04d}] number={number} -> {repr(noun)} | {text[:80]}")

    if failures:
        print(f"\nFailed to extract noun for {len(failures)} items:")
        for idx, n, t in failures:
            print(f"  [{idx:04d}] number={n} | {t}")

    with open("labels/CountBench_nouns.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to labels/CountBench_nouns.json")
