"""Official-style SQuAD v1.1 Exact Match and token F1 metrics."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    common = Counter(predicted_tokens) & Counter(reference_tokens)
    shared = sum(common.values())
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)
    if shared == 0:
        return 0.0
    precision = shared / len(predicted_tokens)
    recall = shared / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction: str, references: list[str]) -> tuple[float, float]:
    return (
        max(exact_match(prediction, reference) for reference in references),
        max(token_f1(prediction, reference) for reference in references),
    )
