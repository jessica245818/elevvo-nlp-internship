"""Reusable text preprocessing for sentiment models."""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


HTML_RE = re.compile(r"<[^>]+>")
NON_LETTER_RE = re.compile(r"[^a-z\s]")
SPACE_RE = re.compile(r"\s+")
CONTRACTION_FRAGMENTS = {
    "s", "t", "d", "ll", "m", "re", "ve",
    "ain", "aren", "couldn", "didn", "doesn", "don", "hadn", "hasn",
    "haven", "isn", "mightn", "mustn", "needn", "shan", "shouldn",
    "wasn", "weren", "won", "wouldn",
}
STOP_WORDS = ENGLISH_STOP_WORDS.union(CONTRACTION_FRAGMENTS)


def clean_text(text: str) -> str:
    """Lowercase text and remove HTML, punctuation, numbers, and stopwords."""
    text = HTML_RE.sub(" ", text.lower())
    text = NON_LETTER_RE.sub(" ", text)
    return " ".join(
        token for token in SPACE_RE.split(text.strip())
        if token and token not in STOP_WORDS
    )

