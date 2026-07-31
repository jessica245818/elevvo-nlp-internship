"""Tokenization, stopword removal, and lemmatization for AG News."""

from __future__ import annotations

import re
from html import unescape

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


TOKEN_RE = re.compile(r"[a-z]{2,}")
LEMMATIZER = WordNetLemmatizer()
NEWSWIRE_STOP_WORDS = {"afp", "ap", "gt", "lt", "new", "reuters", "say", "year"}
STOP_WORDS = ENGLISH_STOP_WORDS.union(NEWSWIRE_STOP_WORDS)


def ensure_nltk_resources() -> None:
    """Download WordNet once when it is not already installed."""
    try:
        nltk.data.find("corpora/wordnet.zip")
    except LookupError:
        nltk.download("wordnet", quiet=True, raise_on_error=True)


def preprocess_text(text: str) -> str:
    """Lowercase, tokenize, remove stopwords, and lemmatize a document."""
    tokens = TOKEN_RE.findall(unescape(text).lower())
    lemmas: list[str] = []
    for token in tokens:
        lemma = LEMMATIZER.lemmatize(LEMMATIZER.lemmatize(token, pos="v"), pos="n")
        if lemma not in STOP_WORDS:
            lemmas.append(lemma)
    return " ".join(lemmas)
