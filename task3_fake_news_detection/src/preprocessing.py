"""Text preprocessing for fake news classification."""

from __future__ import annotations

import re
from html import unescape

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


HTML_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"[a-z]{2,}")
LEMMATIZER = WordNetLemmatizer()
NEWSWIRE_STOP_WORDS = {"afp", "ap", "news", "reuters", "said", "says"}
STOP_WORDS = ENGLISH_STOP_WORDS.union(NEWSWIRE_STOP_WORDS)


def ensure_nltk_resources() -> None:
    try:
        nltk.data.find("corpora/wordnet.zip")
    except LookupError:
        nltk.download("wordnet", quiet=True, raise_on_error=True)


def preprocess_text(text: str) -> str:
    """Remove markup, tokenize, remove stopwords, and lemmatize text."""
    plain_text = HTML_RE.sub(" ", unescape(text).lower())
    lemmas: list[str] = []
    for token in TOKEN_RE.findall(plain_text):
        lemma = LEMMATIZER.lemmatize(LEMMATIZER.lemmatize(token, pos="v"), pos="n")
        if lemma not in STOP_WORDS:
            lemmas.append(lemma)
    return " ".join(lemmas)
