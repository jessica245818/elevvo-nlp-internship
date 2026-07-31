"""Lightweight extractive TextRank summarizer."""

from __future__ import annotations

import re

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def summarize(text: str, sentence_count: int = 3) -> str:
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()]
    if len(sentences) <= sentence_count:
        return " ".join(sentences)
    vectors = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    similarities = cosine_similarity(vectors)
    graph = nx.from_numpy_array(similarities)
    scores = nx.pagerank(graph)
    selected = sorted(sorted(scores, key=scores.get, reverse=True)[:sentence_count])
    return " ".join(sentences[index] for index in selected)
