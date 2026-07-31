"""Discover and compare topics in BBC News with LDA and NMF."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.lda_model
import seaborn as sns
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from nltk.stem import SnowballStemmer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score


STEMMER = SnowballStemmer("english")
TOKEN_RE = re.compile(r"[a-z]{3,}")


def tokenize(text: str) -> list[str]:
    """Lowercase, tokenize, remove stopwords, and stem without external downloads."""
    return [
        STEMMER.stem(token)
        for token in TOKEN_RE.findall(text.lower())
        if token not in ENGLISH_STOP_WORDS
    ]


def topic_words(model: object, vocabulary: np.ndarray, count: int = 12) -> list[list[str]]:
    return [vocabulary[weights.argsort()[-count:][::-1]].tolist() for weights in model.components_]


def topic_diversity(topics: list[list[str]]) -> float:
    words = [word for topic in topics for word in topic]
    return len(set(words)) / len(words)


def save_topics(name: str, topics: list[list[str]], output_dir: Path) -> None:
    rows = [
        {"model": name, "topic": index + 1, "rank": rank + 1, "word": word}
        for index, topic in enumerate(topics)
        for rank, word in enumerate(topic)
    ]
    pd.DataFrame(rows).to_csv(output_dir / f"{name}_topic_words.csv", index=False)


def plot_topics(name: str, model: object, vocabulary: np.ndarray, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharex=False)
    for index, (axis, weights) in enumerate(zip(axes, model.components_)):
        top = weights.argsort()[-10:]
        sns.barplot(x=weights[top], y=vocabulary[top], ax=axis, color="#4472C4")
        axis.set_title(f"Topic {index + 1}")
        axis.set_xlabel("Weight")
        axis.set_ylabel("")
    fig.suptitle(f"{name.upper()} — Most Significant Words per Topic", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_topic_words.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("task5_topic_modeling/data/bbc_news.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("task5_topic_modeling/artifacts"))
    parser.add_argument("--topics", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data)
    tokenized_documents = data["text"].map(tokenize).tolist()
    count_vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, min_df=5, max_df=0.85)
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, min_df=5, max_df=0.85)
    counts = count_vectorizer.fit_transform(data["text"])
    tfidf = tfidf_vectorizer.fit_transform(data["text"])

    lda = LatentDirichletAllocation(
        n_components=args.topics, max_iter=30, learning_method="batch", random_state=42, n_jobs=-1
    ).fit(counts)
    nmf = NMF(n_components=args.topics, init="nndsvda", max_iter=500, random_state=42).fit(tfidf)
    lda_document_topics = lda.transform(counts)
    nmf_document_topics = nmf.transform(tfidf)
    category_codes = pd.Categorical(data["category"]).codes

    lda_topics = topic_words(lda, count_vectorizer.get_feature_names_out())
    nmf_topics = topic_words(nmf, tfidf_vectorizer.get_feature_names_out())
    dictionary = Dictionary(tokenized_documents)
    lda_coherence = CoherenceModel(
        topics=lda_topics, texts=tokenized_documents, dictionary=dictionary, coherence="c_v", processes=1
    ).get_coherence()
    nmf_coherence = CoherenceModel(
        topics=nmf_topics, texts=tokenized_documents, dictionary=dictionary, coherence="c_v", processes=1
    ).get_coherence()
    metrics = {
        "dataset": {"articles": len(data), "vocabulary_size": counts.shape[1], "topics": args.topics},
        "lda": {
            "perplexity": float(lda.perplexity(counts)),
            "topic_diversity": topic_diversity(lda_topics),
            "coherence_c_v": float(lda_coherence),
            "category_nmi": float(normalized_mutual_info_score(category_codes, lda_document_topics.argmax(axis=1))),
        },
        "nmf": {
            "reconstruction_error": float(nmf.reconstruction_err_),
            "topic_diversity": topic_diversity(nmf_topics),
            "coherence_c_v": float(nmf_coherence),
            "category_nmi": float(normalized_mutual_info_score(category_codes, nmf_document_topics.argmax(axis=1))),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_topics("lda", lda_topics, args.output_dir)
    save_topics("nmf", nmf_topics, args.output_dir)
    plot_topics("lda", lda, count_vectorizer.get_feature_names_out(), args.output_dir)
    plot_topics("nmf", nmf, tfidf_vectorizer.get_feature_names_out(), args.output_dir)

    prepared = pyLDAvis.lda_model.prepare(lda, counts, count_vectorizer, sort_topics=False)
    pyLDAvis.save_html(prepared, str(args.output_dir / "lda_visualization.html"))
    dominant = data[["category", "text"]].copy()
    dominant["lda_topic"] = lda_document_topics.argmax(axis=1) + 1
    dominant["nmf_topic"] = nmf_document_topics.argmax(axis=1) + 1
    dominant["text_preview"] = dominant.pop("text").str.replace(r"\s+", " ", regex=True).str[:240]
    dominant.to_csv(args.output_dir / "document_topics.csv", index=False)

    for name, topics in (("LDA", lda_topics), ("NMF", nmf_topics)):
        print(name)
        for index, words in enumerate(topics, 1):
            print(f"  Topic {index}: {', '.join(words)}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
