"""Train and evaluate TF-IDF sentiment classifiers on IMDb reviews."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text

def load_split(data_dir: Path, split: str, max_per_class: int | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label_name, label in (("neg", 0), ("pos", 1)):
        paths = sorted((data_dir / split / label_name).glob("*.txt"))
        if max_per_class is not None:
            paths = paths[:max_per_class]
        for path in paths:
            rows.append(
                {
                    "review": path.read_text(encoding="utf-8"),
                    "sentiment": label,
                }
            )
    if not rows:
        raise FileNotFoundError(
            f"No reviews found under {data_dir / split}. Run src/download_data.py first."
        )
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


def build_pipeline(classifier: object) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    token_pattern=r"(?u)\b[a-z][a-z]+\b",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.95,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def evaluate_model(name: str, pipeline: Pipeline, test: pd.DataFrame, output_dir: Path) -> dict[str, float]:
    predictions = pipeline.predict(test["review"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        test["sentiment"], predictions, average="binary"
    )
    metrics = {
        "accuracy": float(accuracy_score(test["sentiment"], predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    report = classification_report(
        test["sentiment"], predictions, target_names=["negative", "positive"]
    )
    (output_dir / f"{name}_classification_report.txt").write_text(report, encoding="utf-8")

    ConfusionMatrixDisplay.from_predictions(
        test["sentiment"], predictions, display_labels=["Negative", "Positive"], cmap="Blues"
    )
    plt.title(f"{name.replace('_', ' ').title()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=180)
    plt.close()
    return metrics


def plot_top_words(pipeline: Pipeline, output_dir: Path, top_n: int = 20) -> None:
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    negative_indices = coefficients.argsort()[:top_n]
    positive_indices = coefficients.argsort()[-top_n:][::-1]
    chart_data = pd.DataFrame(
        {
            "word": list(feature_names[negative_indices]) + list(feature_names[positive_indices]),
            "weight": list(coefficients[negative_indices]) + list(coefficients[positive_indices]),
            "sentiment": ["Negative"] * top_n + ["Positive"] * top_n,
        }
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(data=chart_data, x="weight", y="word", hue="sentiment", palette={"Negative": "#d95f5f", "Positive": "#4c9f70"})
    plt.title("Most Predictive Positive and Negative Terms")
    plt.xlabel("Logistic Regression Coefficient")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "top_sentiment_words.png", dpi=180)
    plt.close()


def plot_most_frequent_words(train: pd.DataFrame, output_dir: Path, top_n: int = 20) -> None:
    rows: list[dict[str, object]] = []
    for label, sentiment in ((0, "Negative"), (1, "Positive")):
        counts: Counter[str] = Counter()
        for review in train.loc[train["sentiment"] == label, "review"]:
            counts.update(clean_text(review).split())
        rows.extend(
            {"word": word, "count": count, "sentiment": sentiment}
            for word, count in counts.most_common(top_n)
        )

    chart_data = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(14, 8))
    colors = {"Negative": "#d95f5f", "Positive": "#4c9f70"}
    for axis, sentiment in zip(axes, ("Negative", "Positive")):
        subset = chart_data[chart_data["sentiment"] == sentiment]
        sns.barplot(data=subset, x="count", y="word", color=colors[sentiment], ax=axis)
        axis.set_title(f"Most Frequent {sentiment} Words")
        axis.set_xlabel("Count")
        axis.set_ylabel("")
    figure.tight_layout()
    figure.savefig(output_dir / "most_frequent_sentiment_words.png", dpi=180)
    plt.close(figure)


def plot_model_comparison(results: dict[str, dict[str, float]], output_dir: Path) -> None:
    comparison = (
        pd.DataFrame(results)
        .T.reset_index(names="model")
        .melt(id_vars="model", var_name="metric", value_name="score")
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=comparison, x="metric", y="score", hue="model")
    plt.ylim(0.7, 1.0)
    plt.title("Classifier Performance Comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/aclImdb"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-per-class", type=int, default=None, help="Limit each label per split for quick runs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(args.data_dir, "train", args.max_per_class)
    test = load_split(args.data_dir, "test", args.max_per_class)
    print(f"Training on {len(train):,} reviews; testing on {len(test):,} reviews")

    models = {
        "logistic_regression": build_pipeline(LogisticRegression(max_iter=1_000, random_state=42)),
        "naive_bayes": build_pipeline(MultinomialNB()),
    }
    results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(train["review"], train["sentiment"])
        results[name] = evaluate_model(name, model, test, args.output_dir)
        joblib.dump(model, args.output_dir / f"{name}_model.joblib")

    plot_top_words(models["logistic_regression"], args.output_dir)
    plot_most_frequent_words(train, args.output_dir)
    plot_model_comparison(results, args.output_dir)
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(pd.DataFrame(results).T.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
