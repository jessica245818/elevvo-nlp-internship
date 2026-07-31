"""Train and evaluate multiclass TF-IDF classifiers on AG News."""

from __future__ import annotations

import argparse
import json
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
from sklearn.svm import LinearSVC

from task2_news_classification.src.preprocessing import ensure_nltk_resources, preprocess_text


CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def load_split(data_dir: Path, split: str, max_rows: int | None) -> pd.DataFrame:
    path = data_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the download_data module first.")
    frame = pd.read_parquet(path, columns=["text", "label"])
    if max_rows is not None:
        frame = frame.groupby("label", group_keys=False).head(
            max_rows // len(CLASS_NAMES)
        )
    return frame.sample(frac=1, random_state=42).reset_index(drop=True)


def evaluate(
    name: str,
    model: object,
    test_features: object,
    test_labels: pd.Series,
    output_dir: Path,
) -> dict[str, float]:
    predictions = model.predict(test_features)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average="macro"
    )
    metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }
    report = classification_report(test_labels, predictions, target_names=CLASS_NAMES)
    (output_dir / f"{name}_classification_report.txt").write_text(report, encoding="utf-8")

    ConfusionMatrixDisplay.from_predictions(
        test_labels,
        predictions,
        display_labels=CLASS_NAMES,
        cmap="Blues",
        xticks_rotation=20,
    )
    plt.title(f"{name.replace('_', ' ').title()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=180)
    plt.close()
    return metrics


def plot_model_comparison(results: dict[str, dict[str, float]], output_dir: Path) -> None:
    comparison = (
        pd.DataFrame(results)
        .T.reset_index(names="model")
        .melt(id_vars="model", var_name="metric", value_name="score")
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=comparison, x="metric", y="score", hue="model")
    plt.ylim(0.8, 1.0)
    plt.title("AG News Classifier Performance")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close()


def plot_frequent_words(
    vectorizer: TfidfVectorizer,
    train_features: object,
    train_labels: pd.Series,
    output_dir: Path,
    top_n: int = 15,
) -> None:
    feature_names = vectorizer.get_feature_names_out()
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    colors = ["#4c78a8", "#59a14f", "#f28e2b", "#b07aa1"]
    for label, (axis, category, color) in enumerate(zip(axes.flat, CLASS_NAMES, colors)):
        mean_scores = train_features[train_labels.to_numpy() == label].mean(axis=0).A1
        indices = mean_scores.argsort()[-top_n:]
        chart = pd.DataFrame(
            {"term": feature_names[indices], "mean_tfidf": mean_scores[indices]}
        )
        sns.barplot(data=chart, x="mean_tfidf", y="term", color=color, ax=axis)
        axis.set_title(f"{category}: Frequent and Distinctive Terms")
        axis.set_xlabel("Mean TF-IDF")
        axis.set_ylabel("")
    figure.suptitle("Top Terms by AG News Category", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_dir / "top_words_per_category.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("task2_news_classification/data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task2_news_classification/artifacts"),
    )
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_nltk_resources()

    train = load_split(args.data_dir, "train", args.max_train_rows)
    test = load_split(args.data_dir, "test", args.max_test_rows)
    print(f"Training on {len(train):,} articles; testing on {len(test):,} articles")

    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=120_000,
        sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(train["text"])
    test_features = vectorizer.transform(test["text"])

    models = {
        "logistic_regression": LogisticRegression(max_iter=1_000, random_state=42),
        "linear_svm": LinearSVC(random_state=42),
    }
    results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(train_features, train["label"])
        results[name] = evaluate(name, model, test_features, test["label"], args.output_dir)
        joblib.dump(
            {"vectorizer": vectorizer, "model": model, "class_names": CLASS_NAMES},
            args.output_dir / f"{name}_bundle.joblib",
        )

    plot_frequent_words(vectorizer, train_features, train["label"], args.output_dir)
    plot_model_comparison(results, args.output_dir)
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).T.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
