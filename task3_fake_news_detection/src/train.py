"""Train and evaluate TF-IDF fake news classifiers."""

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
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from wordcloud import WordCloud

from task3_fake_news_detection.src.preprocessing import ensure_nltk_resources, preprocess_text


CLASS_NAMES = ["Fake", "Real"]


def load_dataset(data_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for filename, label in (("Fake.csv", 0), ("True.csv", 1)):
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run the download_data module first.")
        frame = pd.read_csv(path, usecols=["title", "text"])
        frame["label"] = label
        frames.append(frame)

    dataset = pd.concat(frames, ignore_index=True).fillna("")
    dataset["content"] = (
        dataset["title"].str.strip() + " " + dataset["text"].str.strip()
    ).str.strip()
    dataset = dataset[dataset["content"].str.len() > 0]
    dataset = dataset.drop_duplicates(subset=["content"], keep=False)
    return dataset[["content", "label"]].reset_index(drop=True)


def evaluate(
    name: str,
    model: object,
    test_features: object,
    test_labels: pd.Series,
    output_dir: Path,
) -> dict[str, float]:
    predictions = model.predict(test_features)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average="binary"
    )
    metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    report = classification_report(test_labels, predictions, target_names=CLASS_NAMES)
    (output_dir / f"{name}_classification_report.txt").write_text(report, encoding="utf-8")

    ConfusionMatrixDisplay.from_predictions(
        test_labels, predictions, display_labels=CLASS_NAMES, cmap="Blues"
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
    plt.figure(figsize=(11, 5))
    axis = sns.barplot(data=comparison, x="metric", y="score", hue="model")
    plt.ylim(0.9, 1.0)
    plt.title("Fake News Classifier Performance")
    plt.xlabel("")
    plt.ylabel("Score")
    axis.legend(title="Model", loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close()


def plot_word_clouds(train: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    configurations = [(0, "Fake", "Reds"), (1, "Real", "Blues")]
    for axis, (label, title, colormap) in zip(axes, configurations):
        sample = train.loc[train["label"] == label, "content"].sample(
            n=min(5_000, (train["label"] == label).sum()), random_state=42
        )
        cleaned_text = " ".join(preprocess_text(text) for text in sample)
        cloud = WordCloud(
            width=1_600,
            height=900,
            background_color="white",
            colormap=colormap,
            max_words=150,
            collocations=False,
            random_state=42,
        ).generate(cleaned_text)
        axis.imshow(cloud, interpolation="bilinear")
        axis.set_title(f"Common Terms in {title} News", fontsize=16)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "fake_vs_real_wordclouds.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("task3_fake_news_detection/data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task3_fake_news_detection/artifacts"),
    )
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_nltk_resources()

    dataset = load_dataset(args.data_dir)
    if args.max_rows is not None:
        dataset = dataset.groupby("label", group_keys=False).head(args.max_rows // 2)
    train, test = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=dataset["label"],
    )
    print(f"Training on {len(train):,} articles; testing on {len(test):,} articles")

    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=150_000,
        sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(train["content"])
    test_features = vectorizer.transform(test["content"])

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

    plot_model_comparison(results, args.output_dir)
    plot_word_clouds(train, args.output_dir)
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).T.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
