"""Evaluate DistilBART and TextRank summaries with ROUGE."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from rouge_score import rouge_scorer
from transformers import pipeline

from task7_text_summarization.src.textrank import summarize as textrank_summarize


MODEL_ID = "sshleifer/distilbart-cnn-6-6"
ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")


def device_name() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def average_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)
    totals = defaultdict(float)
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        for metric in ROUGE_TYPES:
            totals[metric] += scores[metric].fmeasure
    return {metric: 100 * totals[metric] / len(predictions) for metric in ROUGE_TYPES}


def save_plot(metrics: dict[str, dict[str, float]], output_dir: Path) -> None:
    rows = [
        {"method": method, "metric": metric.upper(), "score": values[metric]}
        for method, values in metrics.items()
        for metric in ROUGE_TYPES
    ]
    plt.figure(figsize=(9, 5))
    axis = sns.barplot(data=pd.DataFrame(rows), x="metric", y="score", hue="method")
    axis.set_ylim(0, max(row["score"] for row in rows) + 8)
    axis.set_xlabel("")
    axis.set_ylabel("ROUGE F1")
    axis.set_title("CNN/DailyMail Summarization Comparison")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.1f")
    plt.tight_layout()
    plt.savefig(output_dir / "rouge_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("task7_text_summarization/data/evaluation_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("task7_text_summarization/artifacts"))
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data)
    articles = data["article"].tolist()
    references = data["highlights"].str.replace("\n", " ").tolist()

    device = device_name()
    print(f"Loading {MODEL_ID} on {device}")
    summarizer = pipeline("summarization", model=MODEL_ID, tokenizer=MODEL_ID, device=device)
    start = time.perf_counter()
    generated = summarizer(
        articles,
        batch_size=args.batch_size,
        truncation=True,
        max_length=140,
        min_length=30,
        do_sample=False,
    )
    abstractive = [item["summary_text"] for item in generated]
    abstractive_seconds = time.perf_counter() - start
    start = time.perf_counter()
    extractive = [textrank_summarize(article) for article in articles]
    extractive_seconds = time.perf_counter() - start

    metrics = {
        "DistilBART": {
            **average_rouge(abstractive, references),
            "seconds": abstractive_seconds,
            "articles_per_second": len(data) / abstractive_seconds,
            "model_id": MODEL_ID,
            "device": device,
        },
        "TextRank": {
            **average_rouge(extractive, references),
            "seconds": extractive_seconds,
            "articles_per_second": len(data) / extractive_seconds,
        },
        "evaluation_articles": len(data),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "id": data["id"],
            "reference": references,
            "distilbart_summary": abstractive,
            "textrank_summary": extractive,
        }
    ).to_csv(args.output_dir / "generated_summaries.csv", index=False)
    save_plot({key: value for key, value in metrics.items() if isinstance(value, dict)}, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
