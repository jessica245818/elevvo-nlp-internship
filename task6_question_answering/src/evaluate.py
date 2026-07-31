"""Compare fine-tuned transformer models on SQuAD v1.1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from transformers import pipeline

from task6_question_answering.src.metrics import score_prediction


MODELS = {
    "DistilBERT": "distilbert/distilbert-base-cased-distilled-squad",
    "TinyRoBERTa": "deepset/tinyroberta-squad2",
}


def best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def evaluate_model(name: str, model_id: str, data: pd.DataFrame, batch_size: int) -> tuple[dict, pd.DataFrame]:
    device = best_device()
    print(f"Loading {name} ({model_id}) on {device}")
    answerer = pipeline("question-answering", model=model_id, tokenizer=model_id, device=device)
    inputs = [{"question": row.question, "context": row.context} for row in data.itertuples()]
    start = time.perf_counter()
    predictions = answerer(inputs, batch_size=batch_size)
    elapsed = time.perf_counter() - start
    rows = []
    exact_total = f1_total = 0.0
    for source, prediction in zip(data.itertuples(), predictions):
        references = json.loads(source.answers)
        exact, f1 = score_prediction(prediction["answer"], references)
        exact_total += exact
        f1_total += f1
        rows.append(
            {
                "id": source.id,
                "model": name,
                "question": source.question,
                "prediction": prediction["answer"],
                "references": " | ".join(references),
                "confidence": prediction["score"],
                "exact_match": exact,
                "f1": f1,
            }
        )
    count = len(data)
    metrics = {
        "model_id": model_id,
        "examples": count,
        "exact_match": 100 * exact_total / count,
        "f1": 100 * f1_total / count,
        "seconds": elapsed,
        "examples_per_second": count / elapsed,
        "device": device,
    }
    print(f"{name}: EM={metrics['exact_match']:.2f} F1={metrics['f1']:.2f}")
    return metrics, pd.DataFrame(rows)


def save_plot(metrics: dict[str, dict], output_dir: Path) -> None:
    rows = [
        {"model": model, "metric": metric.replace("_", " ").title(), "score": values[metric]}
        for model, values in metrics.items()
        for metric in ("exact_match", "f1")
    ]
    plt.figure(figsize=(8, 5))
    axis = sns.barplot(data=pd.DataFrame(rows), x="metric", y="score", hue="model")
    axis.set_ylim(0, 100)
    axis.set_xlabel("")
    axis.set_ylabel("Score")
    axis.set_title("SQuAD v1.1 Transformer QA Comparison")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.1f")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("task6_question_answering/data/squad_v1_1_dev.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("task6_question_answering/artifacts"))
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data).sample(n=args.max_examples, random_state=42).reset_index(drop=True)
    all_metrics = {}
    all_predictions = []
    for name, model_id in MODELS.items():
        metrics, predictions = evaluate_model(name, model_id, data, args.batch_size)
        all_metrics[name] = metrics
        all_predictions.append(predictions)
    (args.output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    pd.concat(all_predictions, ignore_index=True).to_csv(args.output_dir / "predictions.csv", index=False)
    save_plot(all_metrics, args.output_dir)
    print(f"Saved evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
