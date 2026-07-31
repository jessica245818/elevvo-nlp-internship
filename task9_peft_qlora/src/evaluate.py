"""Compare base and adapter-tuned models with ROUGE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from task9_peft_qlora.src.train import MODEL_ID


def generate(model: object, tokenizer: object, dialogues: list[str], batch_size: int = 8) -> list[str]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    outputs = []
    for start in range(0, len(dialogues), batch_size):
        batch = tokenizer([f"summarize dialogue: {x}" for x in dialogues[start:start + batch_size]], max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            ids = model.generate(**batch, max_new_tokens=96, num_beams=4)
        outputs.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
    return outputs


def rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(("rouge1", "rouge2", "rougeL"), use_stemmer=True)
    return {metric: 100 * sum(scorer.score(ref, pred)[metric].fmeasure for pred, ref in zip(predictions, references)) / len(references) for metric in ("rouge1", "rouge2", "rougeL")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("task9_peft_qlora/data/test.parquet"))
    parser.add_argument("--adapter", type=Path, default=Path("task9_peft_qlora/artifacts/adapter"))
    parser.add_argument("--output-dir", type=Path, default=Path("task9_peft_qlora/artifacts"))
    parser.add_argument("--examples", type=int, default=100)
    args = parser.parse_args()
    data = pd.read_parquet(args.data).sample(n=args.examples, random_state=42).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    base_predictions = generate(base, tokenizer, data.dialogue.tolist())
    del base
    tuned = PeftModel.from_pretrained(AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID), args.adapter)
    tuned_predictions = generate(tuned, tokenizer, data.dialogue.tolist())
    metrics = {"examples": len(data), "base": rouge(base_predictions, data.summary.tolist()), "fine_tuned": rouge(tuned_predictions, data.summary.tolist())}
    metrics["improvement"] = {key: metrics["fine_tuned"][key] - metrics["base"][key] for key in metrics["base"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rouge_comparison.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"dialogue": data.dialogue, "reference": data.summary, "base_summary": base_predictions, "fine_tuned_summary": tuned_predictions}).to_csv(args.output_dir / "summary_comparison.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
