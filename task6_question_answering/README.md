# Task 6: Question Answering with Transformers

Extractive question answering on SQuAD v1.1 with pretrained transformer models fine-tuned to select an answer span from a supplied context.

## Models

- **DistilBERT:** `distilbert/distilbert-base-cased-distilled-squad`, fine-tuned on SQuAD v1.1.
- **TinyRoBERTa:** `deepset/tinyroberta-squad2`, a compact RoBERTa model included for the bonus architecture comparison.

## Setup and evaluation

```bash
source .venv/bin/activate
pip install -r task6_question_answering/requirements.txt
python -m task6_question_answering.src.download_data
python -m task6_question_answering.src.evaluate
```

The reproducible comparison uses a fixed random sample of 500 official development questions. Increase `--max-examples` to evaluate a larger sample or all 10,570 questions. The evaluator automatically uses Apple Metal, CUDA, or CPU and reports official-style normalized Exact Match and token-level F1 scores.

## Results

| Model | Exact Match | F1 | Throughput |
| --- | ---: | ---: | ---: |
| DistilBERT | 78.40 | 86.35 | 11.32 examples/s |
| **TinyRoBERTa** | **83.40** | **90.05** | 8.22 examples/s |

TinyRoBERTa was 5.0 points higher in Exact Match and 3.70 points higher in F1. DistilBERT was faster, illustrating the accuracy–latency tradeoff. Results were measured on the fixed 500-question sample using Apple Metal acceleration.

## Command-line interface

```bash
python -m task6_question_answering.src.ask \
  "Where is the Eiffel Tower?" \
  --context "The Eiffel Tower is a landmark in Paris, France."
```

For long passages, provide `--context-file article.txt`. Output includes the extracted answer, confidence, and character offsets.

## Dataset

[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) contains over 100,000 crowd-written questions whose answers are spans in Wikipedia passages. The project downloads and validates the official 10,570-question development set.

## Outputs

- `metrics.json` — Exact Match, F1, speed, model, and device details
- `predictions.csv` — questions, predicted spans, references, confidence, and per-example scores
- `model_comparison.png` — DistilBERT versus TinyRoBERTa
