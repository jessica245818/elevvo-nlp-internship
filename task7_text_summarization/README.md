# Task 7: Text Summarization Using Pre-trained Models

Abstractive news summarization with a pretrained DistilBART encoder–decoder model, evaluated against a TextRank extractive baseline on CNN/DailyMail 3.0.0.

## Approaches

- **DistilBART:** `sshleifer/distilbart-cnn-6-6`, fine-tuned for CNN/DailyMail summarization. Articles are tokenized and truncated to the model's 1,024-token input limit; summaries use deterministic beam-search generation.
- **TextRank:** an unsupervised extractive bonus baseline using TF-IDF sentence similarity and PageRank.
- **Evaluation:** ROUGE-1, ROUGE-2, and ROUGE-L F1 against journalist-written highlights.

## Setup and run

```bash
source .venv/bin/activate
pip install -r task7_text_summarization/requirements.txt
python -m task7_text_summarization.src.download_data
python -m task7_text_summarization.src.evaluate
```

The downloader validates the complete 11,490-article test split and creates a fixed 50-article evaluation sample.

## Results

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| **DistilBART** | **43.25** | **21.90** | **30.74** |
| TextRank | 33.97 | 14.18 | 22.75 |

DistilBART substantially outperformed the extractive baseline on every ROUGE measure, while TextRank was much faster (190.62 versus 0.22 articles per second). Results use the fixed 50-article sample with Apple Metal acceleration.

## Command-line summarizer

```bash
python -m task7_text_summarization.src.summarize --file article.txt
```

Use `--text "..."` for direct input. Long documents are truncated safely to the model input limit.

## Dataset

[CNN/DailyMail 3.0.0](https://huggingface.co/datasets/abisee/cnn_dailymail) contains over 300,000 news articles paired with highlights. The project uses the official 11,490-article test split and keeps downloaded data outside Git.

## Outputs

- `metrics.json` — ROUGE and runtime comparison
- `generated_summaries.csv` — reference, abstractive, and extractive summaries
- `rouge_comparison.png` — ROUGE-1/2/L comparison chart
