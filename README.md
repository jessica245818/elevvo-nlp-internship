# Elevvo NLP Internship Projects

Completed projects:

1. **Task 1 — Sentiment Analysis:** this root project classifies IMDb reviews as positive or negative.
2. **Task 2 — News Category Classification:** see [`task2_news_classification/`](task2_news_classification/) for AG News multiclass classification.
3. **Task 3 — Fake News Detection:** see [`task3_fake_news_detection/`](task3_fake_news_detection/) for binary fake-versus-real article classification.

## Task 1: Sentiment Analysis

Binary sentiment classification of IMDb reviews using TF-IDF features. The project trains a logistic regression baseline, compares it with Multinomial Naive Bayes, evaluates both models, and visualizes frequent and predictive positive and negative terms.

## Project structure

```text
.
├── src/
│   ├── download_data.py
│   └── train.py
├── artifacts/                 # Generated metrics, plots, and models
├── data/                      # Downloaded dataset (not committed)
├── requirements.txt
└── README.md
```

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.download_data
python -m src.train
```

For a faster development run, limit the number of reviews loaded from each sentiment class in each split:

```bash
python -m src.train --max-per-class 1000
```

## Method

1. Load the balanced positive and negative IMDb train/test splits.
2. Lowercase reviews and remove HTML, punctuation, numbers, and English stopwords.
3. Convert cleaned text into unigram and bigram TF-IDF features.
4. Train logistic regression and Multinomial Naive Bayes classifiers.
5. Compare accuracy, precision, recall, and F1 score on the held-out official test set.
6. Save confusion matrices, a model-comparison chart, and the strongest positive/negative logistic-regression terms.

## Results

Results on the official 25,000-review test split:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 88.62% | 88.25% | 89.09% | 88.67% |
| Multinomial Naive Bayes | 85.81% | 87.57% | 83.46% | 85.47% |

Logistic regression performed best overall, exceeding Naive Bayes accuracy by 2.81 percentage points and producing the stronger F1 score.

## Generated outputs

Running the training script creates:

- `artifacts/metrics.json`
- `artifacts/logistic_regression_classification_report.txt`
- `artifacts/naive_bayes_classification_report.txt`
- `artifacts/logistic_regression_confusion_matrix.png`
- `artifacts/naive_bayes_confusion_matrix.png`
- `artifacts/model_comparison.png`
- `artifacts/most_frequent_sentiment_words.png`
- `artifacts/top_sentiment_words.png`
- trained `.joblib` model pipelines

## Dataset

This project uses the [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), containing 25,000 labeled training reviews and 25,000 labeled test reviews. The dataset is downloaded locally and excluded from Git.
