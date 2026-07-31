# Task 2: News Category Classification

Multiclass classification of AG News articles into **World**, **Sports**, **Business**, and **Sci/Tech** using lemmatized TF-IDF features.

## Features

- Standard preprocessing: lowercasing, regex tokenization, stopword removal, and WordNet lemmatization
- Unigram and bigram TF-IDF feature engineering
- Multiclass Logistic Regression and Linear SVM comparison
- Accuracy and macro-averaged precision, recall, and F1 evaluation
- Classification reports and confusion matrices
- Bonus visualization of frequent and distinctive words for every category
- Saved, reloadable model bundles and a prediction CLI

## Setup and run

From the repository root:

```bash
source .venv/bin/activate
pip install -r task2_news_classification/requirements.txt
python -m task2_news_classification.src.download_data
python -m task2_news_classification.src.train
```

Quick development run:

```bash
python -m task2_news_classification.src.train --max-train-rows 20000 --max-test-rows 2000
```

Classify a new article after training:

```bash
python -m task2_news_classification.src.predict \
  "The company reported record quarterly revenue as technology shares rose."
```

## Dataset

The standard [AG News dataset](https://huggingface.co/datasets/fancyzhx/ag_news) contains 120,000 training and 7,600 test articles evenly distributed across four classes. The download script validates the expected columns, labels, and row counts.

## Results

Results on the official 7,600-article test split:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 91.47% | 91.47% | 91.47% | 91.46% |
| Linear SVM | 91.84% | 91.83% | 91.84% | 91.83% |

Linear SVM produced the best overall result, with balanced performance across all four categories.

## Outputs

Training creates `task2_news_classification/artifacts/` with:

- `metrics.json`
- per-model classification reports
- per-model confusion matrices
- `model_comparison.png`
- `top_words_per_category.png`
- reloadable `.joblib` model bundles (excluded from Git because of their size)
