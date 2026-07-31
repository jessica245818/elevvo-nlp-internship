# Task 3: Fake News Detection

Binary classification of article titles and bodies as **Fake** or **Real** using lemmatized TF-IDF features.

## Features

- Combines each article's title and body as the model input
- Removes HTML, tokenizes, removes stopwords and newswire markers, and applies WordNet lemmatization
- Uses unigram and bigram TF-IDF feature engineering
- Trains and compares Logistic Regression and Linear SVM
- Reports accuracy, precision, recall, F1, classification reports, and confusion matrices
- Bonus fake-versus-real word-cloud visualization
- Includes saved model bundles and a prediction CLI

## Setup and run

From the repository root:

```bash
source .venv/bin/activate
pip install -r task3_fake_news_detection/requirements.txt
python -m task3_fake_news_detection.src.download_data
python -m task3_fake_news_detection.src.train
```

Quick development run:

```bash
python -m task3_fake_news_detection.src.train --max-rows 4000
```

Classify a new title and article body:

```bash
python -m task3_fake_news_detection.src.predict \
  "Article title followed by the article body."
```

## Dataset

This project uses Kaggle's [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), also known as the ISOT fake-news dataset. It provides separate `Fake.csv` and `True.csv` files containing article titles, bodies, subjects, and publication dates.

The downloader uses a public Git LFS mirror of those files and validates their schema and minimum row counts. Downloaded CSVs are excluded from Git.

## Important limitation

This model learns textual and source-style patterns in a historical dataset; it does **not** verify claims against evidence. The real articles largely originate from Reuters, while fake articles come from different publishers, so measured accuracy can partly reflect publisher style. It should not be treated as a general-purpose fact checker.

## Results

After removing empty and duplicated articles, the stratified 80/20 split contained 27,038 training and 6,760 test articles.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 98.34% | 97.89% | 99.48% | 98.68% |
| Linear SVM | 99.23% | 99.01% | 99.76% | 99.38% |

Linear SVM produced the best result. The unusually high score should be interpreted alongside the dataset source-style limitation above.

## Outputs

Training creates `task3_fake_news_detection/artifacts/` with:

- `metrics.json`
- per-model classification reports
- per-model confusion matrices
- `model_comparison.png`
- `fake_vs_real_wordclouds.png`
- reloadable `.joblib` model bundles (excluded from Git because of their size)
