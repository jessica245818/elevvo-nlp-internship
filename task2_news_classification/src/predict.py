"""Classify one news article with a trained Task 2 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="News headline or article text")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("task2_news_classification/artifacts/linear_svm_bundle.joblib"),
    )
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    features = bundle["vectorizer"].transform([args.text])
    prediction = int(bundle["model"].predict(features)[0])
    print(bundle["class_names"][prediction])


if __name__ == "__main__":
    main()
