"""Classify one article as fake or real with a trained Task 3 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Combined article title and body")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("task3_fake_news_detection/artifacts/linear_svm_bundle.joblib"),
    )
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    features = bundle["vectorizer"].transform([args.text])
    prediction = int(bundle["model"].predict(features)[0])
    print(bundle["class_names"][prediction])


if __name__ == "__main__":
    main()
