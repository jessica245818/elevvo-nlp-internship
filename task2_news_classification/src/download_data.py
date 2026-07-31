"""Download the standard AG News train and test splits from Hugging Face."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd


URLS = {
    "train": "https://huggingface.co/api/datasets/fancyzhx/ag_news/parquet/default/train/0.parquet",
    "test": "https://huggingface.co/api/datasets/fancyzhx/ag_news/parquet/default/test/0.parquet",
}
EXPECTED_ROWS = {"train": 120_000, "test": 7_600}


def validate_split(path: Path, split: str) -> None:
    frame = pd.read_parquet(path)
    required = {"text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    if len(frame) != EXPECTED_ROWS[split]:
        raise ValueError(
            f"Unexpected {split} row count: expected {EXPECTED_ROWS[split]:,}, got {len(frame):,}"
        )
    if set(frame["label"].unique()) != {0, 1, 2, 3}:
        raise ValueError(f"Unexpected labels in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("task2_news_classification/data"),
    )
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    for split, url in URLS.items():
        destination = args.data_dir / f"{split}.parquet"
        if not destination.exists():
            print(f"Downloading {split} split...")
            urllib.request.urlretrieve(url, destination)
        validate_split(destination, split)
        print(f"Validated {destination} ({EXPECTED_ROWS[split]:,} rows)")


if __name__ == "__main__":
    main()
