"""Download and validate data-only CoNLL-2003 Parquet splits."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd


URLS = {
    "train": "https://huggingface.co/api/datasets/lhoestq/conll2003/parquet/default/train/0.parquet",
    "validation": "https://huggingface.co/api/datasets/lhoestq/conll2003/parquet/default/validation/0.parquet",
    "test": "https://huggingface.co/api/datasets/lhoestq/conll2003/parquet/default/test/0.parquet",
}
EXPECTED_ROWS = {"train": 14_041, "validation": 3_250, "test": 3_453}
REQUIRED_COLUMNS = {"tokens", "ner_tags"}


def validate_split(path: Path, split: str) -> None:
    frame = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if len(frame) != EXPECTED_ROWS[split]:
        raise ValueError(
            f"Unexpected {split} row count: expected {EXPECTED_ROWS[split]:,}, got {len(frame):,}"
        )
    if not all(len(tokens) == len(tags) for tokens, tags in zip(frame["tokens"], frame["ner_tags"])):
        raise ValueError(f"Token/tag length mismatch in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("task4_named_entity_recognition/data"),
    )
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    for split, url in URLS.items():
        destination = args.data_dir / f"{split}.parquet"
        if not destination.exists():
            print(f"Downloading {split} split...")
            urllib.request.urlretrieve(url, destination)
        validate_split(destination, split)
        print(f"Validated {destination} ({EXPECTED_ROWS[split]:,} sentences)")


if __name__ == "__main__":
    main()
