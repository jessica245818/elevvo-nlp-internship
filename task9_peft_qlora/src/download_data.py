"""Download and validate SAMSum dialogue summarization splits."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd


URLS = {split: f"https://huggingface.co/api/datasets/Yuhthe/samsum/parquet/default/{split}/0.parquet" for split in ("train", "validation", "test")}
MINIMUM_ROWS = {"train": 14000, "validation": 800, "test": 800}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("task9_peft_qlora/data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, url in URLS.items():
        path = args.output_dir / f"{split}.parquet"
        if not path.exists():
            print(f"Downloading {split}...")
            urllib.request.urlretrieve(url, path)
        data = pd.read_parquet(path)
        if len(data) < MINIMUM_ROWS[split] or not {"dialogue", "summary"}.issubset(data.columns):
            raise ValueError(f"Unexpected {split} split: {len(data)} rows, {list(data.columns)}")
        print(f"Validated {split}: {len(data):,} examples")


if __name__ == "__main__":
    main()
