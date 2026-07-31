"""Download a reproducible sample of CNN/DailyMail 3.0.0 test articles."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd


DATASET_URL = "https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/main/3.0.0/test-00000-of-00001.parquet"
EXPECTED_TEST_ROWS = 11490


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("task7_text_summarization/data"))
    parser.add_argument("--sample-size", type=int, default=50)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / "cnn_dailymail_test.parquet"
    if not parquet_path.exists():
        print(f"Downloading {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, parquet_path)
    data = pd.read_parquet(parquet_path)
    if len(data) != EXPECTED_TEST_ROWS or not {"article", "highlights", "id"}.issubset(data.columns):
        raise ValueError(f"Unexpected CNN/DailyMail test data: {len(data)} rows, {list(data.columns)}")
    sample = data.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
    output = args.output_dir / "evaluation_sample.csv"
    sample.to_csv(output, index=False)
    print(f"Validated {len(data):,} test articles and saved {len(sample)} sampled articles to {output}")


if __name__ == "__main__":
    main()
