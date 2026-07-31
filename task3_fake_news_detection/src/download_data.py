"""Download and validate the Kaggle Fake and Real News Dataset files."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd


# Git LFS mirror of Kaggle's clmentbisaillon/fake-and-real-news-dataset files.
URLS = {
    "Fake.csv": "https://media.githubusercontent.com/media/timooo-thy/fake-real-news-classifier/master/Fake.csv",
    "True.csv": "https://media.githubusercontent.com/media/timooo-thy/fake-real-news-classifier/master/True.csv",
}
REQUIRED_COLUMNS = {"title", "text", "subject", "date"}


def validate_file(path: Path) -> int:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if len(frame) < 20_000:
        raise ValueError(f"Unexpectedly small dataset file: {path} ({len(frame):,} rows)")
    return len(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("task3_fake_news_detection/data"),
    )
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in URLS.items():
        destination = args.data_dir / filename
        if not destination.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, destination)
        rows = validate_file(destination)
        print(f"Validated {destination} ({rows:,} rows)")


if __name__ == "__main__":
    main()
