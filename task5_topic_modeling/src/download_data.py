"""Download and convert the original BBC News archive to CSV."""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


DATASET_URL = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
CATEGORIES = {"business", "entertainment", "politics", "sport", "tech"}
EXPECTED_ARTICLES = 2225


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("task5_topic_modeling/data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_URL}")
    with urllib.request.urlopen(DATASET_URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    rows = []
    for member in sorted(archive.namelist()):
        parts = Path(member).parts
        if len(parts) != 3 or parts[0] != "bbc" or parts[1] not in CATEGORIES or not member.endswith(".txt"):
            continue
        text = archive.read(member).decode("latin-1").strip()
        rows.append({"category": parts[1], "text": text, "source_file": member})

    data = pd.DataFrame(rows)
    if len(data) != EXPECTED_ARTICLES or set(data["category"]) != CATEGORIES:
        raise ValueError(f"Unexpected dataset contents: {len(data)} articles, {set(data['category'])}")
    if data["text"].str.strip().eq("").any():
        raise ValueError("Dataset contains empty articles")
    output = args.output_dir / "bbc_news.csv"
    data.to_csv(output, index=False)
    print(f"Saved {len(data):,} validated articles to {output}")
    print(data["category"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
