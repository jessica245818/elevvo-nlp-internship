"""Download and flatten the official SQuAD v1.1 development set."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd


DATASET_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
EXPECTED_QUESTIONS = 10570


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("task6_question_answering/data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_URL}")
    with urllib.request.urlopen(DATASET_URL) as response:
        payload = json.load(response)
    rows = []
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                rows.append(
                    {
                        "id": qa["id"],
                        "title": article["title"],
                        "context": paragraph["context"],
                        "question": qa["question"],
                        "answers": json.dumps([answer["text"] for answer in qa["answers"]]),
                    }
                )
    data = pd.DataFrame(rows)
    if payload.get("version") != "1.1" or len(data) != EXPECTED_QUESTIONS:
        raise ValueError(f"Unexpected SQuAD data: version={payload.get('version')}, rows={len(data)}")
    if data[["context", "question", "answers"]].isna().any().any():
        raise ValueError("SQuAD contains missing required values")
    output = args.output_dir / "squad_v1_1_dev.csv"
    data.to_csv(output, index=False)
    print(f"Saved {len(data):,} validated questions to {output}")


if __name__ == "__main__":
    main()
