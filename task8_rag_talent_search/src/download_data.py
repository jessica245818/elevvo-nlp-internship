"""Download and normalize an anonymized resume corpus."""

from __future__ import annotations

import argparse
import ast
import urllib.request
from pathlib import Path

import pandas as pd


DATASET_URL = "https://huggingface.co/api/datasets/Tiger20111/Resumes/parquet/default/train/0.parquet"
EXPECTED_MINIMUM = 2400


def clean_collection(value: object) -> str:
    if not isinstance(value, str):
        return ""
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (set, list, tuple)):
            return ", ".join(sorted(str(item) for item in parsed))
    except (ValueError, SyntaxError):
        pass
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("task8_rag_talent_search/data"))
    parser.add_argument("--max-resumes", type=int, default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / "resumes.parquet"
    if not parquet_path.exists():
        print(f"Downloading {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, parquet_path)
    raw = pd.read_parquet(parquet_path)
    if len(raw) < EXPECTED_MINIMUM or not {"Skills", "Education", "Context"}.issubset(raw.columns):
        raise ValueError(f"Unexpected resume corpus: {len(raw)} rows, {list(raw.columns)}")
    data = raw.sample(n=min(args.max_resumes, len(raw)), random_state=42).reset_index(drop=True)
    data["candidate_id"] = [f"Candidate-{index:04d}" for index in range(1, len(data) + 1)]
    data["skills"] = data["Skills"].map(clean_collection)
    data["education"] = data["Education"].map(clean_collection)
    data["resume_text"] = data["Context"].map(clean_collection)
    data = data[["candidate_id", "skills", "education", "resume_text"]]
    if data["resume_text"].str.len().lt(20).any():
        raise ValueError("Resume corpus contains empty or malformed text")
    output = args.output_dir / "resumes.csv"
    data.to_csv(output, index=False)
    print(f"Validated {len(raw):,} source resumes; saved {len(data):,} anonymized resumes to {output}")


if __name__ == "__main__":
    main()
