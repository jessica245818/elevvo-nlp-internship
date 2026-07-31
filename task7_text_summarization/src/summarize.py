"""Summarize text supplied directly or through a UTF-8 file."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import pipeline


MODEL_ID = "sshleifer/distilbart-cnn-6-6"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", type=Path)
    parser.add_argument("--max-length", type=int, default=140)
    args = parser.parse_args()
    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    summarizer = pipeline("summarization", model=MODEL_ID, tokenizer=MODEL_ID)
    result = summarizer(text, truncation=True, max_length=args.max_length, min_length=20, do_sample=False)
    print(result[0]["summary_text"])


if __name__ == "__main__":
    main()
