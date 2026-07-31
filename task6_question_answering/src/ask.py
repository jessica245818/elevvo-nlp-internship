"""Answer a question from a supplied context using a transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import pipeline


DEFAULT_MODEL = "distilbert/distilbert-base-cased-distilled-squad"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    context_group = parser.add_mutually_exclusive_group(required=True)
    context_group.add_argument("--context", help="Passage containing the answer")
    context_group.add_argument("--context-file", type=Path, help="UTF-8 passage file")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    context = args.context if args.context is not None else args.context_file.read_text(encoding="utf-8")
    answerer = pipeline("question-answering", model=args.model, tokenizer=args.model)
    result = answerer(question=args.question, context=context)
    print(json.dumps({key: result[key] for key in ("answer", "score", "start", "end")}, indent=2))


if __name__ == "__main__":
    main()
