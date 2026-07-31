"""Search resumes from the command line and explain the top matches."""

from __future__ import annotations

import argparse
import json

from task8_rag_talent_search.src.engine import TalentSearchEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    engine = TalentSearchEngine()
    print(json.dumps(engine.search(args.query, args.top_k), indent=2))


if __name__ == "__main__":
    main()
