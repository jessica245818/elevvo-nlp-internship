"""Evaluate whether semantic retrieval returns resumes containing requested skills."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from task8_rag_talent_search.src.engine import TalentSearchEngine


SKILLS = ("python", "sql", "excel", "java", "accounting", "sales", "marketing", "project management", "customer service", "leadership")


def contains_skill(candidate: dict, skill: str) -> bool:
    text = f"{candidate['skills']} {candidate['resume_text']}".lower()
    return re.search(rf"\b{re.escape(skill)}\b", text) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("task8_rag_talent_search/artifacts/retrieval_evaluation.json"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    engine = TalentSearchEngine(load_llm=False)
    results = []
    reciprocal_ranks = []
    for skill in SKILLS:
        candidates = engine.retrieve(f"Find a candidate with {skill} experience", args.top_k)
        matches = [index + 1 for index, candidate in enumerate(candidates) if contains_skill(candidate, skill)]
        reciprocal_rank = 1 / matches[0] if matches else 0.0
        reciprocal_ranks.append(reciprocal_rank)
        results.append(
            {
                "skill": skill,
                "hit_at_k": bool(matches),
                "first_relevant_rank": matches[0] if matches else None,
                "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
            }
        )
    report = {
        "queries": len(results),
        "top_k": args.top_k,
        "hit_rate_at_k": sum(item["hit_at_k"] for item in results) / len(results),
        "mean_reciprocal_rank": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "details": results,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
