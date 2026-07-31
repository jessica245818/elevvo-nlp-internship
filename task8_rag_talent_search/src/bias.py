"""Transparent demographic-proxy checks for recruiter queries and retrieved text."""

from __future__ import annotations

import re


PROXY_GROUPS = {
    "age": ("young", "older", "age", "years old", "recent graduate"),
    "gender": ("woman", "women", "female", "man", "men", "male", "mother", "father"),
    "race_or_ethnicity": ("race", "racial", "ethnicity", "ethnic", "nationality"),
    "religion": ("religion", "religious", "christian", "muslim", "jewish", "hindu"),
    "disability": ("disabled", "disability", "wheelchair", "medical condition"),
    "family_status": ("married", "single", "pregnant", "children", "childcare"),
}


def find_proxies(text: str) -> dict[str, list[str]]:
    lowered = text.lower()
    return {
        group: [term for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered)]
        for group, terms in PROXY_GROUPS.items()
        if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in terms)
    }


def audit(query: str, retrieved: list[dict]) -> dict:
    query_flags = find_proxies(query)
    resume_flags = {
        candidate["candidate_id"]: find_proxies(candidate["resume_text"])
        for candidate in retrieved
        if find_proxies(candidate["resume_text"])
    }
    warnings = []
    if query_flags:
        warnings.append("The query contains demographic proxies; remove them and use job-relevant criteria.")
    if resume_flags:
        warnings.append("Retrieved text contains demographic proxies; do not use them in hiring decisions.")
    if not warnings:
        warnings.append("No configured demographic proxy terms were detected. This is a screening aid, not proof of fairness.")
    return {"query_flags": query_flags, "resume_flags": resume_flags, "warnings": warnings}
