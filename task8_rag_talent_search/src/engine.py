"""Reusable FAISS retrieval and local-LLM generation engine."""

from __future__ import annotations

import json
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from task8_rag_talent_search.src.bias import audit, find_proxies
from task8_rag_talent_search.src.build_index import EMBEDDING_MODEL


LLM_MODEL = "google/flan-t5-small"


class TalentSearchEngine:
    def __init__(self, artifact_dir: Path = Path("task8_rag_talent_search/artifacts"), load_llm: bool = True):
        self.index = faiss.read_index(str(artifact_dir / "index.faiss"))
        self.records = json.loads((artifact_dir / "resume_metadata.json").read_text(encoding="utf-8"))
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.generator = pipeline("text2text-generation", model=LLM_MODEL, tokenizer=LLM_MODEL) if load_llm else None

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        vector = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(vector, top_k)
        results = []
        for score, index in zip(scores[0], indices[0]):
            candidate = dict(self.records[index])
            candidate["similarity"] = float(score)
            results.append(candidate)
        return results

    def explain(self, query: str, candidates: list[dict]) -> str:
        if self.generator is None:
            raise RuntimeError("The LLM was not loaded")
        explanations = []
        for candidate in candidates:
            evidence_text = f"{candidate['skills']} {candidate['resume_text']}".lower()
            query_terms = [
                term
                for term in re.findall(r"[a-z+#.]{3,}", query.lower())
                if term not in {"find", "junior", "candidate", "knows", "with", "who", "and"}
            ]
            matched_terms = sorted({term for term in query_terms if term in evidence_text})
            prompt = (
                "Write one factual hiring-fit sentence using only the evidence. Never infer demographics.\n"
                f"Job request: {query}\nMatched request terms: {', '.join(matched_terms) or 'none explicit'}\n"
                f"Candidate evidence: Skills: {candidate['skills'][:500]}. "
                f"Experience: {candidate['resume_text'][:350]}\nSentence:"
            )
            evaluation = self.generator(
                prompt, max_new_tokens=90, do_sample=False, truncation=True
            )[0]["generated_text"]
            if find_proxies(evaluation):
                evaluation = "LLM output withheld by the demographic-proxy guardrail; review the retrieved evidence."
            explanations.append(
                f"{candidate['candidate_id']} — matched terms: {', '.join(matched_terms) or 'semantic match only'}. "
                f"LLM evidence summary: {evaluation}"
            )
        return "\n".join(explanations)

    def search(self, query: str, top_k: int = 3) -> dict:
        candidates = self.retrieve(query, top_k)
        return {"query": query, "candidates": candidates, "llm_evaluation": self.explain(query, candidates), "bias_check": audit(query, candidates)}

    def ask_candidate(self, candidate: dict, question: str) -> str:
        prompt = (
            "Answer using only this resume. If the resume does not say, answer 'Not stated in the resume.'\n"
            f"Resume: {candidate['resume_text'][:1800]}\nQuestion: {question}\nAnswer:"
        )
        return self.generator(prompt, max_new_tokens=100, do_sample=False, truncation=True)[0]["generated_text"]
