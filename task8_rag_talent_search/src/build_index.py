"""Embed resumes and build a persistent FAISS vector index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def searchable_text(row: object) -> str:
    return f"Skills: {row.skills}\nEducation: {row.education}\nExperience: {row.resume_text}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("task8_rag_talent_search/data/resumes.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("task8_rag_talent_search/artifacts"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data).fillna("")
    records = data.to_dict(orient="records")
    texts = [searchable_text(row) for row in data.itertuples()]
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, str(args.output_dir / "index.faiss"))
    (args.output_dir / "resume_metadata.json").write_text(json.dumps(records), encoding="utf-8")
    manifest = {"resumes": len(records), "dimensions": embeddings.shape[1], "embedding_model": EMBEDDING_MODEL}
    (args.output_dir / "index_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
