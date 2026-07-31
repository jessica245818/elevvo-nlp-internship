# Task 8: RAG-Powered Talent Search Engine

A local Retrieval-Augmented Generation pipeline for semantic resume search. Recruiters enter a natural-language requirement, FAISS retrieves the three closest anonymized resumes, and an instruction-tuned FLAN-T5 language model explains the job-relevant evidence for and gaps in each match.

## Architecture

1. Normalize and anonymize 1,000 resumes from a 2,484-resume corpus.
2. Embed skill, education, and experience text with `all-MiniLM-L6-v2`.
3. Store normalized 384-dimensional vectors in a FAISS inner-product index.
4. Retrieve the top three candidates for a recruiter query.
5. Ground a local `google/flan-t5-small` LLM prompt in those resumes and generate an evidence-based explanation.

The system runs locally and requires no paid API key.

## Results

The retrieval benchmark issues ten job-skill queries covering Python, SQL, Excel, Java, accounting, sales, marketing, project management, customer service, and leadership.

| Metric | Result |
| --- | ---: |
| Hit rate@3 | **100%** |
| Mean reciprocal rank | **0.95** |
| Indexed resumes | 1,000 |
| Embedding dimensions | 384 |

The example “junior data analyst with SQL and Tableau” query returned three resumes with explicit data/analyst/SQL evidence, and two also contained Tableau evidence. Similarity scores are used only for retrieval; the local LLM produces the recruiter-facing explanation.

## Setup

```bash
source .venv/bin/activate
pip install -r task8_rag_talent_search/requirements.txt
python -m task8_rag_talent_search.src.download_data
python -m task8_rag_talent_search.src.build_index
```

## Command-line search

```bash
python -m task8_rag_talent_search.src.search \
  "Find a junior data analyst who knows SQL and Tableau"
```

The JSON output contains similarity-ranked candidates, the LLM evaluation, and the bias audit.

## Streamlit recruiter UI

```bash
streamlit run task8_rag_talent_search/app.py
```

The UI supports talent search, candidate evidence inspection, bias warnings, and grounded follow-up questions such as “Does this candidate have leadership experience?”

## Bias check

The audit flags configured demographic proxy terms in the recruiter query and retrieved evidence. The LLM prompt explicitly forbids demographic inference, and generated text containing a proxy is withheld by a post-generation guardrail. This is a safety check—not proof of fairness—and all hiring decisions require human review. The system intentionally exposes anonymized candidate IDs instead of names.

## Dataset

The recommended DataTurks Resume Entities dataset contains only 220 NER annotations, so this retrieval project uses the larger [Tiger20111/Resumes](https://huggingface.co/datasets/Tiger20111/Resumes) corpus (2,484 resumes) while preserving the recommended skills, education, and experience entity concepts. A reproducible 1,000-resume sample is used for the index.

## Generated artifacts

- `index_manifest.json` — index size, vector dimensions, and embedding model
- `example_search.json` — verified top-three retrieval, LLM explanation, and bias audit
- `retrieval_evaluation.json` — skill-query hit-rate benchmark

The generated FAISS index and resume metadata are excluded from Git and rebuilt with the commands above.
