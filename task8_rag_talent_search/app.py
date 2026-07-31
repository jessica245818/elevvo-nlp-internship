"""Streamlit recruiter interface for the resume RAG engine."""

from __future__ import annotations

import streamlit as st

from task8_rag_talent_search.src.engine import TalentSearchEngine


st.set_page_config(page_title="Talent Search RAG", page_icon="🔎", layout="wide")
st.title("RAG-Powered Talent Search")
st.caption("Search anonymized resumes by job-relevant skills and experience. Human review is required.")


@st.cache_resource
def load_engine() -> TalentSearchEngine:
    return TalentSearchEngine()


engine = load_engine()
query = st.text_input("Recruiter request", "Find a junior data analyst who knows SQL and Tableau")
if st.button("Search candidates", type="primary"):
    with st.spinner("Retrieving and evaluating candidates..."):
        st.session_state.result = engine.search(query)

result = st.session_state.get("result")
if result:
    st.subheader("LLM evaluation")
    st.write(result["llm_evaluation"])
    st.subheader("Retrieved candidates")
    for candidate in result["candidates"]:
        with st.expander(f"{candidate['candidate_id']} — similarity {candidate['similarity']:.3f}"):
            st.write("Skills:", candidate["skills"])
            st.write("Education:", candidate["education"])
            st.write(candidate["resume_text"][:1800])
    st.subheader("Bias check")
    for warning in result["bias_check"]["warnings"]:
        st.warning(warning)
    candidate_ids = [candidate["candidate_id"] for candidate in result["candidates"]]
    selected = st.selectbox("Ask about a candidate", candidate_ids)
    follow_up = st.text_input("Follow-up question", "Does this candidate have leadership experience?")
    if st.button("Ask"):
        candidate = next(item for item in result["candidates"] if item["candidate_id"] == selected)
        st.write(engine.ask_candidate(candidate, follow_up))
