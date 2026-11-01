"""Interactive GraphRAG demonstration for the portfolio's Streamlit dialog."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from src.demo import (
    DemoBusyError, QueryComparison, get_demo_documents, list_demo_questions,
    run_demo_benchmark, run_demo_query,
)
from src.reporting import BenchmarkReport
from src.streamlit_ui import render_architecture, render_benchmark, render_intro, render_query_result


logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


@st.cache_data(ttl=300,max_entries=32,show_spinner=False)
def query_data(question: str) -> dict:
    return run_demo_query(question).model_dump(mode="json")


@st.cache_data(ttl=300,max_entries=1,show_spinner=False)
def benchmark_data() -> dict:
    return run_demo_benchmark().model_dump(mode="json")


def choose_example() -> None:
    st.session_state["question"] = st.session_state["example_question"]


def execute_question(question: str) -> None:
    try:
        with st.spinner("Retrieving passages and graph paths..."):
            comparison = QueryComparison.model_validate(query_data(question))
        st.session_state["comparison"] = comparison.model_dump(mode="json")
    except (ValueError,DemoBusyError) as error:
        st.error(str(error))
    except Exception:
        logger.exception("Interactive query could not complete")
        st.error("This request could not complete. Try again in a moment.")


def ask_view() -> None:
    examples = list_demo_questions()
    st.session_state.setdefault("question",examples[0])
    st.selectbox("Try an example question",examples,key="example_question",on_change=choose_example)
    with st.form("question_form",border=False):
        question = st.text_input("Question",key="question",max_chars=500)
        submitted = st.form_submit_button("Compare retrieval",type="primary")
    st.caption("Questions use the fixed corpus below. Custom questions may exceed the rule-based extractor's coverage.")
    if submitted or "comparison" not in st.session_state:
        execute_question(question)
    if "comparison" in st.session_state:
        render_query_result(QueryComparison.model_validate(st.session_state["comparison"]))
    with st.expander("Read the 11 source documents"):
        for document in get_demo_documents():
            st.markdown(f"**{document.title}**")
            st.write(document.text)
            st.divider()


def benchmark_view() -> None:
    st.subheader("Three-question benchmark",anchor=False)
    st.caption("Run both systems on all three built-in questions. Scores apply only to this small synthetic corpus.")
    if st.button("Run benchmark",type="primary"):
        try:
            with st.spinner("Evaluating both retrieval methods..."):
                report = BenchmarkReport.model_validate(benchmark_data())
            st.session_state["benchmark"] = report.model_dump(mode="json")
        except DemoBusyError as error:
            st.error(str(error))
        except Exception:
            logger.exception("Interactive benchmark could not complete")
            st.error("The benchmark could not complete. Try again in a moment.")
    if "benchmark" in st.session_state:
        render_benchmark(BenchmarkReport.model_validate(st.session_state["benchmark"]))


def main() -> None:
    st.set_page_config(page_title="Multi-Hop GraphRAG | CGC Labs",page_icon=str(ROOT/"src/assets/cgc-labs-logo.png"),layout="wide")
    render_intro()
    views = st.tabs(["Ask a question","Benchmark","Architecture"],key="view",on_change="rerun")
    if views[0].open:
        ask_view()
    if views[1].open:
        benchmark_view()
    if views[2].open:
        render_architecture()
    st.divider()
    st.caption("CGC Labs / Independent engineering demonstration · Python, Kuzu, LanceDB, Streamlit")


if __name__ == "__main__":
    main()
