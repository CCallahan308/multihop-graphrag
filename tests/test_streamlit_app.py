"""Exercise the real Streamlit entry point and its public demonstration flows."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.demo import list_demo_questions
from src.reporting import BenchmarkReport


APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"
APP_TIMEOUT = 60


@pytest.fixture(autouse=True)
def fresh_app_cache():
    """Each test executes its initial query rather than inheriting cached output."""
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture
def app():
    return AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT).run(timeout=APP_TIMEOUT)


def click_button(app, label, *, view=None):
    button = next(item for item in app.button if item.label == label)
    button.click()
    if view is not None:
        # AppTest 1.57 has no tab selector and does not serialize keyed tab events.
        app.session_state["view"] = view
    return app.run(timeout=APP_TIMEOUT)


def captions(app):
    return [item.value for item in app.caption]


def test_initial_question_executes_real_retrieval(app):
    assert not app.exception
    assert not app.error
    result = app.session_state["comparison"]
    assert result["question"] == list_demo_questions()[0]
    assert result["graphrag"]["synthesis"]["final_answer"] == "Chicago"
    assert result["document_count"] == 11
    assert result["chunk_count"] == 11
    assert result["graphrag"]["retrieval"]["top_chunks"]
    assert result["plain_rag"]["retrieval"]["top_chunks"]
    assert "Chicago" in [item.value for item in app.subheader]
    assert len(app.get("download_button")) == 1


def test_selecting_three_hop_example_updates_the_question_and_answer(app):
    question = list_demo_questions()[2]
    app.selectbox(key="example_question").select(question).run(timeout=APP_TIMEOUT)
    assert app.text_input(key="question").value == question

    click_button(app, "Compare retrieval")

    assert not app.exception
    assert not app.error
    result = app.session_state["comparison"]
    assert result["question"] == question
    assert result["expected_answer"] == "United Kingdom"
    assert result["graphrag"]["synthesis"]["final_answer"] == "United Kingdom"
    assert f"Results for: {question}" in captions(app)


def test_benchmark_runs_all_three_questions_on_demand(app):
    app.session_state["view"] = "Benchmark"
    app.run(timeout=APP_TIMEOUT)
    assert "benchmark" not in app.session_state
    assert not app.metric

    click_button(app, "Run benchmark", view="Benchmark")

    assert not app.exception
    assert not app.error
    report = BenchmarkReport.model_validate(app.session_state["benchmark"])
    assert report.sample_count == 3
    assert report.document_count == 11
    assert report.graphrag.exact_match == 1.0
    assert report.graphrag.token_f1 == 1.0
    assert [record.predicted_answer for record in report.graphrag_records] == [
        "Chicago", "Grand Prix Spécial du Jury", "United Kingdom",
    ]
    assert [item.value for item in app.metric] == ["100%", "100%", "100%"]
    assert len(app.get("download_button")) == 2


def test_architecture_view_emits_both_diagrams_without_query_execution():
    app = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT)
    app.session_state["view"] = "Architecture"

    app.run(timeout=APP_TIMEOUT)

    assert not app.exception
    assert "comparison" not in app.session_state
    assert "Pipeline and storage" in [item.value for item in app.subheader]
    assert len(app.get("imgs")) == 2


def test_blank_question_shows_error_and_labels_the_retained_result(app):
    previous = app.session_state["comparison"]
    app.text_input(key="question").set_value("   ")

    click_button(app, "Compare retrieval")

    assert not app.exception
    assert [item.value for item in app.error] == ["Enter a question"]
    assert app.session_state["comparison"] == previous
    assert f"Results for: {previous['question']}" in captions(app)


def test_backend_failure_is_visible_without_fabricating_a_new_result(app, monkeypatch):
    import src.demo as demo

    previous = app.session_state["comparison"]

    def unavailable(_question):
        raise RuntimeError("Simulated retrieval service failure")

    monkeypatch.setattr(demo, "run_demo_query", unavailable)
    app.text_input(key="question").set_value("A new question that must use the unavailable backend")

    click_button(app, "Compare retrieval")

    assert not app.exception
    assert [item.value for item in app.error] == [
        "This request could not complete. Try again in a moment.",
    ]
    assert app.session_state["comparison"] == previous
    assert f"Results for: {previous['question']}" in captions(app)
