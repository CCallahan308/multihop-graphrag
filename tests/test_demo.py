"""Interactive operations use actual retrieval with bounded, isolated resources."""

from pathlib import Path
from threading import BoundedSemaphore
from unittest.mock import MagicMock

import pytest

from src import demo, pipeline, storage
from src.demo import (
    BUFFER_POOL_BYTES, QUERY_THREADS, DemoBusyError,
    get_demo_documents, list_demo_questions, run_demo_benchmark, run_demo_query,
    validate_question,
)
from src.pipeline import MultiHopGraphRAGSystem
from src.storage import KuzuGraphStore


@pytest.mark.parametrize(
    ("question_index", "answer"),
    [(0, "Chicago"), (1, "Grand Prix Spécial du Jury"), (2, "United Kingdom")],
)
def test_live_demo_questions_return_answers_and_retrieved_evidence(question_index, answer):
    question = list_demo_questions()[question_index]
    result = run_demo_query(question)

    assert result.question == question
    assert result.graphrag.synthesis.final_answer == answer
    assert result.expected_answer == answer
    assert result.document_count == result.chunk_count == 11
    assert result.graphrag.retrieval.subgraph_paths
    assert result.graphrag.retrieval.top_chunks
    assert all(chunk.text for chunk in result.graphrag.retrieval.top_chunks)
    assert result.plain_rag.retrieval.query == question
    assert result.plain_rag.retrieval.top_chunks
    assert result.model_validate_json(result.model_dump_json()) == result


def test_live_benchmark_evaluates_all_builtin_questions():
    report = run_demo_benchmark()

    assert report.dataset_kind == "synthetic"
    assert report.document_count == 11 and report.sample_count == 3
    assert report.graphrag.exact_match == 1.0
    assert report.graphrag.token_f1 == 1.0
    assert len(report.plain_rag_records) == 3


def test_custom_question_is_compared_without_inventing_a_reference_answer():
    result = run_demo_query("  Who founded Company X?  ")

    assert result.question == "Who founded Company X?"
    assert result.expected_answer is None
    assert result.graphrag.synthesis.final_answer == "John Doe"


@pytest.mark.parametrize("question", [None, 42, True, "", "   ", "line\nbreak", "null\x00byte", "x" * 501, "\ud800"])
def test_invalid_question_is_rejected_before_store_creation(question, monkeypatch):
    def forbidden_store():
        pytest.fail("Invalid input reached database initialization")

    monkeypatch.setattr(demo, "_demo_system", forbidden_store)
    with pytest.raises(ValueError):
        run_demo_query(question)


def test_question_limit_and_unicode_are_explicit():
    assert validate_question("x" * 500) == "x" * 500
    assert validate_question("  Où est le café?  ") == "Où est le café?"


def test_documents_are_fresh_public_models():
    documents = get_demo_documents()
    assert len(documents) == 11
    documents[0].text = "caller changes"
    assert get_demo_documents()[0].text != "caller changes"


def test_exhausted_work_slots_fail_without_creating_a_store(monkeypatch):
    slots = BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(demo, "_WORK_SLOTS", slots)
    monkeypatch.setattr(demo, "WORKER_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(demo, "MultiHopGraphRAGSystem", lambda **kwargs: pytest.fail("Busy run opened a store"))
    try:
        with pytest.raises(DemoBusyError, match="busy"):
            run_demo_query(list_demo_questions()[0])
    finally:
        slots.release()


def test_indexing_failure_closes_stores_removes_files_and_releases_slot(monkeypatch):
    captured = []
    slots = BoundedSemaphore(1)
    monkeypatch.setattr(demo, "_WORK_SLOTS", slots)

    class ObservedSystem(MultiHopGraphRAGSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.append(self)

        def extract_and_index_knowledge(self, chunks):
            raise RuntimeError("Indexing failed")

    monkeypatch.setattr(demo, "MultiHopGraphRAGSystem", ObservedSystem)
    with pytest.raises(RuntimeError, match="Indexing failed"):
        run_demo_query(list_demo_questions()[0])

    system = captured[0]
    assert system.graph_store.db.is_closed
    assert not system.vector_store.db.is_open()
    assert not Path(system.storage_dir).parent.exists()
    assert slots.acquire(blocking=False)
    slots.release()


def test_pipeline_context_closes_native_stores_and_accepts_resource_limits(tmp_path):
    with MultiHopGraphRAGSystem(
        str(tmp_path / "store"), clean_start=False,
        buffer_pool_size=BUFFER_POOL_BYTES, max_num_threads=QUERY_THREADS,
    ) as system:
        assert system.graph_store.db.buffer_pool_size == BUFFER_POOL_BYTES
        assert system.graph_store.db.max_num_threads == QUERY_THREADS
        assert system.vector_store.db.is_open()

    assert system.graph_store.db.is_closed
    assert not system.vector_store.db.is_open()
    system.close()


def test_partial_pipeline_initialization_preserves_original_failure(tmp_path, monkeypatch, caplog):
    graph = MagicMock()
    graph.close.side_effect = RuntimeError("Cleanup also failed")
    monkeypatch.setattr(pipeline, "KuzuGraphStore", lambda **kwargs: graph)

    def fail_vector_store(**kwargs):
        raise OSError("Vector store unavailable")

    monkeypatch.setattr(pipeline, "LanceVectorStore", fail_vector_store)
    with pytest.raises(OSError, match="Vector store unavailable"):
        MultiHopGraphRAGSystem(str(tmp_path / "store"))

    graph.close.assert_called_once()
    assert "cleanup failed" in caplog.text.lower()


def test_graph_connection_failure_closes_database(tmp_path, monkeypatch):
    database = MagicMock()
    monkeypatch.setattr(storage.kuzu, "Database", lambda *args, **kwargs: database)

    def fail_connection(*args):
        raise RuntimeError("Connection unavailable")

    monkeypatch.setattr(storage.kuzu, "Connection", fail_connection)
    with pytest.raises(RuntimeError, match="Connection unavailable"):
        KuzuGraphStore(str(tmp_path / "graph"))
    database.close.assert_called_once()


@pytest.mark.parametrize("limits", [{"buffer_pool_size": -1}, {"buffer_pool_size": True}, {"max_num_threads": "2"}, {"max_num_threads": 1.5}])
def test_invalid_resource_limits_never_reset_existing_data(tmp_path, limits):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("existing data", encoding="utf-8")

    with pytest.raises(ValueError, match="nonnegative integer"):
        MultiHopGraphRAGSystem(str(tmp_path), **limits)
    assert sentinel.read_text(encoding="utf-8") == "existing data"
