"""Bounded, temporary-store operations for the public interactive demonstration."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore
from typing import Iterator
import unicodedata

from pydantic import BaseModel, ConfigDict, Field

from src.dataset import MuSiQueDatasetParser
from src.embeddings import HashProjectionEmbedding
from src.models import Document, EvaluationSample, RetrievalResult, SynthesisOutput
from src.pipeline import MultiHopGraphRAGSystem
from src.reporting import BenchmarkReport, build_report


MAX_QUESTION_CHARS = 500
MAX_CONCURRENT_OPERATIONS = 2
WORKER_WAIT_SECONDS = 2.0
BUFFER_POOL_BYTES = 64 * 1024 * 1024
QUERY_THREADS = 2
_WORK_SLOTS = BoundedSemaphore(MAX_CONCURRENT_OPERATIONS)


class DemoBusyError(RuntimeError):
    """The demo's bounded processing slots were unavailable within the wait limit."""


class QueryResult(BaseModel):
    """Actual retrieval and synthesis from one system."""

    model_config = ConfigDict(extra="forbid")
    retrieval: RetrievalResult
    synthesis: SynthesisOutput


class QueryComparison(BaseModel):
    """A comparison over one public corpus, without live database handles."""

    model_config = ConfigDict(extra="forbid")
    question: str
    graphrag: QueryResult
    plain_rag: QueryResult
    expected_answer: str | None = None
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)


def list_demo_questions() -> list[str]:
    """Return the three built-in questions in their benchmark order."""
    _, samples = MuSiQueDatasetParser.create_mock_musique_benchmark()
    return [sample.question for sample in samples]


def get_demo_documents() -> list[Document]:
    """Return fresh models containing only the built-in public synthetic passages."""
    documents, _ = MuSiQueDatasetParser.create_mock_musique_benchmark()
    return documents


def validate_question(question: str) -> str:
    """Accept one nonempty line, up to 500 Unicode characters, without controls."""
    if not isinstance(question, str):
        raise ValueError("Question must be text")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"Question must contain at most {MAX_QUESTION_CHARS} characters")
    if any(unicodedata.category(character) in {"Cc", "Cs"} for character in question):
        raise ValueError("Question must be a single line without control characters")
    normalized = question.strip()
    if not normalized:
        raise ValueError("Enter a question")
    return normalized


@contextmanager
def _demo_system() -> Iterator[tuple[MultiHopGraphRAGSystem, list[Document], list[EvaluationSample], int]]:
    """Keep each operation's stores private and release them before file cleanup."""
    if not _WORK_SLOTS.acquire(timeout=WORKER_WAIT_SECONDS):
        raise DemoBusyError("The demo is busy. Wait a moment and try again.")
    try:
        documents, samples = MuSiQueDatasetParser.create_mock_musique_benchmark()
        with TemporaryDirectory(prefix="graphrag-demo-") as directory:
            # The child does not exist yet. No caller-selected directory is reset.
            with MultiHopGraphRAGSystem(
                storage_dir=str(Path(directory) / "store"),
                embedding_provider=HashProjectionEmbedding(dimension=384),
                clean_start=False,
                buffer_pool_size=BUFFER_POOL_BYTES,
                max_num_threads=QUERY_THREADS,
            ) as system:
                chunks = system.ingest_documents(documents)
                system.extract_and_index_knowledge(chunks)
                yield system, documents, samples, len(chunks)
    finally:
        _WORK_SLOTS.release()


def run_demo_query(question: str) -> QueryComparison:
    """Evaluate a bounded question against graph-plus-vector and dense retrieval."""
    normalized = validate_question(question)
    with _demo_system() as (system, documents, samples, chunk_count):
        graph_retrieval, graph_synthesis = system.query_graphrag(normalized)
        dense_retrieval, dense_synthesis = system.query_plain_rag(normalized)
        expected = next(
            (sample.gold_answer for sample in samples if sample.question.casefold() == normalized.casefold()),
            None,
        )
        return QueryComparison(
            question=normalized,
            graphrag=QueryResult(retrieval=graph_retrieval, synthesis=graph_synthesis),
            plain_rag=QueryResult(retrieval=dense_retrieval, synthesis=dense_synthesis),
            expected_answer=expected,
            document_count=len(documents),
            chunk_count=chunk_count,
        )


def run_demo_benchmark() -> BenchmarkReport:
    """Execute all three built-in cases, preserving both systems' actual records."""
    with _demo_system() as (system, documents, samples, _):
        return build_report(
            system.run_benchmark(samples),
            dataset_kind="synthetic",
            dataset_label="Built-in synthetic smoke test",
            document_count=len(documents),
            embedding_provider="Hash projection, 384 dimensions",
            generation_mode="Heuristic span extraction",
        )
