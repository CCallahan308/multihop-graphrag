"""Validated benchmark records and a compact terminal presentation."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from numbers import Real
from typing import Annotated, Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints, field_validator, model_validator

from src.models import EvaluationMetrics


DatasetKind = Literal["synthetic", "jsonl", "huggingface"]
NonEmptyText = Annotated[str, StringConstraints(strict=True, min_length=1)]
Score = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
METRIC_FIELDS = ("multi_hop_recall", "mrr_at_5", "exact_match", "token_f1", "faithfulness")


def _validate_score(value: object) -> float:
    """Accept finite numeric proportions; never coerce missing or text values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Scores must be numeric proportions between 0 and 1")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("Scores must be finite proportions between 0 and 1")
    return number


class ReportMetrics(EvaluationMetrics):
    """Aggregate proportions in [0, 1], with a nonnegative sample count."""

    model_config = ConfigDict(extra="forbid")
    total_samples: StrictInt = Field(ge=0)
    multi_hop_recall: Score
    mrr_at_5: Score
    exact_match: Score
    token_f1: Score
    faithfulness: Score

    @field_validator(*METRIC_FIELDS, mode="before")
    @classmethod
    def validate_score(cls, value: object) -> float:
        return _validate_score(value)


class BenchmarkRecord(BaseModel):
    """One evaluated question and its retrieved evidence for one system."""

    model_config = ConfigDict(extra="forbid", strict=True)
    sample_id: NonEmptyText
    question: NonEmptyText
    gold_answer: str
    predicted_answer: str
    gold_chunks: list[NonEmptyText]
    retrieved_chunks: list[NonEmptyText]
    multi_hop_recall: Score
    mrr_at_5: Score
    exact_match: Score
    token_f1: Score
    faithfulness: Score

    @field_validator(*METRIC_FIELDS, mode="before")
    @classmethod
    def validate_score(cls, value: object) -> float:
        return _validate_score(value)


class BenchmarkReport(BaseModel):
    """Versioned export of a single comparative run, including both systems."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    generated_at: datetime
    dataset_kind: DatasetKind
    dataset_label: NonEmptyText
    sample_count: StrictInt = Field(ge=0)
    document_count: StrictInt = Field(ge=0)
    embedding_provider: NonEmptyText
    generation_mode: NonEmptyText
    graphrag: ReportMetrics
    plain_rag: ReportMetrics
    graphrag_records: list[BenchmarkRecord]
    plain_rag_records: list[BenchmarkRecord]

    @field_validator("generated_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_comparison(self) -> "BenchmarkReport":
        for metrics, records in (
            (self.graphrag, self.graphrag_records),
            (self.plain_rag, self.plain_rag_records),
        ):
            if metrics.total_samples != self.sample_count or len(records) != self.sample_count:
                raise ValueError("Both systems must contain exactly sample_count evaluated records")
            sample_ids = [record.sample_id for record in records]
            if len(set(sample_ids)) != len(sample_ids):
                raise ValueError("Sample identifiers must be unique within each system")
            for field in METRIC_FIELDS:
                # Empty evaluations use zero aggregates, rather than a success rate.
                average = sum(getattr(record, field) for record in records) / len(records) if records else 0.0
                if not math.isclose(getattr(metrics, field), average, rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError(f"Aggregate {field} does not match the evaluated records")

        baseline = {record.sample_id: record for record in self.plain_rag_records}
        for record in self.graphrag_records:
            peer = baseline.get(record.sample_id)
            if peer is None or (record.question, record.gold_answer, record.gold_chunks) != (
                peer.question, peer.gold_answer, peer.gold_chunks
            ):
                raise ValueError("Both systems must evaluate the same questions and gold evidence")
        return self


def build_report(
    results: Mapping[str, Any],
    *,
    dataset_kind: DatasetKind,
    dataset_label: str,
    document_count: int,
    embedding_provider: str,
    generation_mode: str,
    generated_at: datetime | None = None,
) -> BenchmarkReport:
    """Validate pipeline output before rendering or exporting any result."""
    graphrag = results["graphrag"]
    baseline = results["plain_rag"]
    graph_metrics = graphrag.model_dump() if isinstance(graphrag, EvaluationMetrics) else graphrag
    baseline_metrics = baseline.model_dump() if isinstance(baseline, EvaluationMetrics) else baseline
    return BenchmarkReport.model_validate({
        "generated_at": generated_at or datetime.now(timezone.utc),
        "dataset_kind": dataset_kind,
        "dataset_label": dataset_label,
        "sample_count": graph_metrics["total_samples"],
        "document_count": document_count,
        "embedding_provider": embedding_provider,
        "generation_mode": generation_mode,
        "graphrag": graph_metrics,
        "plain_rag": baseline_metrics,
        "graphrag_records": results["graphrag_records"],
        "plain_rag_records": results["baseline_records"],
    })


def _terminal_text(value: str) -> str:
    """Keep untrusted question text on one line and strip terminal controls."""
    return " ".join("".join(character for character in value if character.isprintable() or character.isspace()).split())


def render_console(report: BenchmarkReport) -> str:
    """Render actual values as aligned text without terminal or color dependencies."""
    kind = "Synthetic smoke test" if report.dataset_kind == "synthetic" else report.dataset_label
    question_label = "question" if report.sample_count == 1 else "questions"
    document_label = "document" if report.document_count == 1 else "documents"
    lines = [
        "",
        "MULTI-HOP GRAPHRAG  /  EVALUATION",
        f"{_terminal_text(kind)}  |  {report.sample_count} {question_label}  |  {report.document_count} {document_label}",
        f"{_terminal_text(report.embedding_provider)}  /  {_terminal_text(report.generation_mode)}",
        "",
        f"{'Metric':<29} {'GraphRAG':>11} {'Plain RAG':>11}",
        "-" * 53,
    ]
    for title, field in (
        ("All-support recall", "multi_hop_recall"),
        ("MRR@5", "mrr_at_5"),
        ("Answer exact match", "exact_match"),
        ("Answer token F1", "token_f1"),
        ("Token-overlap grounding", "faithfulness"),
    ):
        graph_value = getattr(report.graphrag, field)
        baseline_value = getattr(report.plain_rag, field)
        if field == "mrr_at_5":
            left, right = f"{graph_value:.4f}", f"{baseline_value:.4f}"
        else:
            left, right = f"{graph_value:.2%}", f"{baseline_value:.2%}"
        lines.append(f"{title:<29} {left:>11} {right:>11}")

    lines.extend(["", "Question evidence"])
    peers = {record.sample_id: record for record in report.plain_rag_records}
    for number, record in enumerate(report.graphrag_records, start=1):
        peer = peers[record.sample_id]
        lines.extend([
            f"  {number:02d}  {_terminal_text(record.question)}",
            f"      GraphRAG: {_terminal_text(record.predicted_answer) or '(empty answer)'}",
            f"      Plain RAG: {_terminal_text(peer.predicted_answer) or '(empty answer)'}",
            f"      Reference: {_terminal_text(record.gold_answer) or '(empty reference)'}",
        ])
    if not report.sample_count:
        lines.append("  No questions were evaluated; aggregate scores are zero.")
    if report.dataset_kind == "synthetic":
        lines.extend(["", "Built-in synthetic scenarios. These scores do not measure MuSiQue validation performance."])
    lines.append("Grounding measures token overlap with retrieved text; it is not a factual correctness judgment.")
    return "\n".join(lines)
