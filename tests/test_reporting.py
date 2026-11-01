"""Validation and boundary checks for shareable benchmark results."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from scripts.run_musique_eval import main, parse_args
from src.models import EvaluationMetrics
from src.reporting import BenchmarkReport, build_report, render_console


@pytest.fixture
def pipeline_results():
    record = {
        "sample_id": "sample_01",
        "question": "Which city was the founder of Company X born in?",
        "gold_answer": "Chicago",
        "predicted_answer": "Chicago",
        "gold_chunks": ["founder", "birthplace"],
        "retrieved_chunks": ["founder", "birthplace"],
        "multi_hop_recall": 1.0,
        "mrr_at_5": 1.0,
        "exact_match": 1.0,
        "token_f1": 1.0,
        "faithfulness": 1.0,
    }
    metrics = EvaluationMetrics(
        total_samples=1, multi_hop_recall=1.0, mrr_at_5=1.0,
        exact_match=1.0, token_f1=1.0, faithfulness=1.0,
    )
    return {
        "graphrag": metrics,
        "plain_rag": metrics.model_copy(deep=True),
        "graphrag_records": [record],
        "baseline_records": [deepcopy(record)],
    }


def make_report(results, **overrides):
    metadata = {
        "dataset_kind": "synthetic",
        "dataset_label": "Built-in synthetic smoke test",
        "document_count": 2,
        "embedding_provider": "Hash projection, 384 dimensions",
        "generation_mode": "Heuristic span extraction",
        "generated_at": datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
    }
    metadata.update(overrides)
    return build_report(results, **metadata)


def test_report_json_round_trip_preserves_both_systems(pipeline_results):
    report = make_report(pipeline_results)

    restored = BenchmarkReport.model_validate_json(report.model_dump_json())

    assert restored == report
    assert restored.sample_count == 1
    assert restored.schema_version == 1
    assert restored.plain_rag_records[0].retrieved_chunks == ["founder", "birthplace"]
    assert restored.generated_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.1, 1.1, None, "1.0", True])
def test_report_rejects_invalid_record_scores(pipeline_results, value):
    pipeline_results["graphrag_records"][0]["token_f1"] = value

    with pytest.raises(ValidationError, match="Scores must"):
        make_report(pipeline_results)


def test_report_rejects_nonfinite_aggregate_scores(pipeline_results):
    # EvaluationMetrics itself currently permits non-finite values.
    pipeline_results["graphrag"].exact_match = float("nan")

    with pytest.raises(ValidationError, match="finite proportions"):
        make_report(pipeline_results)


def test_report_rejects_mismatched_record_count(pipeline_results):
    pipeline_results["baseline_records"] = []

    with pytest.raises(ValidationError, match="sample_count"):
        make_report(pipeline_results)


def test_report_rejects_duplicate_sample_identifiers(pipeline_results):
    for key in ("graphrag_records", "baseline_records"):
        pipeline_results[key].append(deepcopy(pipeline_results[key][0]))
    for key in ("graphrag", "plain_rag"):
        pipeline_results[key].total_samples = 2

    with pytest.raises(ValidationError, match="identifiers must be unique"):
        make_report(pipeline_results)


def test_report_rejects_different_questions_between_systems(pipeline_results):
    pipeline_results["baseline_records"][0]["question"] = "A different question"

    with pytest.raises(ValidationError, match="same questions"):
        make_report(pipeline_results)


def test_report_rejects_aggregate_that_disagrees_with_records(pipeline_results):
    pipeline_results["graphrag"].exact_match = 0.0

    with pytest.raises(ValidationError, match="Aggregate exact_match"):
        make_report(pipeline_results)


def test_report_rejects_naive_timestamp(pipeline_results):
    with pytest.raises(ValidationError, match="include a timezone"):
        make_report(pipeline_results, generated_at=datetime(2026, 10, 30, 12))


def test_empty_report_does_not_claim_a_success_rate(pipeline_results):
    for key in ("graphrag_records", "baseline_records"):
        pipeline_results[key] = []
    for key in ("graphrag", "plain_rag"):
        pipeline_results[key] = EvaluationMetrics(
            total_samples=0, multi_hop_recall=0.0, mrr_at_5=0.0,
            exact_match=0.0, token_f1=0.0, faithfulness=0.0,
        )

    report = make_report(pipeline_results)

    assert report.sample_count == 0
    assert "No questions were evaluated" in render_console(report)


def test_console_labels_synthetic_scope_and_removes_terminal_controls(pipeline_results):
    for key in ("graphrag_records", "baseline_records"):
        pipeline_results[key][0]["predicted_answer"] = "Chicago\n\x1b[31m"
    output = render_console(make_report(pipeline_results))

    assert "Synthetic smoke test" in output
    assert "1 question" in output
    assert "do not measure MuSiQue validation performance" in output
    assert "Token-overlap grounding" in output
    assert "\x1b" not in output
    assert "100.00%" in output


@pytest.mark.parametrize("limit", ["0", "-1", "many", "1.5"])
def test_invalid_cli_limit_fails_before_touching_storage(tmp_path, limit):
    store = tmp_path / "keep"
    store.mkdir()
    sentinel = store / "untouched.txt"
    sentinel.write_text("existing data", encoding="utf-8")

    with pytest.raises(SystemExit) as failure:
        main(["--max-samples", limit, "--storage-dir", str(store)])

    assert failure.value.code == 2
    assert sentinel.read_text(encoding="utf-8") == "existing data"


def test_missing_explicit_dataset_never_falls_back_to_synthetic(tmp_path, capsys):
    store = tmp_path / "not-created"

    with pytest.raises(SystemExit) as failure:
        main(["--data-file", str(tmp_path / "missing.jsonl"), "--storage-dir", str(store)])

    assert failure.value.code == 2
    assert not store.exists()
    assert "--data-file" in capsys.readouterr().err


def test_cli_rejects_dataset_inside_reset_storage(tmp_path):
    source = tmp_path / "sample.jsonl"
    source.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as failure:
        main(["--data-file", str(source), "--storage-dir", str(tmp_path)])

    assert failure.value.code == 2
    assert source.read_text(encoding="utf-8") == "{}"


def test_cli_rejects_colliding_output_files(tmp_path):
    output = str(tmp_path / "same-file")

    with pytest.raises(SystemExit) as failure:
        parse_args(["--report-file", output, "--results-file", output])

    assert failure.value.code == 2
