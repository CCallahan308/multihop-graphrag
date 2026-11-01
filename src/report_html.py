"""Render an offline benchmark report from the validated run envelope."""

from __future__ import annotations

from base64 import b64encode
from html import escape
from importlib.resources import files
from string import Template

from src.reporting import BenchmarkRecord, BenchmarkReport


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _metric_rows(report: BenchmarkReport) -> str:
    metrics = [
        ("Answer exact match", "exact_match", False),
        ("Answer token F1", "token_f1", False),
        ("All-support retrieval recall", "multi_hop_recall", False),
        ("Retrieval MRR@5", "mrr_at_5", True),
        ("Token grounding proxy", "faithfulness", False),
    ]
    rows = []
    for label, attribute, decimal in metrics:
        graph = getattr(report.graphrag, attribute)
        dense = getattr(report.plain_rag, attribute)
        graph_value = f"{graph:.3f}" if decimal else _percent(graph)
        dense_value = f"{dense:.3f}" if decimal else _percent(dense)
        delta = f"{graph - dense:+.3f}" if decimal else f"{(graph - dense) * 100:+.1f} pp"
        rows.append(
            f'<tr><th scope="row">{label}</th><td class="graph-value">{graph_value}</td>'
            f'<td>{dense_value}</td><td class="delta">{delta}</td></tr>'
        )
    return "\n".join(rows)


def _chunk_ids(chunk_ids: list[str]) -> str:
    if not chunk_ids:
        return '<span class="muted">None</span>'
    return " ".join(f"<code>{escape(chunk_id)}</code>" for chunk_id in chunk_ids)


def _query_detail(index: int, graph: BenchmarkRecord, dense: BenchmarkRecord) -> str:
    matched = graph.exact_match == 1.0
    state = "Exact match" if matched else "Answer differs"
    css_state = "matched" if matched else "different"
    expanded = " open" if index == 1 else ""
    return f"""
    <details class="query"{expanded}>
      <summary>
        <span class="query-number">{index:02d}</span>
        <span class="query-question">{escape(graph.question)}</span>
        <span class="query-status {css_state}">{state}</span>
        <span class="expand-icon" aria-hidden="true">+</span>
      </summary>
      <div class="query-body">
        <div class="answer-grid">
          <div><p class="label">GraphRAG answer</p><p class="answer">{escape(graph.predicted_answer) or 'No answer'}</p>
            <p class="answer-score">Token F1 <strong>{_percent(graph.token_f1)}</strong></p></div>
          <div><p class="label">Dense-only answer</p><p class="answer dense-answer">{escape(dense.predicted_answer) or 'No answer'}</p>
            <p class="answer-score">Token F1 <strong>{_percent(dense.token_f1)}</strong></p></div>
        </div>
        <p class="expected"><span class="label">Expected answer</span> {escape(graph.gold_answer)}</p>
        <dl class="evidence-list">
          <div><dt>Supporting chunk IDs</dt><dd>{_chunk_ids(graph.gold_chunks)}</dd></div>
          <div><dt>GraphRAG retrieved</dt><dd>{_chunk_ids(graph.retrieved_chunks)}</dd></div>
          <div><dt>Dense-only retrieved</dt><dd>{_chunk_ids(dense.retrieved_chunks)}</dd></div>
        </dl>
        <p class="sample-id">Sample <code>{escape(graph.sample_id)}</code></p>
      </div>
    </details>"""


def render_html_report(report: BenchmarkReport) -> str:
    """Return a standalone report with embedded artwork and escaped source text."""
    if not isinstance(report, BenchmarkReport):
        raise TypeError("report must be a validated BenchmarkReport")
    baseline = {record.sample_id: record for record in report.plain_rag_records}
    queries = "\n".join(
        _query_detail(index, record, baseline[record.sample_id])
        for index, record in enumerate(report.graphrag_records, start=1)
    )
    if not queries:
        queries = '<p class="empty-state">No queries evaluated in this run.</p>'

    if report.dataset_kind == "synthetic":
        scope_label = "Synthetic smoke test"
        scope_note = (
            "Built-in scenarios check the retrieval pipeline. "
            "These results do not establish performance on the MuSiQue validation set."
        )
    else:
        scope_label = "Dataset evaluation"
        scope_note = (
            "Results describe only the evaluated records and the providers listed below. "
            "They do not establish production performance."
        )

    # Escape '<' so user-controlled text cannot close the inert JSON script element.
    serialized = report.model_dump_json(indent=2).replace("<", "\\u003c").replace("&", "\\u0026")
    template = Template(files("src").joinpath("report_template.html").read_text(encoding="utf-8"))
    return template.substitute(
        brand_logo="data:image/png;base64," + b64encode(files("src").joinpath("assets/cgc-labs-logo.png").read_bytes()).decode("ascii"),
        dataset_label=escape(report.dataset_label),
        scope_label=scope_label,
        scope_note=scope_note,
        sample_count=report.sample_count,
        query_label="query" if report.sample_count == 1 else "queries",
        document_count=report.document_count,
        document_label="document" if report.document_count == 1 else "documents",
        generated_at=escape(report.generated_at.strftime("%d %b %Y / %H:%M UTC")),
        embedding_provider=escape(report.embedding_provider),
        generation_mode=escape(report.generation_mode),
        exact_match=_percent(report.graphrag.exact_match),
        token_f1=_percent(report.graphrag.token_f1),
        recall=_percent(report.graphrag.multi_hop_recall),
        metric_rows=_metric_rows(report),
        query_details=queries,
        json_data=serialized,
    )
