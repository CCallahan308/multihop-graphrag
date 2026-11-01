"""Report rendering preserves evidence and treats source text as untrusted."""

from datetime import datetime, timezone
from base64 import b64decode
from html.parser import HTMLParser
from importlib.resources import files
import json

import pytest

from src.report_html import render_html_report
from src.reporting import BenchmarkReport, build_report


def make_report(question="Where was the founder born?", *, empty=False):
    score = 0.0 if empty else 1.0
    metrics = {
        "total_samples": 0 if empty else 1,
        "multi_hop_recall": score, "mrr_at_5": score,
        "exact_match": score, "token_f1": score, "faithfulness": score,
    }
    record = {
        "sample_id": "sample-1", "question": question,
        "gold_answer": "Chicago", "predicted_answer": "Chicago",
        "gold_chunks": ["chunk-1"], "retrieved_chunks": ["chunk-1"],
        **{key: value for key, value in metrics.items() if key != "total_samples"},
    }
    records = [] if empty else [record]
    return build_report(
        {"graphrag": metrics, "plain_rag": metrics,
         "graphrag_records": records, "baseline_records": records},
        dataset_kind="synthetic", dataset_label='Built-in <examples> & "evidence"',
        document_count=0 if empty else 2, embedding_provider="Hash projection",
        generation_mode="Heuristic span extraction",
        generated_at=datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc),
    )


class EmbeddedDataParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_data = False
        self.data_parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("id") == "run-data":
            self.in_data = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_data = False

    def handle_data(self, data):
        if self.in_data:
            self.data_parts.append(data)


def test_report_preserves_actual_results_and_scope():
    report = make_report()
    rendered = render_html_report(report)
    assert "Synthetic smoke test" in rendered
    assert "MuSiQue validation set" in rendered
    assert "1 query / 2 documents" in rendered
    assert "100.0%" in rendered
    assert "Chicago" in rendered
    assert 'scope="col"' in rendered
    assert 'class="skip-link"' in rendered
    assert '<details class="query" open>' in rendered
    assert "Built-in &lt;examples&gt; &amp; &quot;evidence&quot;" in rendered


def test_untrusted_text_cannot_escape_markup_or_json_script():
    attack = '</script><img src=x onerror="alert(1)">'
    report = make_report(attack)
    rendered = render_html_report(report)
    assert attack not in rendered
    assert "&lt;/script&gt;&lt;img" in rendered
    parser = EmbeddedDataParser()
    parser.feed(rendered)
    exported = json.loads("".join(parser.data_parts))
    assert exported["graphrag_records"][0]["question"] == attack
    assert BenchmarkReport.model_validate(exported) == report


def test_empty_run_is_explicit_and_has_no_invented_queries():
    rendered = render_html_report(make_report(empty=True))
    assert "No queries evaluated in this run." in rendered
    assert "0 queries / 0 documents" in rendered
    assert '<details class="query"' not in rendered
    assert "0.0%" in rendered


def test_renderer_requires_a_validated_report():
    with pytest.raises(TypeError, match="validated BenchmarkReport"):
        render_html_report({})


def test_original_brand_artwork_is_embedded_unchanged():
    class LogoParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.logos = []

        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            if tag == "img" and attributes.get("alt") == "CGC Labs":
                self.logos.append(attributes)

    parser = LogoParser()
    parser.feed(render_html_report(make_report()))
    expected = files("src").joinpath("assets/cgc-labs-logo.png").read_bytes()
    assert len(parser.logos) == 2
    for logo in parser.logos:
        assert logo["src"].startswith("data:image/png;base64,")
        assert b64decode(logo["src"].split(",", 1)[1], validate=True) == expected
        assert (logo["width"], logo["height"]) == ("314", "160")
