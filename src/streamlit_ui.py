"""Native Streamlit views for the CGC Labs GraphRAG demonstration."""

from __future__ import annotations

from base64 import b64encode
from importlib.resources import files
from typing import TYPE_CHECKING

import streamlit as st

from src.report_html import render_html_report
from src.reporting import BenchmarkReport

if TYPE_CHECKING:
    from src.demo import QueryComparison, QueryResult


def render_intro() -> None:
    """Render static brand artwork; no user or model text enters this HTML."""
    logo = b64encode(files("src").joinpath("assets/cgc-labs-logo.png").read_bytes()).decode("ascii")
    route = "data:image/svg+xml;base64," + b64encode(files("src").joinpath("assets/demo-route.svg").read_bytes()).decode("ascii")
    html = ("""
    <style>
      [data-testid="stMainBlockContainer"]{max-width:1280px;padding-top:1.25rem;padding-bottom:3rem}
      .cgc-nav{display:flex;justify-content:space-between;align-items:center;gap:24px;padding:0 0 19px;border-bottom:1px solid #d3ddd4}
      .cgc-nav img{width:108px;height:auto;display:block}.cgc-nav a{color:#142b24;text-decoration:none;min-height:44px;display:inline-flex;align-items:center}
      .cgc-links{display:flex;gap:26px;font-size:13px}.cgc-intro{display:grid;grid-template-columns:1.25fr 1fr;align-items:center;gap:44px;padding:32px 0 24px}
      .cgc-eyebrow{font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;margin:0 0 18px}.cgc-eyebrow:before{content:'';width:7px;height:7px;display:inline-block;margin-right:10px;background:#142b24}
      .cgc-title{font-family:'Cabinet Grotesk',Arial,sans-serif!important;font-size:clamp(48px,5.8vw,80px)!important;font-weight:800!important;line-height:.98!important;letter-spacing:-.055em!important;color:#142b24;margin:0!important;padding:0!important}
      .cgc-title span{position:relative;display:inline-block;z-index:0}.cgc-title span:after{content:'';position:absolute;left:0;right:0;bottom:.04em;height:.18em;background:#d5f36b;z-index:-1}
      .cgc-lede{font-size:17px;line-height:1.55;color:#526259;max-width:510px;margin:20px 0 0}.cgc-board{background:#eaf0e9;border:1px solid #d3ddd4;border-radius:4px;padding:20px}
      .cgc-board-label{display:flex;justify-content:space-between;gap:16px;font-size:10px;color:#526259;padding-bottom:12px;border-bottom:1px solid #c1d0c4}.cgc-route{display:block;width:100%;height:auto;margin-top:10px}
      .cgc-scope{font-size:12px;color:#526259;line-height:1.6;padding:0 0 22px;border-bottom:1px solid #d3ddd4;margin-bottom:14px}.cgc-scope strong{color:#142b24;margin-right:12px}
      .cgc-nav a:focus-visible{outline:3px solid #637d18;outline-offset:5px}
      @media(max-width:700px){.cgc-intro{grid-template-columns:1fr;gap:24px;padding-top:26px}.cgc-board{display:none}.cgc-title{font-size:clamp(46px,12vw,68px)!important}.cgc-links{gap:18px;font-size:12px}.cgc-nav img{width:94px}.cgc-lede{font-size:16px}.cgc-scope strong{display:block;margin:0 0 5px}[data-testid="stMainBlockContainer"]{padding-left:20px;padding-right:20px}}
    </style>
    """
    + '<nav class="cgc-nav" aria-label="Project links"><a href="https://cgclabs.org/" target="_blank" rel="noopener noreferrer" aria-label="CGC Labs home"><img src="data:image/png;base64,'
    + logo
    + '" width="314" height="160" alt="CGC Labs"></a><div class="cgc-links"><a href="https://www.christiangcallahan.tech/" target="_blank" rel="noopener noreferrer">Portfolio ↗</a><a href="https://cgclabs.org/" target="_blank" rel="noopener noreferrer">CGC Labs ↗</a></div></nav>'
    + """
    <div class="cgc-intro"><div><p class="cgc-eyebrow">Independent engineering / interactive demo</p><h1 class="cgc-title">Multi-hop<br>Graph<span>RAG.</span></h1><p class="cgc-lede">Ask a question. Compare graph and vector retrieval. Inspect the passages behind each answer.</p></div>
    <figure class="cgc-board"><div class="cgc-board-label"><span>Two retrieval paths</span><span>RAG / 01</span></div><img class="cgc-route" src="CGC_ROUTE_DATA" width="460" height="210" alt="Dense passages and graph paths feed rank fusion, which selects the answer context"></figure></div>
    <p class="cgc-scope"><strong>11 synthetic documents / 3 example questions</strong>Rule-based extraction and deterministic embeddings. The examples demonstrate the pipeline, rather than general question-answering accuracy.</p>
    """)
    st.html(html.replace("CGC_ROUTE_DATA",route))


def render_query_result(comparison: QueryComparison) -> None:
    st.subheader("Retrieval comparison", anchor=False)
    st.caption(f"Results for: {comparison.question}")
    if comparison.expected_answer is not None:
        st.caption(f"Expected answer: {comparison.expected_answer}")
    graph_column, dense_column = st.columns(2, gap="large")
    for column, label, result in (
        (graph_column, "Graph + vector", comparison.graphrag),
        (dense_column, "Dense only", comparison.plain_rag),
    ):
        with column:
            st.markdown(f"**{label}**")
            st.subheader(result.synthesis.final_answer or "No answer", anchor=False)
            st.caption(f"{len(result.retrieval.top_chunks)} retrieved passages")
            with st.expander(f"Inspect {label.lower()} passages"):
                render_passages(result)

    with st.expander("Inspect graph traversal paths"):
        paths = comparison.graphrag.retrieval.subgraph_paths
        if paths:
            st.dataframe([
                {"Seed": path.seed, "First relation": path.predicate_1,
                 "Bridge": path.bridge, "Terminal relation": path.predicate_2,
                 "Target": path.target, "Hops": path.hop_count}
                for path in paths
            ], hide_index=True, width="stretch")
        else:
            st.caption("No graph paths were retrieved for this question.")
        st.caption("The graph path representation and heuristic extractor have limited relation coverage.")

    st.download_button("Download this comparison", comparison.model_dump_json(indent=2),
                       file_name="graphrag-query.json", mime="application/json", on_click="ignore")


def render_passages(result: QueryResult) -> None:
    for rank, chunk in enumerate(result.retrieval.top_chunks, start=1):
        st.markdown(f"**{rank:02d} / {chunk.doc_title or 'Passage'}**")
        st.write(chunk.text)
        st.caption(f"Chunk ID: {chunk.chunk_id}")
        st.divider()
    if not result.retrieval.top_chunks:
        st.caption("No passages were retrieved.")


def render_benchmark(report: BenchmarkReport) -> None:
    st.caption(f"Completed run: {report.sample_count} synthetic questions / {report.document_count} documents")
    for column, label, value in zip(st.columns(3),
        ("GraphRAG exact match", "All-support recall", "Answer token F1"),
        (report.graphrag.exact_match, report.graphrag.multi_hop_recall, report.graphrag.token_f1)):
        column.metric(label, f"{value:.0%}")
    st.dataframe([
        {"Metric":label,"GraphRAG":f"{getattr(report.graphrag, field):.1%}",
         "Dense only":f"{getattr(report.plain_rag, field):.1%}"}
        for label,field in (("Answer exact match","exact_match"),("Answer token F1","token_f1"),
                            ("All-support recall","multi_hop_recall"),("Token grounding proxy","faithfulness"))
    ], hide_index=True, width="stretch")
    st.caption("All-support recall requires every expected supporting chunk. Grounding is a lexical proxy, not independent factual verification.")
    for record in report.graphrag_records:
        peer = next(row for row in report.plain_rag_records if row.sample_id == record.sample_id)
        with st.expander(record.question):
            st.write({"Expected":record.gold_answer,"GraphRAG":record.predicted_answer,"Dense only":peer.predicted_answer})
    left,right = st.columns(2)
    left.download_button("Download benchmark JSON",report.model_dump_json(indent=2),
                         file_name="graphrag-benchmark.json",mime="application/json",on_click="ignore")
    right.download_button("Download HTML report",render_html_report(report),
                          file_name="graphrag-report.html",mime="text/html",on_click="ignore")


def render_architecture() -> None:
    st.subheader("Pipeline and storage", anchor=False)
    st.caption("Separate graph and vector indexes share entity names and deterministic chunk IDs.")
    st.image(files("src").joinpath("assets/architecture.svg").read_text(encoding="utf-8"), width="stretch")
    st.image(files("src").joinpath("assets/storage-schema.svg").read_text(encoding="utf-8"), width="stretch")
    st.caption("Default configuration: 384-dimensional hash projections, graph depth up to 3 hops, relation confidence at least 0.70, and 5 fused passages.")
