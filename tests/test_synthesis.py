"""Tests for structured answer synthesis and heuristic span extraction."""

import pytest
from src.models import Chunk, RetrievalResult, TraversalPath
from src.synthesis import (
    HeuristicSpanExtractor,
    MockLLMProvider,
    SynthesisGenerator,
    clean_answer_span,
)


def test_clean_answer_span():
    assert clean_answer_span("The answer is: Chicago.") == "Chicago"
    assert clean_answer_span("  'Miquette Giraudy'  ") == "Miquette Giraudy"
    assert clean_answer_span("He was born in Chicago, Illinois.") == "Chicago, Illinois"
    assert clean_answer_span("Answer: Grand Prix Spécial du Jury") == "Grand Prix Spécial du Jury"
    assert clean_answer_span("**United Kingdom**.") == "United Kingdom"


@pytest.mark.parametrize(
    ("raw_answer", "expected"),
    [
        ('"**United Kingdom**."', "United Kingdom"),
        ("**Answer: `Chicago`.**", "Chicago"),
        ("\n  The answer is: **Chicago**.\nExtra explanation", "Chicago"),
        ("   ", ""),
    ],
)
def test_clean_answer_span_nested_formatting(raw_answer, expected):
    assert clean_answer_span(raw_answer) == expected


def test_heuristic_span_extractor_from_paths():
    extractor = HeuristicSpanExtractor()
    query = "Which city was the founder of Company X born in?"
    context = "Retrieved context passages."
    retrieval_res = RetrievalResult(
        query=query,
        top_chunks=[],
        subgraph_paths=[
            TraversalPath(
                seed="Company X",
                predicate_1="FOUNDED_BY",
                bridge="John Doe",
                predicate_2="BORN_IN",
                target="Chicago",
            )
        ],
    )

    output = extractor.extract_span(query, context, retrieval_res)
    assert output.final_answer == "Chicago"
    assert "John Doe" in output.reasoning_trace


@pytest.mark.parametrize("direct_first", [True, False])
def test_heuristic_prefers_terminal_target_over_direct_bridge(direct_first):
    query = "Which city was the founder of Company X born in?"
    direct_path = TraversalPath(
        seed="Company X", predicate_1="FOUNDED_BY", bridge="John Doe",
        predicate_2="DIRECT_NEIGHBOR", target="John Doe", hop_count=1,
    )
    terminal_path = TraversalPath(
        seed="Company X", predicate_1="FOUNDED_BY", bridge="John Doe",
        predicate_2="BORN_IN", target="Chicago", hop_count=2,
    )
    paths = [direct_path, terminal_path] if direct_first else [terminal_path, direct_path]
    result = RetrievalResult(query=query, top_chunks=[], subgraph_paths=paths)

    output = HeuristicSpanExtractor().extract_span(query, "", result)

    assert output.final_answer == "Chicago"
    assert "John Doe" in output.reasoning_trace
    assert result.subgraph_paths == paths


@pytest.mark.parametrize("direct_first", [True, False])
def test_heuristic_preserves_single_hop_answer_with_irrelevant_extension(direct_first):
    query = "Who founded Company X?"
    direct_path = TraversalPath(
        seed="Company X", predicate_1="FOUNDED_BY", bridge="John Doe",
        predicate_2="DIRECT_NEIGHBOR", target="John Doe", hop_count=1,
    )
    extended_path = TraversalPath(
        seed="Company X", predicate_1="FOUNDED_BY", bridge="John Doe",
        predicate_2="BORN_IN", target="Chicago", hop_count=2,
    )
    paths = [direct_path, extended_path] if direct_first else [extended_path, direct_path]
    result = RetrievalResult(query=query, top_chunks=[], subgraph_paths=paths)

    output = HeuristicSpanExtractor().extract_span(query, "", result)

    assert output.final_answer == "John Doe"


def test_heuristic_prefers_three_hop_terminal_target():
    query = "Which country is the university where the inventor of Dynamo studied located in?"
    paths = [
        TraversalPath(
            seed="Dynamo", predicate_1="INVENTED_BY", bridge="Michael Faraday",
            predicate_2="STUDIED_AT", target="Royal Institution", hop_count=2,
        ),
        TraversalPath(
            seed="Dynamo", predicate_1="INVENTED_BY", bridge="Michael Faraday",
            predicate_2="DIRECT_NEIGHBOR", target="Michael Faraday", hop_count=1,
        ),
        TraversalPath(
            seed="Dynamo", predicate_1="INVENTED_BY -> Royal Institution", bridge="Michael Faraday",
            predicate_2="LOCATED_IN", target="United Kingdom", hop_count=3,
        ),
    ]
    result = RetrievalResult(query=query, top_chunks=[], subgraph_paths=paths)

    output = HeuristicSpanExtractor().extract_span(query, "", result)

    assert output.final_answer == "United Kingdom"


def test_heuristic_checks_terminal_targets_before_bridge_fallback():
    query = "Which city was the founder of Company X born in?"
    result = RetrievalResult(
        query=query, top_chunks=[], subgraph_paths=[
            TraversalPath(
                seed="Company X", predicate_1="FOUNDED_BY", bridge="John Doe",
                predicate_2="FOUNDED", target="Company X",
            ),
            TraversalPath(
                seed="Company X", predicate_1="FOUNDED_BY", bridge="John Doe",
                predicate_2="BORN_IN", target="Chicago",
            ),
        ],
    )

    output = HeuristicSpanExtractor().extract_span(query, "", result)

    assert output.final_answer == "Chicago"


def test_synthesis_generator_with_mock_llm():
    mock_responses = {
        "Who founded": "Reasoning: Traced relational path.\nAnswer: John Doe\nCitations: chunk1",
    }
    generator = SynthesisGenerator(llm_callable=MockLLMProvider(mock_responses))
    output = generator.generate("Who founded Company X?", context="[Chunk ID: chunk1] John Doe founded Company X.")

    assert output.final_answer == "John Doe"
    assert "chunk1" in output.cited_chunk_ids


def test_heuristic_fallback_uses_passage_body_and_preserves_context_citations():
    query = "Who founded Company X?"
    chunk = Chunk(
        chunk_id="chunk_1", doc_title="Misleading Document Title",
        text="John Doe founded Company X.",
    )
    retrieval = RetrievalResult(query=query, top_chunks=[chunk])
    context = f"### Retrieved Passages:\n[Chunk ID: {chunk.chunk_id}] [{chunk.doc_title}]\n{chunk.text}"

    output = HeuristicSpanExtractor().extract_span(query, context, retrieval)

    assert output.final_answer == "John Doe"
    assert output.cited_chunk_ids == ["chunk_1"]


def test_heuristic_fallback_matches_benchmark_and_formatted_query_context():
    from src.baseline import PlainRAGRetriever

    query = "Which city was the founder of Company X born in?"
    chunks = [
        Chunk(chunk_id="chunk_1", doc_title="Company Profile", text="John Doe founded Company X."),
        Chunk(chunk_id="chunk_2", doc_title="Founder Biography", text="John Doe was born in Chicago."),
    ]
    retrieval = RetrievalResult(query=query, top_chunks=chunks)
    extractor = HeuristicSpanExtractor()
    benchmark_context = "\n".join(chunk.text for chunk in chunks)
    query_context = PlainRAGRetriever.linearize_context(retrieval)

    benchmark = extractor.extract_span(query, benchmark_context, retrieval)
    interactive = extractor.extract_span(query, query_context, retrieval)

    assert interactive.final_answer == benchmark.final_answer == "John Doe"
    assert interactive.cited_chunk_ids == ["chunk_1", "chunk_2"]


def test_heuristic_fallback_returns_unknown_for_empty_retrieval_metadata():
    query = "Who founded Company X?"
    retrieval = RetrievalResult(query=query, top_chunks=[])

    output = HeuristicSpanExtractor().extract_span(
        query, "### Retrieved Passages:\n[Unused Document Title]", retrieval,
    )

    assert output.final_answer == "Unknown"
    assert output.cited_chunk_ids == []


def test_structured_triples_remain_available_with_empty_passage_list():
    query = "Who founded Company X?"
    retrieval = RetrievalResult(query=query, top_chunks=[])
    context = "(Company X) -[:FOUNDED_BY]-> (John Doe)\n[Chunk ID: bridge_chunk]"

    output = HeuristicSpanExtractor().extract_span(query, context, retrieval)

    assert output.final_answer == "John Doe"
    assert output.cited_chunk_ids == ["bridge_chunk"]


def test_heuristic_context_fallback_without_retrieval_result_is_preserved():
    output = HeuristicSpanExtractor().extract_span(
        "Who founded Company X?", "John Doe founded Company X.",
    )

    assert output.final_answer == "John Doe"
