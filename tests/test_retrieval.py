"""Tests for query extraction, seed linking, and Reciprocal Rank Fusion."""

import pytest
from src.embeddings import HashProjectionEmbedding
from src.retrieval import HybridRetriever, QueryEntityExtractor
from src.storage import KuzuGraphStore, LanceVectorStore


def test_query_entity_extractor():
    extractor = QueryEntityExtractor()
    query = "Which city was the founder of 'Company X' born in?"
    mentions = extractor.extract_mentions(query)
    assert "Company X" in mentions

    query2 = "Where did Andrei Tarkovsky win his film award?"
    mentions2 = extractor.extract_mentions(query2)
    assert "Andrei Tarkovsky" in mentions2


def test_reciprocal_rank_fusion(tmp_path):
    k_store = KuzuGraphStore(db_path=str(tmp_path / "kuzu"))
    v_store = LanceVectorStore(uri=str(tmp_path / "lance"))
    embedder = HashProjectionEmbedding(dimension=32)
    retriever = HybridRetriever(k_store, v_store, embedder, rrf_k=60)

    # Chunk c1 is rank 1 in graph, rank 2 in vector
    # Chunk c2 is rank 2 in graph, not in vector
    # Chunk c3 is rank 1 in vector, not in graph
    graph_chunks = ["c1", "c2"]
    vector_chunks = ["c3", "c1"]

    top_chunks, scores = retriever.compute_rrf(graph_chunks, vector_chunks, top_k=3)

    # c1 score: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639 + 0.01613 = 0.03252
    # c3 score: 1/(60+1) = 0.01639
    # c2 score: 1/(60+2) = 0.01613
    assert top_chunks[0] == "c1"
    assert scores["c1"] > scores["c3"]
    assert scores["c3"] > scores["c2"]
