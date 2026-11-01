"""End-to-end integration test verifying Multi-Hop GraphRAG vs Plain RAG baseline."""

import pytest
from src.dataset import MuSiQueDatasetParser
from src.pipeline import MultiHopGraphRAGSystem


def test_end_to_end_pipeline(tmp_path):
    storage_dir = str(tmp_path / "system_data")
    system = MultiHopGraphRAGSystem(storage_dir=storage_dir, clean_start=True)

    # 1. Load mock multi-hop benchmark
    docs, samples = MuSiQueDatasetParser.create_mock_musique_benchmark()
    assert len(docs) > 0
    assert len(samples) > 0

    # 2. Ingest documents into chunks
    chunks = system.ingest_documents(docs)
    assert len(chunks) >= len(docs)

    # 3. Extract entities & relations and populate dual stores
    system.extract_and_index_knowledge(chunks)

    # 4. Run benchmark
    results = system.run_benchmark(samples)
    graphrag_m = results["graphrag"]
    baseline_m = results["plain_rag"]

    assert graphrag_m.total_samples == len(samples)
    assert baseline_m.total_samples == len(samples)

    # Multi-hop retrieval recall for GraphRAG should be >= baseline
    assert graphrag_m.multi_hop_recall >= baseline_m.multi_hop_recall
