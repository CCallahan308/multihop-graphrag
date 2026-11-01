"""Plain RAG Baseline Retriever for comparative evaluation."""

from __future__ import annotations

from typing import List, Optional
from src.embeddings import BaseEmbeddingProvider
from src.models import Chunk, RetrievalResult
from src.storage import KuzuGraphStore, LanceVectorStore


class PlainRAGRetriever:
    """Standard dense vector retrieval baseline without graph traversal or entity indexing."""

    def __init__(
        self,
        vector_store: LanceVectorStore,
        graph_store: KuzuGraphStore,
        embedding_provider: BaseEmbeddingProvider,
        top_k: int = 5,
    ) -> None:
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedding_provider = embedding_provider
        self.top_k = top_k

    def retrieve(self, query: str) -> RetrievalResult:
        """Pure dense vector retrieval returning top-5 chunks."""
        query_vector = self.embedding_provider.embed_query(query)
        vector_results = self.vector_store.search_chunks(query_vector, top_k=self.top_k)
        vector_chunk_ids = [cid for cid, _ in vector_results]

        top_chunks: List[Chunk] = []
        for cid in vector_chunk_ids:
            chunk = self.graph_store.get_chunk_by_id(cid)
            if chunk:
                top_chunks.append(chunk)

        return RetrievalResult(
            query=query,
            top_chunks=top_chunks,
            subgraph_paths=[],
            graph_chunk_ids=[],
            vector_chunk_ids=vector_chunk_ids,
            fused_scores={cid: score for cid, score in vector_results},
        )

    @staticmethod
    def linearize_context(result: RetrievalResult) -> str:
        """Format purely retrieved passages without graph triples."""
        parts: List[str] = ["### Retrieved Passages:"]
        for chunk in result.top_chunks:
            title = f" [{chunk.doc_title}]" if chunk.doc_title else ""
            parts.append(f"[Chunk ID: {chunk.chunk_id}]{title}\n{chunk.text}\n")
        return "\n".join(parts)
