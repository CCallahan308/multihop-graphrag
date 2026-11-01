"""End-to-end pipeline runner for Multi-Hop GraphRAG and Plain RAG Baseline."""

from __future__ import annotations

import logging
import os
import shutil
from types import TracebackType
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.baseline import PlainRAGRetriever
from src.chunker import DeterministicChunker
from src.embeddings import BaseEmbeddingProvider, HashProjectionEmbedding
from src.evaluation import EvaluationHarness
from src.extraction import EntityCanonicalizer, RuleBasedExtractor
from src.models import (
    Chunk,
    Document,
    Entity,
    EvaluationMetrics,
    EvaluationSample,
    ExtractionResult,
    Relationship,
)
from src.retrieval import HybridRetriever
from src.storage import KuzuGraphStore, LanceVectorStore
from src.synthesis import SynthesisGenerator

logger = logging.getLogger(__name__)


class MultiHopGraphRAGSystem:
    """Integrated Multi-Hop GraphRAG and Plain RAG pipeline."""

    def __init__(
        self,
        storage_dir: str = "./data_store",
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        clean_start: bool = True,
        *,
        buffer_pool_size: int = 0,
        max_num_threads: int = 0,
    ) -> None:
        for name, value in (("buffer_pool_size", buffer_pool_size), ("max_num_threads", max_num_threads)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer; zero uses the Kuzu default")
        self.storage_dir = Path(storage_dir).resolve()
        self.kuzu_dir = str(self.storage_dir / "kuzu")
        self.lance_dir = str(self.storage_dir / "lance")

        if clean_start and self.storage_dir.exists():
            shutil.rmtree(self.storage_dir, ignore_errors=True)

        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.lance_dir, exist_ok=True)

        self.embedding_provider = embedding_provider or HashProjectionEmbedding(dimension=384)
        self.chunker = DeterministicChunker(min_tokens=256, max_tokens=512, overlap_tokens=64)
        self.extractor = RuleBasedExtractor()
        self.canonicalizer = EntityCanonicalizer(
            embedding_provider=self.embedding_provider,
            similarity_threshold=0.88,
        )

        self._closed = False
        try:
            self.graph_store = KuzuGraphStore(
                db_path=self.kuzu_dir, read_only=False,
                buffer_pool_size=buffer_pool_size, max_num_threads=max_num_threads,
            )
            self.vector_store = LanceVectorStore(uri=self.lance_dir)

            self.hybrid_retriever = HybridRetriever(
                graph_store=self.graph_store,
                vector_store=self.vector_store,
                embedding_provider=self.embedding_provider,
                rrf_k=60,
            )

            self.baseline_retriever = PlainRAGRetriever(
                vector_store=self.vector_store,
                graph_store=self.graph_store,
                embedding_provider=self.embedding_provider,
                top_k=5,
            )

            self.generator = SynthesisGenerator()
            self.evaluation_harness = EvaluationHarness()

            self._indexed_chunks: List[Chunk] = []
            self._indexed_entities: List[Entity] = []
        except BaseException:
            try:
                self.close()
            except Exception:
                logger.exception("Store cleanup failed after pipeline initialization failed")
            raise

    def close(self) -> None:
        """Close both stores, including when construction completed only one store."""
        if self._closed:
            return
        try:
            if hasattr(self, "vector_store"):
                self.vector_store.close()
        finally:
            if hasattr(self, "graph_store"):
                self.graph_store.close()
        self._closed = True

    def __enter__(self) -> "MultiHopGraphRAGSystem":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except Exception:
                logger.exception("Store cleanup failed while handling a pipeline error")

    def ingest_documents(self, documents: Sequence[Document]) -> List[Chunk]:
        """Component 1: Ingest documents and generate deterministic chunks."""
        chunks = self.chunker.chunk_documents(documents)
        self._indexed_chunks.extend(chunks)

        # Batch insert chunks into property graph
        self.graph_store.add_chunks(chunks)

        # Generate chunk vector embeddings and insert into LanceDB
        if chunks:
            chunk_texts = [c.text for c in chunks]
            chunk_vectors = self.embedding_provider.embed_texts(chunk_texts)
            self.vector_store.upsert_chunks(chunks, chunk_vectors)

        return chunks

    def extract_and_index_knowledge(self, chunks: Sequence[Chunk]) -> None:
        """Components 2 & 3: Extract, canonicalize, and index entities and relations."""
        raw_entities: List[Entity] = []
        raw_relationships: List[Tuple[Relationship, str]] = []  # (rel, chunk_id)
        entity_chunk_links: List[Tuple[str, str]] = []  # (entity_name, chunk_id)

        for chunk in chunks:
            try:
                extraction = self.extractor.extract(chunk)
                raw_entities.extend(extraction.entities)
                for ent in extraction.entities:
                    entity_chunk_links.append((ent.name, chunk.chunk_id))
                for rel in extraction.relationships:
                    raw_relationships.append((rel, chunk.chunk_id))
            except Exception as e:
                # Engineering controls: Fall back to pure vector indexing on extraction failure
                logger.warning(f"Extraction failed for chunk {chunk.chunk_id}: {e}")

        # Canonicalize extracted entities
        canonical_entities = self.canonicalizer.canonicalize(raw_entities)
        self._indexed_entities.extend(canonical_entities)

        # Index entities in property graph
        self.graph_store.add_entities(canonical_entities)

        # Map variant names to canonical names
        name_map: Dict[str, str] = {}
        for ent in canonical_entities:
            name_map[ent.name.lower()] = ent.name
            for alias in ent.aliases:
                name_map[alias.lower()] = ent.name

        # Index relationships in property graph with canonicalized endpoints
        resolved_relationships: List[Relationship] = []
        for rel, cid in raw_relationships:
            src_canon = name_map.get(rel.source_entity.lower(), rel.source_entity)
            tgt_canon = name_map.get(rel.target_entity.lower(), rel.target_entity)
            if src_canon != tgt_canon:
                canon_rel = Relationship(
                    source_entity=src_canon,
                    target_entity=tgt_canon,
                    predicate=rel.predicate,
                    confidence=rel.confidence,
                    evidence_quote=rel.evidence_quote,
                )
                self.graph_store.add_relationships([canon_rel], chunk_id=cid)
                resolved_relationships.append(canon_rel)

        # Link entities to chunks via MENTIONED_IN edges
        for ent_name, cid in entity_chunk_links:
            canon_name = name_map.get(ent_name.lower(), ent_name)
            self.graph_store.link_entity_to_chunk(canon_name, cid)

        # Generate entity vector embeddings and insert into LanceDB
        if canonical_entities:
            ent_names = [e.name for e in canonical_entities]
            ent_vectors = self.embedding_provider.embed_texts(ent_names)
            self.vector_store.upsert_entities(canonical_entities, ent_vectors)

    def query_graphrag(self, question: str):
        """Run hybrid retrieval and structured answer synthesis."""
        retrieval_res = self.hybrid_retriever.retrieve(question)
        context = self.hybrid_retriever.linearize_context(retrieval_res)
        synthesis_res = self.generator.generate(question, context, retrieval_result=retrieval_res)
        return retrieval_res, synthesis_res

    def query_plain_rag(self, question: str):
        """Run baseline dense vector retrieval and answer synthesis."""
        retrieval_res = self.baseline_retriever.retrieve(question)
        context = self.baseline_retriever.linearize_context(retrieval_res)
        synthesis_res = self.generator.generate(question, context, retrieval_result=retrieval_res)
        return retrieval_res, synthesis_res

    def run_benchmark(
        self,
        samples: Sequence[EvaluationSample],
        on_sample_evaluated: Optional[Callable[[Dict], None]] = None,
    ) -> Dict[str, Any]:
        """Run comparative evaluation between GraphRAG and Plain RAG baseline."""
        # 1. Evaluate GraphRAG
        graphrag_metrics, graphrag_records = self.evaluation_harness.evaluate_system(
            samples=samples,
            retrieval_fn=self.hybrid_retriever.retrieve,
            generation_fn=lambda q, ctx, res=None: self.generator.generate(q, ctx, retrieval_result=res),
            on_sample_evaluated=on_sample_evaluated,
        )

        # 2. Evaluate Plain RAG Baseline
        baseline_metrics, baseline_records = self.evaluation_harness.evaluate_system(
            samples=samples,
            retrieval_fn=self.baseline_retriever.retrieve,
            generation_fn=lambda q, ctx, res=None: self.generator.generate(q, ctx, retrieval_result=res),
        )

        return {
            "graphrag": graphrag_metrics,
            "plain_rag": baseline_metrics,
            "graphrag_records": graphrag_records,
            "baseline_records": baseline_records,
        }
