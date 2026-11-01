"""Hybrid Graph-Plus-Vector Retrieval Engine and Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from src.embeddings import BaseEmbeddingProvider
from src.models import Chunk, RetrievalResult, TraversalPath
from src.storage import KuzuGraphStore, LanceVectorStore

logger = logging.getLogger(__name__)


class QueryEntityExtractor:
    """Extracts entity mention spans from natural language multi-hop questions."""

    def __init__(self) -> None:
        # Match capitalized noun sequences, unicode letters, quoted substrings, or phrases with particles
        self._capitalized_pattern = re.compile(
            r"\b[A-Z\u00C0-\u024F][\w\u00C0-\u024F]*(?:\s+(?:du|de|la|le|of|the|and|in|at|van|von|da|di)\s+[A-Z\u00C0-\u024F][\w\u00C0-\u024F]*|\s+[A-Z\u00C0-\u024F][\w\u00C0-\u024F]*)*\b"
        )
        self._stop_mentions = {
            "Which", "What", "Who", "Where", "When", "Why", "How",
            "The", "A", "An", "In", "On", "At", "By", "For", "With",
            "Name", "Find", "Tell", "City", "Country", "Person", "Year",
        }

    def extract_mentions(self, query: str) -> List[str]:
        """Extract candidate seed entity mentions from question."""
        # 1. Quoted terms first
        quoted = re.findall(r"\"([^\"]+)\"|'([^']+)'", query)
        mentions = [q[0] or q[1] for q in quoted if (q[0] or q[1])]

        # 2. Capitalized token sequences
        matches = self._capitalized_pattern.findall(query)
        for m in matches:
            clean = m.strip()
            if clean not in self._stop_mentions and clean not in mentions and len(clean) > 1:
                mentions.append(clean)

        # 3. Fallback: if no mentions found, use significant non-stop query words
        if not mentions:
            words = [w for w in re.findall(r"\w+", query) if len(w) > 3 and w.capitalize() not in self._stop_mentions]
            if words:
                mentions.append(" ".join(words[:3]))

        return mentions


class HybridRetriever:
    """Hybrid Graph-Plus-Vector Retrieval engine with Reciprocal Rank Fusion."""

    def __init__(
        self,
        graph_store: KuzuGraphStore,
        vector_store: LanceVectorStore,
        embedding_provider: BaseEmbeddingProvider,
        rrf_k: int = 60,
    ) -> None:
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.rrf_k = rrf_k
        self.query_extractor = QueryEntityExtractor()

    def extract_and_link_seeds(self, query: str, top_k_seeds: int = 3) -> List[str]:
        """Step 1 & 2: Extract query mentions and link to seed entities via hybrid lexical and dense matching."""
        mentions = self.query_extractor.extract_mentions(query)
        if not mentions:
            return []

        seed_candidates: Dict[str, float] = {}
        for mention in mentions:
            # 1. Exact or case-insensitive lexical matching in graph store
            try:
                res_lex = self.graph_store.conn.execute(
                    "MATCH (e:Entity) WHERE lower(e.name) = lower($m) RETURN e.name LIMIT 1;",
                    {"m": mention},
                )
                if res_lex.has_next():
                    matched_name = str(res_lex.get_next()[0])
                    seed_candidates[matched_name] = max(seed_candidates.get(matched_name, 0.0), 2.0)
            except Exception:
                pass

            # 2. Dense semantic similarity matching via LanceDB
            mention_vec = self.embedding_provider.embed_query(mention)
            matched = self.vector_store.search_entities(mention_vec, top_k=top_k_seeds)
            for name, score in matched:
                if name.lower() == mention.lower():
                    score = 2.0
                elif mention.lower() in name.lower() or name.lower() in mention.lower():
                    score = max(score, 1.5)
                if name not in seed_candidates or score > seed_candidates[name]:
                    seed_candidates[name] = score

        # Sort candidate seeds by match score descending
        sorted_seeds = sorted(seed_candidates.items(), key=lambda x: x[1], reverse=True)
        # Prioritize strong lexical/substring matches over distant vector approximations
        strong_matches = [name for name, score in sorted_seeds if score >= 1.5]
        if strong_matches:
            return strong_matches[:top_k_seeds]
        return [name for name, _ in sorted_seeds[:top_k_seeds]]

    def compute_rrf(
        self,
        graph_chunk_ids: Sequence[str],
        vector_chunk_ids: Sequence[str],
        top_k: int = 5,
    ) -> Tuple[List[str], Dict[str, float]]:
        """Step 5: Compute Reciprocal Rank Fusion scores across graph and vector rank lists."""
        rrf_scores: Dict[str, float] = {}

        # 1-based ranks for graph retrieval list
        for rank, cid in enumerate(graph_chunk_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # 1-based ranks for vector retrieval list
        for rank, cid in enumerate(vector_chunk_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort descending by fused score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_cids = [cid for cid, _ in sorted_chunks[:top_k]]
        return top_cids, rrf_scores

    def retrieve(self, query: str) -> RetrievalResult:
        """Execute full 6-step hybrid retrieval pipeline."""
        # 1. Query Entity Extraction & Seed Linking (Top-3 seed entities)
        seed_names = self.extract_and_link_seeds(query, top_k_seeds=3)

        # 2. Graph Traversal (2-hop paths and candidate chunks Rg)
        subgraph_paths, graph_chunks_set = self.graph_store.traverse_two_hop(
            seed_names,
            min_weight=0.70,
            limit=20,
        )
        graph_chunk_ids = list(graph_chunks_set)

        # 3. Dense Semantic Retrieval (Top-10 chunks Rv)
        query_vector = self.embedding_provider.embed_query(query)
        vector_results = self.vector_store.search_chunks(query_vector, top_k=10)
        vector_chunk_ids = [cid for cid, _ in vector_results]

        # 4. Reciprocal Rank Fusion (RRF) -> Top-5 chunks
        top_cids, fused_scores = self.compute_rrf(
            graph_chunk_ids=graph_chunk_ids,
            vector_chunk_ids=vector_chunk_ids,
            top_k=5,
        )

        # Fetch Chunk objects
        top_chunks: List[Chunk] = []
        for cid in top_cids:
            chunk = self.graph_store.get_chunk_by_id(cid)
            if chunk:
                top_chunks.append(chunk)

        return RetrievalResult(
            query=query,
            top_chunks=top_chunks,
            subgraph_paths=subgraph_paths,
            graph_chunk_ids=graph_chunk_ids,
            vector_chunk_ids=vector_chunk_ids,
            fused_scores=fused_scores,
        )

    @staticmethod
    def linearize_context(result: RetrievalResult) -> str:
        """Step 6: Context Linearization into structured triples and cited passages."""
        parts: List[str] = []

        # 1. Linearized graph triples
        if result.subgraph_paths:
            parts.append("### Structured Relational Paths:")
            for p in result.subgraph_paths:
                triple_1 = f"({p.seed}) -[:{p.predicate_1}]-> ({p.bridge})"
                triple_2 = f"({p.bridge}) -[:{p.predicate_2}]-> ({p.target})"
                parts.append(f"- {triple_1} AND {triple_2}")
            parts.append("")

        # 2. Retrieved text chunks
        parts.append("### Retrieved Passages:")
        for chunk in result.top_chunks:
            title = f" [{chunk.doc_title}]" if chunk.doc_title else ""
            parts.append(f"[Chunk ID: {chunk.chunk_id}]{title}\n{chunk.text}\n")

        return "\n".join(parts)
