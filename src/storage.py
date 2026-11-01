"""Dual-index storage architecture integrating Kùzu property graph and LanceDB vector store."""

from __future__ import annotations

import os
import math
import logging
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import kuzu
import lancedb
import numpy as np

from src.models import Chunk, Entity, Relationship, TraversalPath


logger = logging.getLogger(__name__)


class KuzuGraphStore:
    """Property graph manager backed by embedded KùzuDB."""

    def __init__(
        self,
        db_path: str,
        read_only: bool = False,
        *,
        buffer_pool_size: int = 0,
        max_num_threads: int = 0,
    ) -> None:
        for name, value in (("buffer_pool_size", buffer_pool_size), ("max_num_threads", max_num_threads)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer; zero uses the Kuzu default")
        self.db_path = str(Path(db_path).resolve())
        os.makedirs(Path(self.db_path).parent, exist_ok=True)
        self.db = kuzu.Database(
            self.db_path, read_only=read_only,
            buffer_pool_size=buffer_pool_size, max_num_threads=max_num_threads,
        )
        try:
            self.conn = kuzu.Connection(self.db)
            if not read_only:
                self._init_schema()
        except BaseException:
            # Preserve the construction failure if releasing the database also fails.
            try:
                if hasattr(self, "conn"):
                    self.close()
                else:
                    self.db.close()
            except Exception:
                logger.exception("Kuzu cleanup failed after graph-store initialization failed")
            raise

    def close(self) -> None:
        """Release the connection before its database; both native APIs are idempotent."""
        try:
            self.conn.close()
        finally:
            self.db.close()

    def _init_schema(self) -> None:
        """Create Node and Rel tables according to system specification."""
        ddl_statements = [
            "CREATE NODE TABLE Entity(name STRING, type STRING, description STRING, PRIMARY KEY(name));",
            "CREATE NODE TABLE Chunk(chunk_id STRING, text STRING, doc_title STRING, PRIMARY KEY(chunk_id));",
            "CREATE REL TABLE RELATION(FROM Entity TO Entity, predicate STRING, weight DOUBLE, source_chunk_id STRING);",
            "CREATE REL TABLE MENTIONED_IN(FROM Entity TO Chunk);",
        ]
        for ddl in ddl_statements:
            try:
                self.conn.execute(ddl)
            except Exception:
                # Table already exists
                pass

    def add_entities(self, entities: Sequence[Entity]) -> None:
        """Insert unique entities into Kùzu using parameterized queries."""
        for ent in entities:
            query = "MERGE (e:Entity {name: $name}) ON CREATE SET e.type = $ent_type, e.description = $description_text"
            params = {
                "name": ent.name,
                "ent_type": ent.type,
                "description_text": ent.description,
            }
            try:
                self.conn.execute(query, params)
            except Exception:
                pass

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Insert document chunks into Kùzu using parameterized queries."""
        for chunk in chunks:
            query = "MERGE (c:Chunk {chunk_id: $cid}) ON CREATE SET c.text = $text, c.doc_title = $doc_title"
            params = {
                "cid": chunk.chunk_id,
                "text": chunk.text,
                "doc_title": chunk.doc_title,
            }
            try:
                self.conn.execute(query, params)
            except Exception:
                pass

    def add_relationships(self, relationships: Sequence[Relationship], chunk_id: str = "") -> None:
        """Insert typed relationships between entities using parameterized queries."""
        for rel in relationships:
            weight = float(rel.confidence)
            try:
                self.conn.execute(
                    "MERGE (s:Entity {name: $name}) ON CREATE SET s.type = 'Concept', s.description = ''",
                    {"name": rel.source_entity},
                )
                self.conn.execute(
                    "MERGE (t:Entity {name: $name}) ON CREATE SET t.type = 'Concept', t.description = ''",
                    {"name": rel.target_entity},
                )
                query = (
                    "MATCH (s:Entity {name: $src}), (t:Entity {name: $tgt}) "
                    "CREATE (s)-[:RELATION {predicate: $pred, weight: $weight, source_chunk_id: $cid}]->(t)"
                )
                params = {
                    "src": rel.source_entity,
                    "tgt": rel.target_entity,
                    "pred": rel.predicate,
                    "weight": weight,
                    "cid": chunk_id,
                }
                self.conn.execute(query, params)
            except Exception:
                pass

    def link_entity_to_chunk(self, entity_name: str, chunk_id: str) -> None:
        """Create MENTIONED_IN edge from Entity to Chunk using parameterized queries."""
        query = (
            "MATCH (e:Entity {name: $name}), (c:Chunk {chunk_id: $cid}) "
            "CREATE (e)-[:MENTIONED_IN]->(c)"
        )
        params = {"name": entity_name, "cid": chunk_id}
        try:
            self.conn.execute(query, params)
        except Exception:
            pass

    def traverse_paths(
        self,
        seed_names: Sequence[str],
        min_weight: float = 0.70,
        max_hops: int = 3,
        limit: int = 20,
    ) -> Tuple[List[TraversalPath], Set[str]]:
        """Traverse edges with inclusive confidence in [min_weight, 1].

        min_weight must be finite and between 0 and 1. Direct seed chunks
        remain eligible without traversing a relation.
        """
        if (
            isinstance(min_weight, bool)
            or not isinstance(min_weight, Real)
            or not math.isfinite(min_weight)
            or not 0.0 <= min_weight <= 1.0
        ):
            raise ValueError("min_weight must be a finite number between 0 and 1")
        if not seed_names:
            return [], set()

        paths: List[TraversalPath] = []
        retrieved_chunk_ids: Set[str] = set()
        seeds_list = list(seed_names)
        weight_val = float(min_weight)

        # 1. Direct seed chunks
        seed_chunk_query = f"""
        MATCH (seed:Entity)-[:MENTIONED_IN]->(c:Chunk)
        WHERE seed.name IN $seeds
        RETURN DISTINCT c.chunk_id AS chunk_id
        LIMIT {limit};
        """
        try:
            res_seed = self.conn.execute(seed_chunk_query, {"seeds": seeds_list})
            while res_seed.has_next():
                retrieved_chunk_ids.add(str(res_seed.get_next()[0]))
        except Exception:
            pass

        # 2. Bi-directional 1-hop traversal
        if max_hops >= 1:
            one_hop_query = f"""
            MATCH (seed:Entity)-[r1:RELATION]-(mid:Entity),
                  (mid)-[:MENTIONED_IN]->(c1:Chunk)
            WHERE seed.name IN $seeds AND r1.weight >= $min_weight AND seed.name <> mid.name
            RETURN DISTINCT c1.chunk_id AS mid_chunk, seed.name AS seed,
                   r1.predicate AS p1, mid.name AS bridge
            LIMIT {limit};
            """
            try:
                res1 = self.conn.execute(one_hop_query, {"seeds": seeds_list, "min_weight": weight_val})
                while res1.has_next():
                    row = res1.get_next()
                    cid, seed, p1, bridge = row
                    if cid:
                        retrieved_chunk_ids.add(str(cid))
                    paths.append(
                        TraversalPath(
                            seed=str(seed),
                            predicate_1=str(p1),
                            bridge=str(bridge),
                            predicate_2="DIRECT_NEIGHBOR",
                            target=str(bridge),
                            hop_count=1,
                            mid_chunk_id=str(cid) if cid else None,
                        )
                    )
            except Exception:
                pass

        # 3. Bi-directional 2-hop traversal
        if max_hops >= 2:
            two_hop_query = f"""
            MATCH (seed:Entity)-[r1:RELATION]-(mid:Entity)-[r2:RELATION]-(target:Entity),
                  (mid)-[:MENTIONED_IN]->(c1:Chunk),
                  (target)-[:MENTIONED_IN]->(c2:Chunk)
            WHERE seed.name IN $seeds
              AND r1.weight >= $min_weight AND r2.weight >= $min_weight
              AND seed.name <> target.name AND seed.name <> mid.name AND mid.name <> target.name
            RETURN DISTINCT c1.chunk_id AS mid_chunk, c2.chunk_id AS target_chunk,
                   seed.name AS seed, r1.predicate AS p1, mid.name AS bridge,
                   r2.predicate AS p2, target.name AS target
            LIMIT {limit};
            """
            try:
                res2 = self.conn.execute(two_hop_query, {"seeds": seeds_list, "min_weight": weight_val})
                while res2.has_next():
                    row = res2.get_next()
                    mid_c, tgt_c, seed, p1, bridge, p2, tgt = row
                    if mid_c:
                        retrieved_chunk_ids.add(str(mid_c))
                    if tgt_c:
                        retrieved_chunk_ids.add(str(tgt_c))
                    paths.append(
                        TraversalPath(
                            seed=str(seed),
                            predicate_1=str(p1),
                            bridge=str(bridge),
                            predicate_2=str(p2),
                            target=str(tgt),
                            hop_count=2,
                            mid_chunk_id=str(mid_c) if mid_c else None,
                            target_chunk_id=str(tgt_c) if tgt_c else None,
                        )
                    )
            except Exception:
                pass

        # 4. Bi-directional 3-hop traversal
        if max_hops >= 3:
            three_hop_query = f"""
            MATCH (seed:Entity)-[r1:RELATION]-(mid:Entity)-[r2:RELATION]-(target:Entity)-[r3:RELATION]-(hop3:Entity),
                  (mid)-[:MENTIONED_IN]->(c1:Chunk),
                  (target)-[:MENTIONED_IN]->(c2:Chunk),
                  (hop3)-[:MENTIONED_IN]->(c3:Chunk)
            WHERE seed.name IN $seeds
              AND r1.weight >= $min_weight AND r2.weight >= $min_weight AND r3.weight >= $min_weight
              AND seed.name <> hop3.name AND mid.name <> hop3.name AND target.name <> hop3.name
              AND seed.name <> mid.name AND mid.name <> target.name AND seed.name <> target.name
            RETURN DISTINCT c1.chunk_id AS mid_chunk, c2.chunk_id AS target_chunk, c3.chunk_id AS hop3_chunk,
                   seed.name AS seed, r1.predicate AS p1, mid.name AS bridge, r2.predicate AS p2,
                   target.name AS intermediate, r3.predicate AS p3, hop3.name AS target
            LIMIT {limit};
            """
            try:
                res3 = self.conn.execute(three_hop_query, {"seeds": seeds_list, "min_weight": weight_val})
                while res3.has_next():
                    row = res3.get_next()
                    c1, c2, c3, seed, p1, bridge, p2, inter, p3, tgt = row
                    if c1:
                        retrieved_chunk_ids.add(str(c1))
                    if c2:
                        retrieved_chunk_ids.add(str(c2))
                    if c3:
                        retrieved_chunk_ids.add(str(c3))
                    paths.append(
                        TraversalPath(
                            seed=str(seed),
                            predicate_1=f"{p1} -> {inter}",
                            bridge=str(bridge),
                            predicate_2=str(p3),
                            target=str(tgt),
                            hop_count=3,
                            mid_chunk_id=str(c1) if c1 else None,
                            target_chunk_id=str(c3) if c3 else None,
                        )
                    )
            except Exception:
                pass

        return paths, retrieved_chunk_ids

    def traverse_two_hop(
        self,
        seed_names: Sequence[str],
        min_weight: float = 0.70,
        limit: int = 20,
    ) -> Tuple[List[TraversalPath], Set[str]]:
        """Backward-compatible signature delegating to upgraded bi-directional multi-hop traversal."""
        return self.traverse_paths(seed_names, min_weight=min_weight, max_hops=3, limit=limit)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Fetch chunk metadata and text by chunk_id."""
        cid_esc = chunk_id.replace("'", "''")
        query = f"MATCH (c:Chunk {{chunk_id: '{cid_esc}'}}) RETURN c.chunk_id, c.text, c.doc_title;"
        try:
            res = self.conn.execute(query)
            if res.has_next():
                cid, text, title = res.get_next()
                return Chunk(chunk_id=str(cid), text=str(text), doc_title=str(title))
        except Exception:
            pass
        return None


class LanceVectorStore:
    """Dense vector database backed by embedded LanceDB."""

    def __init__(self, uri: str) -> None:
        self.uri = str(Path(uri).resolve())
        os.makedirs(self.uri, exist_ok=True)
        self.db = lancedb.connect(self.uri)
        try:
            self._init_tables()
        except BaseException:
            try:
                self.db.close()
            except Exception:
                logger.exception("LanceDB cleanup failed after vector-store initialization failed")
            raise

    def close(self) -> None:
        """Release local LanceDB connection resources."""
        self.db.close()

    def _get_table_names(self) -> Set[str]:
        """Fetch set of existing table names cleanly across LanceDB versions."""
        if hasattr(self.db, "list_tables"):
            return set(list(self.db.list_tables()))
        return set(self.db.table_names())

    def _init_tables(self) -> None:
        """Create collections if not existing."""
        self._entity_table_name = "entity_vectors"
        self._chunk_table_name = "chunk_vectors"

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
    ) -> None:
        """Insert or replace chunk vector embeddings."""
        if not chunks or len(vectors) == 0:
            return

        records = []
        for i, chunk in enumerate(chunks):
            records.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "doc_title": chunk.doc_title,
                "vector": vectors[i].tolist(),
            })

        existing_tables = self._get_table_names()
        if self._chunk_table_name in existing_tables:
            table = self.db.open_table(self._chunk_table_name)
            table.add(records)
        else:
            self.db.create_table(self._chunk_table_name, data=records)

    def upsert_entities(
        self,
        entities: Sequence[Entity],
        vectors: np.ndarray,
    ) -> None:
        """Insert or replace entity name vector embeddings."""
        if not entities or len(vectors) == 0:
            return

        records = []
        for i, ent in enumerate(entities):
            records.append({
                "name": ent.name,
                "type": ent.type,
                "description": ent.description,
                "vector": vectors[i].tolist(),
            })

        existing_tables = self._get_table_names()
        if self._entity_table_name in existing_tables:
            table = self.db.open_table(self._entity_table_name)
            table.add(records)
        else:
            self.db.create_table(self._entity_table_name, data=records)

    def search_entities(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search top-k most similar entities to query vector."""
        if self._entity_table_name not in self._get_table_names():
            return []

        table = self.db.open_table(self._entity_table_name)
        results = table.search(query_vector.tolist()).limit(top_k).to_list()
        # LanceDB returns _distance (L2 distance or cosine distance)
        matched = []
        for r in results:
            name = r.get("name", "")
            # Convert distance to approximate similarity score
            dist = float(r.get("_distance", 1.0))
            sim = 1.0 / (1.0 + dist)
            matched.append((name, sim))
        return matched

    def search_chunks(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search top-k nearest chunks to query vector."""
        if self._chunk_table_name not in self._get_table_names():
            return []

        table = self.db.open_table(self._chunk_table_name)
        results = table.search(query_vector.tolist()).limit(top_k).to_list()
        matched = []
        for r in results:
            cid = r.get("chunk_id", "")
            dist = float(r.get("_distance", 1.0))
            sim = 1.0 / (1.0 + dist)
            matched.append((cid, sim))
        return matched
