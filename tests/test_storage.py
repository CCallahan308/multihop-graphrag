"""Tests for Kùzu property graph and LanceDB vector store."""

import shutil
import numpy as np
import pytest
from src.models import Chunk, Entity, Relationship
from src.storage import KuzuGraphStore, LanceVectorStore


@pytest.fixture
def temp_stores(tmp_path):
    kuzu_path = str(tmp_path / "kuzu")
    lance_path = str(tmp_path / "lance")
    k_store = KuzuGraphStore(db_path=kuzu_path, read_only=False)
    v_store = LanceVectorStore(uri=lance_path)
    yield k_store, v_store


def test_kuzu_schema_and_two_hop_traversal(temp_stores):
    k_store, _ = temp_stores

    entities = [
        Entity(name="Company X", type="Organization", description="Tech firm"),
        Entity(name="John Doe", type="Person", description="Founder"),
        Entity(name="Chicago", type="Location", description="City in Illinois"),
    ]
    chunks = [
        Chunk(chunk_id="c1", text="John Doe founded Company X in 1998.", doc_title="Doc1"),
        Chunk(chunk_id="c2", text="John Doe was born in Chicago, Illinois.", doc_title="Doc2"),
    ]
    rels = [
        Relationship(source_entity="Company X", target_entity="John Doe", predicate="FOUNDED_BY", confidence=0.95),
        Relationship(source_entity="John Doe", target_entity="Chicago", predicate="BORN_IN", confidence=0.95),
    ]

    k_store.add_entities(entities)
    k_store.add_chunks(chunks)
    k_store.add_relationships(rels, chunk_id="c1")
    k_store.link_entity_to_chunk("John Doe", "c1")
    k_store.link_entity_to_chunk("Chicago", "c2")

    paths, chunk_ids = k_store.traverse_two_hop(["Company X"], min_weight=0.70)
    assert len(paths) >= 1
    assert "c1" in chunk_ids or "c2" in chunk_ids
    assert any(p.target == "Chicago" for p in paths)


def test_bidirectional_inverted_edge_traversal(temp_stores):
    k_store, _ = temp_stores

    entities = [
        Entity(name="Green", type="Work", description="Album"),
        Entity(name="Steve Hillage", type="Person", description="Lead artist"),
        Entity(name="Miquette Giraudy", type="Person", description="Spouse"),
    ]
    chunks = [
        Chunk(chunk_id="chunk_album", text="Steve Hillage played Green.", doc_title="Doc1"),
        Chunk(chunk_id="chunk_spouse", text="Steve Hillage married Miquette Giraudy.", doc_title="Doc2"),
    ]
    # Inverted edge: (Steve Hillage)-[:PERFORMER]->(Green) rather than Green -> Steve
    rels = [
        Relationship(source_entity="Steve Hillage", target_entity="Green", predicate="PERFORMER", confidence=0.95),
        Relationship(source_entity="Steve Hillage", target_entity="Miquette Giraudy", predicate="SPOUSE_OF", confidence=0.95),
    ]

    k_store.add_entities(entities)
    k_store.add_chunks(chunks)
    k_store.add_relationships(rels, chunk_id="chunk_album")
    k_store.link_entity_to_chunk("Steve Hillage", "chunk_album")
    k_store.link_entity_to_chunk("Miquette Giraudy", "chunk_spouse")

    # Traverse from seed 'Green' (should follow inverted edge to Steve Hillage and on to Miquette Giraudy)
    paths, chunk_ids = k_store.traverse_paths(["Green"], min_weight=0.70, max_hops=2)
    assert len(paths) >= 1
    assert "chunk_spouse" in chunk_ids
    assert any(p.target == "Miquette Giraudy" for p in paths)


def test_lancedb_upsert_and_search(temp_stores):
    _, v_store = temp_stores

    chunks = [
        Chunk(chunk_id="chunk_a", text="Apple announced new hardware.", doc_title="DocA"),
        Chunk(chunk_id="chunk_b", text="Chicago weather is windy.", doc_title="DocB"),
    ]
    vec_a = np.zeros(64, dtype=np.float32)
    vec_a[0] = 1.0
    vec_b = np.zeros(64, dtype=np.float32)
    vec_b[1] = 1.0

    vectors = np.stack([vec_a, vec_b])
    v_store.upsert_chunks(chunks, vectors)

    # Search for chunk matching vec_a
    results = v_store.search_chunks(vec_a, top_k=1)
    assert len(results) == 1
    assert results[0][0] == "chunk_a"


@pytest.mark.parametrize("weak_edge", [None, 0, 1, 2])
@pytest.mark.parametrize("reverse_edges", [False, True])
def test_traversal_filters_each_edge_at_inclusive_confidence_threshold(
    temp_stores, weak_edge, reverse_edges
):
    k_store, _ = temp_stores
    names = ["Seed", "Bridge", "Target", "Terminal"]
    chunks = [Chunk(chunk_id=f"c{i}", text=name) for i, name in enumerate(names)]
    k_store.add_entities([Entity(name=name) for name in names])
    k_store.add_chunks(chunks)
    for name, chunk in zip(names, chunks):
        k_store.link_entity_to_chunk(name, chunk.chunk_id)

    for index, (source, target) in enumerate(zip(names, names[1:])):
        if reverse_edges:
            source, target = target, source
        k_store.add_relationships([
            Relationship(
                source_entity=source,
                target_entity=target,
                predicate="RELATED_TO",
                confidence=0.69 if index == weak_edge else 0.70,
            )
        ])

    paths, chunk_ids = k_store.traverse_paths(["Seed"], min_weight=0.70)
    last_reachable = 3 if weak_edge is None else weak_edge
    assert chunk_ids == {f"c{i}" for i in range(last_reachable + 1)}
    assert {path.target for path in paths} == set(names[1:last_reachable + 1])
    assert {path.target: path.hop_count for path in paths} == {
        names[index]: index for index in range(1, last_reachable + 1)
    }


def test_traversal_accepts_confidence_endpoints(temp_stores):
    k_store, _ = temp_stores
    k_store.add_entities([Entity(name="Seed"), Entity(name="Target")])
    k_store.add_chunks([Chunk(chunk_id="target", text="Target passage")])
    k_store.link_entity_to_chunk("Target", "target")
    k_store.add_relationships([
        Relationship(source_entity="Seed", target_entity="Target", predicate="RELATED_TO", confidence=0.0)
    ])

    paths, chunk_ids = k_store.traverse_paths(["Seed"], min_weight=0.0)
    assert [path.target for path in paths] == ["Target"]
    assert paths[0].hop_count == 1
    assert chunk_ids == {"target"}
    assert k_store.traverse_paths(["Seed"], min_weight=1.0) == ([], set())


def test_traversal_rejects_invalid_confidence_thresholds(temp_stores):
    k_store, _ = temp_stores
    for min_weight in (-0.01, 1.01, float("nan"), float("inf"), -float("inf"), None, "0.7", True):
        with pytest.raises(ValueError, match="min_weight must be a finite number between 0 and 1"):
            k_store.traverse_paths(["Seed"], min_weight=min_weight)
        with pytest.raises(ValueError, match="min_weight"):
            k_store.traverse_two_hop([], min_weight=min_weight)
