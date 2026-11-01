"""Tests for entity canonicalization and resolution."""

import pytest
from src.embeddings import HashProjectionEmbedding
from src.extraction import EntityCanonicalizer
from src.models import Entity


def test_lexical_case_folding_and_alias_merging():
    embedder = HashProjectionEmbedding(dimension=128)
    canonicalizer = EntityCanonicalizer(embedding_provider=embedder, similarity_threshold=0.88)

    entities = [
        Entity(name="John Doe", type="Person", description="Industrialist and founder"),
        Entity(name="john doe", type="Person", description="American engineer"),
        Entity(name="JOHN DOE", type="Person", description="CEO of Company X"),
    ]

    canonical = canonicalizer.canonicalize(entities)
    assert len(canonical) == 1
    assert canonical[0].type == "Person"
    assert "john doe" in [a.lower() for a in canonical[0].aliases] or canonical[0].name.lower() == "john doe"
    assert "industrialist" in canonical[0].description.lower()


def test_semantic_threshold_merging():
    import numpy as np
    from unittest.mock import MagicMock

    mock_embedder = MagicMock()
    # Return two vectors with cosine similarity > 0.88 (e.g. dot product = 0.92)
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.95, 0.312], dtype=np.float32)
    v2 = v2 / np.linalg.norm(v2)
    mock_embedder.embed_texts.return_value = np.stack([v1, v2])
    mock_embedder.cosine_similarity.return_value = float(np.dot(v1, v2))  # ~0.95

    canonicalizer = EntityCanonicalizer(embedding_provider=mock_embedder, similarity_threshold=0.88)

    entities = [
        Entity(name="Apple Inc.", type="Organization", description="Consumer technology creator"),
        Entity(name="Apple Incorporated", type="Organization", description="Consumer technology creator"),
    ]

    canonical = canonicalizer.canonicalize(entities)
    assert len(canonical) == 1
    assert "Apple Inc." in canonical[0].name or "Apple Incorporated" in canonical[0].name


def test_fuzzy_token_abbreviation_similarity():
    embedder = HashProjectionEmbedding(dimension=128)
    canonicalizer = EntityCanonicalizer(embedding_provider=embedder, similarity_threshold=0.99, fuzzy_threshold=0.85)

    entities = [
        Entity(name="Google Inc", type="Organization", description="Search company"),
        Entity(name="Google Incorporated", type="Organization", description="Search company"),
    ]

    canonical = canonicalizer.canonicalize(entities)
    assert len(canonical) == 1
    assert "Google Inc" in canonical[0].name or "Google Incorporated" in canonical[0].name
