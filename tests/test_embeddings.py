"""Tests for embedding providers with caching and fallback."""

import sys
from types import ModuleType
from unittest.mock import MagicMock
import numpy as np
import pytest
from src.embeddings import HashProjectionEmbedding, SentenceTransformerEmbedding


@pytest.fixture
def sentence_transformer_mock(monkeypatch):
    """Replace the optional package before the provider imports its constructor."""
    constructor = MagicMock()
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = constructor
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return constructor


def test_sentence_transformer_caching(sentence_transformer_mock):
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 64
    mock_model.encode.return_value = np.ones((2, 64), dtype=np.float32)

    sentence_transformer_mock.return_value = mock_model
    embedder = SentenceTransformerEmbedding(batch_size=16, dimension=64)
    texts = ["apple pie", "banana bread", "apple pie"]

    vecs1 = embedder.embed_texts(texts)
    assert len(vecs1) == 3
    assert "apple pie" in embedder._cache
    sentence_transformer_mock.assert_called_once_with(embedder.model_name)

    # Verify second embed uses cached vector without re-encoding
    mock_model.encode.reset_mock()
    vecs2 = embedder.embed_texts(["apple pie"])
    assert np.array_equal(vecs1[0], vecs2[0])
    mock_model.encode.assert_not_called()


def test_sentence_transformer_batch_encoding(sentence_transformer_mock):
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 64
    mock_model.encode.side_effect = lambda batch, **kwargs: np.ones((len(batch), 64), dtype=np.float32)

    sentence_transformer_mock.return_value = mock_model
    embedder = SentenceTransformerEmbedding(batch_size=4, dimension=64)
    texts = [f"sample text number {i}" for i in range(10)]
    vecs = embedder.embed_texts(texts)
    assert vecs.shape == (10, 64)
    # 10 items in batches of 4 -> 3 calls to encode
    assert mock_model.encode.call_count == 3
    sentence_transformer_mock.assert_called_once_with(embedder.model_name)


def test_embedding_fallback_on_invalid_model(sentence_transformer_mock):
    sentence_transformer_mock.side_effect = RuntimeError("Cannot load")
    embedder = SentenceTransformerEmbedding(model_name="non_existent/model_xyz", dimension=64)
    assert embedder._fallback_provider is not None
    vec = embedder.embed_query("test query")
    assert vec.shape == (64,)
    sentence_transformer_mock.assert_called_once_with("non_existent/model_xyz")
