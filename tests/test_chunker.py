"""Tests for deterministic chunking and hashing."""

import hashlib
import pytest
from src.chunker import DeterministicChunker
from src.models import Document


def test_deterministic_chunk_id():
    text = "John Doe founded Company X in 1998."
    expected_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    computed_id = DeterministicChunker.compute_chunk_id(text)
    assert computed_id == expected_id
    assert len(computed_id) == 16


def test_chunking_token_limits_and_overlap():
    chunker = DeterministicChunker(min_tokens=10, max_tokens=20, overlap_tokens=5, min_valid_tokens=5)
    doc_text = "word " * 35  # 35 words
    doc = Document(doc_id="d1", title="Test Doc", text=doc_text)

    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1

    # Verify chunk_id determinism
    for chunk in chunks:
        assert chunk.chunk_id == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:16]
        assert chunk.token_count <= 20


def test_reject_short_passage():
    chunker = DeterministicChunker(min_tokens=10, max_tokens=50, overlap_tokens=5, min_valid_tokens=10)
    short_doc = Document(doc_id="short", title="Short", text="Too short.")
    chunks = chunker.chunk_document(short_doc)
    assert len(chunks) == 0


def test_utf8_sanitization():
    chunker = DeterministicChunker(min_tokens=5, max_tokens=20, overlap_tokens=2, min_valid_tokens=3)
    doc = Document(doc_id="u1", title="UTF8", text="Café münchen naïve résumé testing valid tokens.")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert "münchen" in chunks[0].text
