"""Deterministic passage chunker for document ingestion."""

from __future__ import annotations

import hashlib
import re
from typing import List, Sequence
from src.models import Chunk, Document


class DeterministicChunker:
    """Splits documents into overlapping token passages with deterministic SHA-256 keys."""

    def __init__(
        self,
        min_tokens: int = 256,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        min_valid_tokens: int = 10,
    ) -> None:
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be strictly less than max_tokens.")
        if min_tokens > max_tokens:
            raise ValueError("min_tokens cannot exceed max_tokens.")

        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_valid_tokens = min_valid_tokens
        # Tokenizer regex preserves words, punctuation, and contractions
        self._token_pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def tokenize(self, text: str) -> List[str]:
        """Convert input text into token strings."""
        return self._token_pattern.findall(text)

    def detokenize(self, tokens: Sequence[str]) -> str:
        """Reconstruct readable text from token sequence with proper spacing."""
        text = " ".join(tokens)
        # Clean up space before punctuation
        text = re.sub(r"\s+([.,;:!?')\]}])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        return text.strip()

    @staticmethod
    def compute_chunk_id(chunk_text: str) -> str:
        """Assign deterministic primary key via 16-character SHA-256 hex digest."""
        return hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16]

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a single document into deterministic passages."""
        # Validate UTF-8 encoding and clean raw string
        raw_text = document.text
        if isinstance(raw_text, bytes):
            raw_text = raw_text.decode("utf-8", errors="replace")

        tokens = self.tokenize(raw_text)
        total_tokens = len(tokens)

        # Reject passages with fewer than minimum valid tokens
        if total_tokens < self.min_valid_tokens:
            return []

        # If document fits within max_tokens, emit single chunk
        if total_tokens <= self.max_tokens:
            clean_text = self.detokenize(tokens)
            return [
                Chunk(
                    chunk_id=self.compute_chunk_id(clean_text),
                    text=clean_text,
                    doc_title=document.title,
                    token_count=total_tokens,
                )
            ]

        chunks: List[Chunk] = []
        stride = self.max_tokens - self.overlap_tokens
        start = 0

        while start < total_tokens:
            end = min(start + self.max_tokens, total_tokens)
            chunk_tokens = tokens[start:end]

            # If the trailing slice is smaller than min_valid_tokens, skip or append to previous
            if len(chunk_tokens) < self.min_valid_tokens:
                break

            chunk_text = self.detokenize(chunk_tokens)
            chunk_id = self.compute_chunk_id(chunk_text)

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    doc_title=document.title,
                    token_count=len(chunk_tokens),
                )
            )

            if end >= total_tokens:
                break
            start += stride

        return chunks

    def chunk_documents(self, documents: Sequence[Document]) -> List[Chunk]:
        """Process multiple documents in sequence."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
