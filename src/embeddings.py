"""Embedding providers for dense vector indexing and cosine matching."""

from __future__ import annotations

import abc
import hashlib
import math
from typing import List, Sequence
import numpy as np


class BaseEmbeddingProvider(abc.ABC):
    """Abstract interface for dense vector embeddings."""

    @abc.abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return 2D float32 numpy array of shape (len(texts), dimension)."""
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """Return 1D float32 numpy array for single query."""
        vectors = self.embed_texts([query])
        return vectors[0]

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Return embedding vector dimension."""
        pass

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two 1D vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    @staticmethod
    def batch_cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Compute cosine similarity of a 1D vector against every row in a 2D matrix."""
        if matrix.size == 0:
            return np.array([], dtype=np.float32)
        vector_norm = np.linalg.norm(vector)
        if vector_norm == 0.0:
            return np.zeros(matrix.shape[0], dtype=np.float32)
        row_norms = np.linalg.norm(matrix, axis=1)
        row_norms[row_norms == 0.0] = 1.0
        normalized_matrix = matrix / row_norms[:, np.newaxis]
        normalized_vector = vector / vector_norm
        return np.dot(normalized_matrix, normalized_vector)


class HashProjectionEmbedding(BaseEmbeddingProvider):
    """Deterministic hashing-based dense projection embedding.

    Provides zero-external-dependency, reproducible dense representations
    ideal for unit testing and local offline validation.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        results = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            clean = text.strip().lower()
            if not clean:
                continue

            # Token and character n-gram feature hashing
            words = clean.split()
            vec = np.zeros(self._dim, dtype=np.float32)

            for w in words:
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                idx = h % self._dim
                sign = 1.0 if ((h >> 8) & 1) else -1.0
                vec[idx] += sign

            # Add 3-character n-grams for subword / lexical overlap
            for k in range(len(clean) - 2):
                gram = clean[k : k + 3]
                h_gram = int(hashlib.sha1(gram.encode("utf-8")).hexdigest(), 16)
                idx_g = h_gram % self._dim
                sign_g = 1.0 if ((h_gram >> 8) & 1) else -1.0
                vec[idx_g] += 0.5 * sign_g

            norm = np.linalg.norm(vec)
            if norm > 0.0:
                vec = vec / norm
            results[i] = vec

        return results


class SentenceTransformerEmbedding(BaseEmbeddingProvider):
    """Local dense embedding provider using sentence-transformers with auto-loading, caching, and fallback."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        dimension: int = 384,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._target_dim = dimension
        self._model = None
        self._cache: dict[str, np.ndarray] = {}
        self._fallback_provider: Optional[HashProjectionEmbedding] = None

        self._init_model()

    def _init_model(self) -> None:
        """Auto-load sentence-transformers model with fallback on failure."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.model_name)
            self._target_dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            # Fall back to deterministic HashProjectionEmbedding
            self._fallback_provider = HashProjectionEmbedding(dimension=self._target_dim)

    @property
    def dimension(self) -> int:
        return self._target_dim

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Batch encode texts with in-memory caching and fallback."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        results = np.zeros((len(texts), self.dimension), dtype=np.float32)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        # Check in-memory cache first
        for i, text in enumerate(texts):
            clean_text = text.strip()
            if clean_text in self._cache:
                results[i] = self._cache[clean_text]
            else:
                missing_indices.append(i)
                missing_texts.append(clean_text)

        if not missing_texts:
            return results

        # Compute embeddings for missing texts
        if self._model is not None:
            try:
                # Optimized batch encoding
                encoded_batches: list[np.ndarray] = []
                for b_start in range(0, len(missing_texts), self.batch_size):
                    batch = missing_texts[b_start : b_start + self.batch_size]
                    batch_vecs = self._model.encode(
                        batch,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    encoded_batches.append(batch_vecs.astype(np.float32))

                computed = np.vstack(encoded_batches)
            except Exception:
                # Fallback to hash projection if model call fails
                if self._fallback_provider is None:
                    self._fallback_provider = HashProjectionEmbedding(dimension=self.dimension)
                computed = self._fallback_provider.embed_texts(missing_texts)
        else:
            if self._fallback_provider is None:
                self._fallback_provider = HashProjectionEmbedding(dimension=self.dimension)
            computed = self._fallback_provider.embed_texts(missing_texts)

        # Store in cache and populate results
        for idx, text, vec in zip(missing_indices, missing_texts, computed):
            self._cache[text] = vec
            results[idx] = vec

        return results
