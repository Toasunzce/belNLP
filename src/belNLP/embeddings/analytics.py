from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from belNLP.embeddings.base import StaticEmbedder


class BaseSimilarity(ABC):
    """Base class for word similarity scorers."""

    def __init__(self, embedder: StaticEmbedder) -> None:
        self._embedder = embedder

    @abstractmethod
    def score(self, a: str, b: str) -> float:
        pass

    def __call__(self, a: str, b: str) -> float:
        return self.score(a, b)


class CosineSimilarity(BaseSimilarity):
    """Cosine similarity in [-1, 1]. Higher = more similar.

    Example:
        >>> CosineSimilarity(embedder).score("кот", "сабака")  # -> ~0.7
    """

    def score(self, a: str, b: str) -> float:
        va, vb = self._embedder.embed_word(a), self._embedder.embed_word(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom else 0.0


class EuclideanSimilarity(BaseSimilarity):
    """Negative Euclidean distance (score ≤ 0). Higher = more similar.

    Example:
        >>> EuclideanSimilarity(embedder).score("кот", "сабака")  # -> -2.3
    """

    def score(self, a: str, b: str) -> float:
        va, vb = self._embedder.embed_word(a), self._embedder.embed_word(b)
        return -float(np.linalg.norm(va - vb))


class AnalogyEngine:
    """Solves word analogies: a : b = c : ? via vector arithmetic (b − a + c).

    Example:
        >>> engine = AnalogyEngine(embedder)
        >>> engine.solve("кароль", "каралева", "цар")  # -> ["царыца", ...]
    """

    def __init__(self, embedder: StaticEmbedder) -> None:
        self._embedder = embedder

    def solve(self, a: str, b: str, c: str, topn: int = 5) -> list[str]:
        target = self._embedder.embed_word(b) - self._embedder.embed_word(a) + self._embedder.embed_word(c)
        return self._embedder.most_similar_to_vector(target, topn=topn, exclude={a, b, c})


class PCAReducer:
    """Reduces embedding vectors to n dimensions via PCA.

    Example:
        >>> reducer = PCAReducer(n_components=2)
        >>> coords = reducer.fit_transform(vectors)  # shape (n, 2)
    """

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self._components: np.ndarray | None = None
        self._mean: np.ndarray | None = None

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        """Fit PCA on vectors and return projected coordinates."""
        self._mean = vectors.mean(axis=0)
        centered = vectors - self._mean
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        self._components = Vt[: self.n_components]
        return centered @ self._components.T

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Project new vectors using already-fitted components."""
        if self._components is None or self._mean is None:
            raise RuntimeError("Call fit_transform() before transform().")
        return (vectors - self._mean) @ self._components.T
