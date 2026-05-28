from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from belNLP.embeddings.base import StaticEmbedder



# ------------------------------------------------------------------ #
# Similarity                                                          #
# ------------------------------------------------------------------ #

class BaseSimilarity(ABC):
    """Computes a scalar similarity score between two words."""

    def __init__(self, embedder: StaticEmbedder) -> None:
        self._embedder = embedder

    @abstractmethod
    def score(self, a: str, b: str) -> float:
        pass

    def __call__(self, a: str, b: str) -> float:
        return self.score(a, b)



class CosineSimilarity(BaseSimilarity):
    """
    Cosine similarity: score in [-1, 1], higher = more similar.

    Usage:
        >>> sim = CosineSimilarity(embedder)
        >>> sim.score("кот", "пёс")
    """

    def score(self, a: str, b: str) -> float:
        va = self._embedder.embed_word(a)
        vb = self._embedder.embed_word(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)



class EuclideanSimilarity(BaseSimilarity):
    """
    Similarity as negative Euclidean distance: score <= 0, higher = more similar.

    Usage:
        >>> sim = EuclideanSimilarity(embedder)
        >>> sim.score("кот", "пёс")
    """

    def score(self, a: str, b: str) -> float:
        va = self._embedder.embed_word(a)
        vb = self._embedder.embed_word(b)
        return -float(np.linalg.norm(va - vb))



# ------------------------------------------------------------------ #
# Analogy                                                             #
# ------------------------------------------------------------------ #

class AnalogyEngine:
    """
    Solves word analogies: a is to b as c is to ?
    Uses the classic vector arithmetic: vec(b) - vec(a) + vec(c).

    Usage:
        >>> engine = AnalogyEngine(embedder)
        >>> engine.solve("кароль", "каралева", "цар")   # -> ["царыца", ...]
    """

    def __init__(self, embedder: StaticEmbedder) -> None:
        self._embedder = embedder

    def solve(self, a: str, b: str, c: str, topn: int = 5) -> list[str]:
        va = self._embedder.embed_word(a)
        vb = self._embedder.embed_word(b)
        vc = self._embedder.embed_word(c)
        target = vb - va + vc
        return self._embedder.most_similar_to_vector(target, topn=topn, exclude={a, b, c})



# ------------------------------------------------------------------ #
# Dimensionality reduction                                            #
# ------------------------------------------------------------------ #

class PCAReducer:
    """
    Reduces embedding vectors to 2-D (or n-D) via PCA.
    Useful for visualisation in the web service.

    Usage:
        >>> reducer = PCAReducer(n_components=2)
        >>> coords = reducer.fit_transform(vectors)   # np.ndarray (n, 2)
    """

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self._components: np.ndarray | None = None
        self._mean: np.ndarray | None = None

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        """Fit PCA on *vectors* and return the projected coordinates."""
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