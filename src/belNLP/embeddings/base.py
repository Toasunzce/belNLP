from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EmbeddingResult:
    """Container for a batch of word embeddings.

    Attributes:
        tokens:  The input words.
        vectors: Float32 matrix of shape (n, dim).

    Example:
        >>> result = embedder.embed(["кот", "сабака"])
        >>> result.vectors.shape  # -> (2, 300)
        >>> result["кот"]         # -> np.ndarray of shape (300,)
    """
    tokens:  list[str]
    vectors: np.ndarray

    @property
    def dim(self) -> int:
        return self.vectors.shape[1] if self.vectors.ndim == 2 else self.vectors.shape[0]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, token: str) -> np.ndarray:
        return self.vectors[self.tokens.index(token)]


class BaseEmbedder(ABC):
    """Base class for all embedding models."""

    @abstractmethod
    def embed(self, tokens: list[str]) -> EmbeddingResult:
        """Embed a list of tokens, one vector per token."""

    @abstractmethod
    def embed_word(self, word: str) -> np.ndarray:
        """Return a single vector for one word."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""

    def __call__(self, tokens: list[str]) -> EmbeddingResult:
        return self.embed(tokens)


class StaticEmbedder(BaseEmbedder):
    """Base for static embedders (Word2Vec, FastText, GloVe).
    Each word maps to a fixed vector regardless of context.
    """

    @abstractmethod
    def get_vector(self, word: str) -> np.ndarray:
        """Return the stored vector for word. Raises KeyError if OOV."""

    @abstractmethod
    def most_similar(self, word: str, topn: int = 10) -> list[str]:
        """Return topn nearest neighbours by cosine similarity."""

    @abstractmethod
    def __contains__(self, word: str) -> bool:
        """True if word is in the vocabulary."""

    def embed_word(self, word: str) -> np.ndarray:
        return self.get_vector(word)

    def embed(self, tokens: list[str]) -> EmbeddingResult:
        vectors = np.stack([self.embed_word(t) for t in tokens])
        return EmbeddingResult(tokens=tokens, vectors=vectors)

    def most_similar_to_vector(
        self,
        vector: np.ndarray,
        topn: int = 10,
        exclude: set[str] | None = None,
    ) -> list[str]:
        """Return topn vocabulary words closest to vector by cosine similarity."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement most_similar_to_vector(). "
            "Override this method or use a model that exposes its full vocabulary."
        )


class ContextualEmbedder(BaseEmbedder):
    """Base for contextual embedders (ELMo, BERT).
    Vectors depend on the full token sequence.
    """

    @abstractmethod
    def embed_sentence(self, tokens: list[str]) -> EmbeddingResult:
        """Embed a full sentence, producing context-aware vectors."""

    def embed(self, tokens: list[str]) -> EmbeddingResult:
        return self.embed_sentence(tokens)

    def embed_word(self, word: str) -> np.ndarray:
        return self.embed_sentence([word]).vectors[0]


class BaseSentenceEmbedder(ABC):
    """Base for sentence-level embedders that produce a single vector per sentence."""

    @abstractmethod
    def embed_sentence(self, tokens: list[str]) -> np.ndarray:
        """Return a 1-D vector of shape (dim,)."""

    def __call__(self, tokens: list[str]) -> np.ndarray:
        return self.embed_sentence(tokens)
