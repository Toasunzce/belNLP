from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np



@dataclass
class EmbeddingResult:
    """Container for embedding output."""
    tokens:  list[str]
    vectors: np.ndarray        # shape (n, dim)

    @property
    def dim(self) -> int:
        return self.vectors.shape[1] if self.vectors.ndim == 2 else self.vectors.shape[0]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, token: str) -> np.ndarray:
        idx = self.tokens.index(token)
        return self.vectors[idx]



class BaseEmbedder(ABC):
    """Abstract base for all embedding models."""

    @abstractmethod
    def embed(self, tokens: list[str]) -> EmbeddingResult:
        """Embed a list of tokens, returning one vector per token."""

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
    """
    Base for static (non-contextual) embedders.
    Every word maps to a fixed vector regardless of context.
    """

    @abstractmethod
    def get_vector(self, word: str) -> np.ndarray:
        """Return the stored vector for *word*. Raises KeyError if OOV."""

    @abstractmethod
    def most_similar(self, word: str, topn: int = 10) -> list[str]:
        """Return *topn* nearest neighbours by cosine similarity."""

    @abstractmethod
    def __contains__(self, word: str) -> bool:
        """True if *word* is in the vocabulary."""

    # ------------------------------------------------------------------ #
    # Concrete implementations shared by all static embedders             #
    # ------------------------------------------------------------------ #

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
        """
        Return *topn* vocabulary words whose vectors are closest
        to the given *vector* by cosine similarity.
        Used by AnalogyEngine.
        Subclasses may override this with a faster implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support most_similar_to_vector(). "
            "Override this method or use a model that exposes its full vocabulary."
        )



class ContextualEmbedder(BaseEmbedder):
    """
    Base for contextual embedders (ELMo, BERT, …).
    Vectors depend on the full token sequence.
    """

    @abstractmethod
    def embed_sentence(self, tokens: list[str]) -> EmbeddingResult:
        """
        Embed a sentence, producing context-aware vectors.
        Unlike embed(), the whole sequence is processed at once.
        """

    # embed() delegates to embed_sentence() by default
    def embed(self, tokens: list[str]) -> EmbeddingResult:
        return self.embed_sentence(tokens)

    def embed_word(self, word: str) -> np.ndarray:
        """Single-word context (no surrounding tokens)."""
        return self.embed_sentence([word]).vectors[0]



# ------------------------------------------------------------------ #
# Sentence-level embedders                                            #
# ------------------------------------------------------------------ #

class BaseSentenceEmbedder(ABC):
    """Produces a single fixed-size vector for a whole sentence/sequence."""

    @abstractmethod
    def embed_sentence(self, tokens: list[str]) -> np.ndarray:
        """Return a 1-D vector of shape (dim,)."""

    def __call__(self, tokens: list[str]) -> np.ndarray:
        return self.embed_sentence(tokens)