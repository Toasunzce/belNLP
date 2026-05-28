from __future__ import annotations

import numpy as np

from belNLP.embeddings.base import BaseEmbedder, BaseSentenceEmbedder



class MeanPoolingSentenceEmbedder(BaseSentenceEmbedder):
    """
    Produces a sentence vector by averaging token embeddings.
    Accepts any BaseEmbedder as the underlying word-level model.

    Pattern: Adapter / Strategy — the embedder is injected at construction,
    making it easy to swap Word2Vec for FastText without changing this class.

    Usage:
        >>> embedder = FastTextEmbedder.load("models/cc.be.300.bin")
        >>> sentence_embedder = MeanPoolingSentenceEmbedder(embedder)
        >>> vec = sentence_embedder.embed_sentence(["Я", "іду", "дадому"])
    """

    def __init__(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder

    @property
    def dim(self) -> int:
        return self._embedder.dim

    def embed_sentence(self, tokens: list[str]) -> np.ndarray:
        result = self._embedder.embed(tokens)
        return result.vectors.mean(axis=0)