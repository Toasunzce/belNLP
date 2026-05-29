from __future__ import annotations

import numpy as np

from belNLP.embeddings.base import BaseEmbedder, BaseSentenceEmbedder


class MeanPoolingSentenceEmbedder(BaseSentenceEmbedder):
    """Produces a sentence vector by averaging word embeddings (mean pooling).

    Accepts any BaseEmbedder as the underlying word model.

    Example:
        >>> sent_emb = MeanPoolingSentenceEmbedder(word_embedder)
        >>> vec = sent_emb.embed_sentence(["я", "іду", "дадому"])
        >>> vec.shape  # -> (300,)
    """

    def __init__(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder

    @property
    def dim(self) -> int:
        return self._embedder.dim

    def embed_sentence(self, tokens: list[str]) -> np.ndarray:
        return self._embedder.embed(tokens).vectors.mean(axis=0)
