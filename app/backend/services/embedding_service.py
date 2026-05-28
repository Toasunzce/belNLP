from __future__ import annotations

import numpy as np

from services.base import BaseService
from core.model_registry import registry
from schemas.embeddings import (
    EmbedRequest, EmbedResponse, EmbedPoint,
    SimilarityResponse,
    AnalogyResponse,
    NearestResponse,
)


class EmbeddingService(BaseService):
    """
    Facade over FastTextEmbedder and analytics components.
    Isolates the API layer from the belNLP library internals.
    """

    def ready(self) -> bool:
        return registry._initialized

    # ------------------------------------------------------------------ #

    def embed(self, req: EmbedRequest) -> EmbedResponse:
        embedder = registry.fasttext
        words = [w for w in req.words if w in embedder]
        if not words:
            return EmbedResponse(points=[], dim=embedder.dim)

        result = embedder.embed(words)
        vectors = result.vectors

        reduced = self._reduce(vectors, req.reduction, req.n_components)
        if reduced.ndim == 1:
            reduced = reduced.reshape(1, -1)

        points = []
        for word, coords in zip(words, reduced):
            x = float(coords[0]) if len(coords) > 0 else 0.0
            y = float(coords[1]) if len(coords) > 1 else 0.0
            z = float(coords[2]) if len(coords) > 2 else 0.0
            points.append(EmbedPoint(word=word, x=x, y=y, z=z))

        return EmbedResponse(points=points, dim=embedder.dim)

    def similarity(self, word_a: str, word_b: str) -> SimilarityResponse:
        from belNLP.embeddings.analytics import CosineSimilarity, EuclideanSimilarity
        embedder = registry.fasttext
        cos = CosineSimilarity(embedder).score(word_a, word_b)
        euc = EuclideanSimilarity(embedder).score(word_a, word_b)
        return SimilarityResponse(cosine=round(cos, 4), euclidean=round(euc, 4))

    def analogy(self, a: str, b: str, c: str, topn: int = 5) -> AnalogyResponse:
        from belNLP.embeddings.analytics import AnalogyEngine
        engine = AnalogyEngine(registry.fasttext)
        results = engine.solve(a, b, c, topn=topn)
        return AnalogyResponse(results=results)

    def nearest(self, word: str, topn: int = 10) -> NearestResponse:
        neighbours = registry.fasttext.most_similar(word, topn=topn)
        return NearestResponse(neighbours=neighbours)

    # ------------------------------------------------------------------ #

    def _reduce(self, vectors: np.ndarray, method: str, n: int) -> np.ndarray:
        if method == "tsne":
            from sklearn.manifold import TSNE
            perplexity = min(30, max(5, len(vectors) - 1))
            return TSNE(n_components=n, perplexity=perplexity,
                        random_state=42).fit_transform(vectors)
        if method == "umap":
            try:
                import umap
                return umap.UMAP(n_components=n,
                                 random_state=42).fit_transform(vectors)
            except ImportError:
                pass
        # default PCA
        from belNLP.embeddings.analytics import PCAReducer
        return PCAReducer(n_components=n).fit_transform(vectors)