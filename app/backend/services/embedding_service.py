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
from belNLP.tokenization.tokenizers import WordTokenizer
from belNLP.tokenization.preprocessors import LowercasePreprocessor
from belNLP.tokenization.filters import PunctuationFilter

# Compiled once at module level
_tokenizer  = WordTokenizer()
_lowercase  = LowercasePreprocessor()
_punct      = PunctuationFilter()


def _clean_word(raw: str) -> str:
    """Lowercase a single word and strip any attached punctuation.

    E.g. "Кот," -> "кот";  "Слова" -> "слова"
    Returns the first non-symbol token, or an empty string if none found.
    """
    lowered = _lowercase.process(raw.strip())
    word_parts = _punct.filter(_tokenizer.tokenize(lowered))
    return word_parts[0] if word_parts else ""


class EmbeddingService(BaseService):
    """
    Facade over FastTextEmbedder and analytics components.
    Isolates the API layer from the belNLP library internals.

    All incoming words are lowercased automatically.
    Pure symbol tokens are stripped before embedding lookups.
    """

    def ready(self) -> bool:
        return registry._initialized

    # ------------------------------------------------------------------ #

    def embed(self, req: EmbedRequest) -> EmbedResponse:
        embedder = registry.fasttext

        # Lowercase + strip attached punctuation; deduplicate preserving order
        seen: set[str] = set()
        words: list[str] = []
        for raw in req.words:
            w = _clean_word(raw)
            if w and w not in seen and w in embedder:
                seen.add(w)
                words.append(w)

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
        a = _clean_word(word_a)
        b = _clean_word(word_b)
        cos = CosineSimilarity(embedder).score(a, b)
        euc = EuclideanSimilarity(embedder).score(a, b)
        return SimilarityResponse(cosine=round(cos, 4), euclidean=round(euc, 4))

    def analogy(self, a: str, b: str, c: str, topn: int = 5) -> AnalogyResponse:
        from belNLP.embeddings.analytics import AnalogyEngine
        engine = AnalogyEngine(registry.fasttext)
        results = engine.solve(_clean_word(a), _clean_word(b), _clean_word(c), topn=topn)
        return AnalogyResponse(results=results)

    def nearest(self, word: str, topn: int = 10) -> NearestResponse:
        neighbours = registry.fasttext.most_similar(_clean_word(word), topn=topn)
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