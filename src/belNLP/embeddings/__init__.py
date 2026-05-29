"""
belNLP.embeddings
~~~~~~~~~~~~~~~~~
Static and contextual word embeddings, sentence embeddings, and analytics.

Example usage:
    from gensim.models import KeyedVectors
    from belNLP.embeddings.static import GensimAdapter
    from belNLP.embeddings.analytics import CosineSimilarity, AnalogyEngine, PCAReducer
    from belNLP.embeddings.sentence import MeanPoolingSentenceEmbedder

    kv = KeyedVectors.load("models/bel_ft.model.kv")
    emb = GensimAdapter(kv)

    emb.embed_word("кот")                          # -> np.ndarray (100,)
    emb.embed(["кот", "сабака"]).vectors           # -> ndarray (2, 100)
    emb.most_similar("кот", topn=5)               # -> ["кацяня", ...]

    CosineSimilarity(emb).score("кот", "сабака")   # -> 0.73
    AnalogyEngine(emb).solve("кот", "кацяня", "сабака")  # -> ["шчаня", ...]

    PCAReducer(n_components=2).fit_transform(emb.embed(words).vectors)

    MeanPoolingSentenceEmbedder(emb).embed_sentence(["я", "іду"])  # -> (100,)
"""

from belNLP.embeddings.base import (
    EmbeddingResult,
    BaseEmbedder,
    StaticEmbedder,
    ContextualEmbedder,
    BaseSentenceEmbedder,
)
from belNLP.embeddings.static import (
    GensimAdapter,
    Word2VecEmbedder,
    FastTextEmbedder,
    GloVeEmbedder,
)
from belNLP.embeddings.sentence import MeanPoolingSentenceEmbedder
from belNLP.embeddings.analytics import (
    CosineSimilarity,
    EuclideanSimilarity,
    AnalogyEngine,
    PCAReducer,
)

__all__ = [
    # base
    "EmbeddingResult", "BaseEmbedder", "StaticEmbedder",
    "ContextualEmbedder", "BaseSentenceEmbedder",
    # static models
    "GensimAdapter", "Word2VecEmbedder", "FastTextEmbedder", "GloVeEmbedder",
    # sentence
    "MeanPoolingSentenceEmbedder",
    # analytics
    "CosineSimilarity", "EuclideanSimilarity", "AnalogyEngine", "PCAReducer",
]
