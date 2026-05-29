from __future__ import annotations

from pathlib import Path

import numpy as np

from belNLP.embeddings.base import StaticEmbedder, EmbeddingResult



class FastTextEmbedder(StaticEmbedder):
    """Wraps the native fastText library. Supports OOV via subword n-grams.

    Example:
        >>> embedder = FastTextEmbedder.load("models/cc.be.300.bin")
        >>> embedder.embed_word("кот")  # -> np.ndarray (300,)
    """

    def __init__(self, model) -> None:
        self._model = model

    @classmethod
    def load(cls, path: str | Path) -> "FastTextEmbedder":
        try:
            from gensim.models import KeyedVectors
        except ImportError:
            raise ImportError("gensim is required: pip install gensim")
        model = KeyedVectors.load_word2vec_format(str(path), binary=True)
        return cls(model)

    @property
    def dim(self) -> int:
        return self._model.get_dimension()

    def get_vector(self, word: str) -> np.ndarray:
        return np.array(self._model.get_word_vector(word), dtype=np.float32)

    def most_similar(self, word: str, topn: int = 10) -> list[str]:
        neighbours = self._model.get_nearest_neighbors(word, k=topn)
        return [w for _, w in neighbours]

    def __contains__(self, word: str) -> bool:
        return word in self._model.get_words()

    def most_similar_to_vector(
        self,
        vector: np.ndarray,
        topn: int = 10,
        exclude: set[str] | None = None,
    ) -> list[str]:
        exclude = exclude or set()
        words  = self._model.get_words()
        matrix = np.array([self._model.get_word_vector(w) for w in words], dtype=np.float32)
        norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / (norms + 1e-9)
        query  = vector / (np.linalg.norm(vector) + 1e-9)
        scores = matrix @ query
        top_indices = np.argsort(scores)[::-1]
        result = []
        for i in top_indices:
            w = words[i]
            if w not in exclude:
                result.append(w)
            if len(result) == topn:
                break
        return result



class Word2VecEmbedder(StaticEmbedder):
    """Wraps gensim KeyedVectors (.bin/.kv). OOV words return a zero vector.

    Example:
        >>> embedder = Word2VecEmbedder.load("models/w2v.bin")
        >>> embedder.embed_word("кот")  # -> np.ndarray (300,)
    """

    def __init__(self, keyed_vectors) -> None:
        self._kv = keyed_vectors

    @classmethod
    def load(cls, path: str | Path, binary: bool = True) -> "Word2VecEmbedder":
        try:
            from gensim.models import KeyedVectors
        except ImportError:
            raise ImportError("gensim is required: pip install gensim")
        kv = KeyedVectors.load_word2vec_format(str(path), binary=binary)
        return cls(kv)

    @property
    def dim(self) -> int:
        return self._kv.vector_size

    def get_vector(self, word: str) -> np.ndarray:
        if word in self._kv:
            return np.array(self._kv[word], dtype=np.float32)
        return np.zeros(self.dim, dtype=np.float32)

    def most_similar(self, word: str, topn: int = 10) -> list[str]:
        results = self._kv.most_similar(word, topn=topn)
        return [w for w, _ in results]

    def most_similar_to_vector(
        self,
        vector: np.ndarray,
        topn: int = 10,
        exclude: set[str] | None = None,
    ) -> list[str]:
        exclude = exclude or set()
        results = self._kv.similar_by_vector(vector, topn=topn + len(exclude))
        return [w for w, _ in results if w not in exclude][:topn]

    def __contains__(self, word: str) -> bool:
        return word in self._kv



class GloVeEmbedder(StaticEmbedder):
    """Loads plain-text GloVe vectors (one "word f1 f2 … fn" per line).

    Example:
        >>> embedder = GloVeEmbedder.load("models/glove.txt")
        >>> embedder.embed_word("кот")  # -> np.ndarray (300,)
    """

    def __init__(self, vectors: dict[str, np.ndarray], dim: int) -> None:
        self._vectors = vectors
        self._dim = dim

    @classmethod
    def load(cls, path: str | Path) -> "GloVeEmbedder":
        vectors: dict[str, np.ndarray] = {}
        dim = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word, values = parts[0], parts[1:]
                vectors[word] = np.array(values, dtype=np.float32)
                dim = len(values)
        return cls(vectors, dim)

    @property
    def dim(self) -> int:
        return self._dim

    def get_vector(self, word: str) -> np.ndarray:
        if word not in self._vectors:
            return np.zeros(self._dim, dtype=np.float32)
        return self._vectors[word]

    def most_similar(self, word: str, topn: int = 10) -> list[str]:
        if word not in self._vectors:
            return []
        query = self._vectors[word]
        words  = list(self._vectors.keys())
        matrix = np.stack(list(self._vectors.values()))
        scores = matrix @ query / (
            np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-9
        )
        top_indices = np.argsort(scores)[::-1]
        result = []
        for i in top_indices:
            if words[i] != word:
                result.append(words[i])
            if len(result) == topn:
                break
        return result

    def __contains__(self, word: str) -> bool:
        return word in self._vectors

    def most_similar_to_vector(
        self,
        vector: np.ndarray,
        topn: int = 10,
        exclude: set[str] | None = None,
    ) -> list[str]:
        exclude = exclude or set()
        words  = list(self._vectors.keys())
        matrix = np.stack(list(self._vectors.values()))
        query  = vector / (np.linalg.norm(vector) + 1e-9)
        norms  = np.linalg.norm(matrix, axis=1)
        scores = matrix @ query / (norms + 1e-9)
        top_indices = np.argsort(scores)[::-1]
        result = []
        for i in top_indices:
            w = words[i]
            if w not in exclude:
                result.append(w)
            if len(result) == topn:
                break
        return result



class GensimAdapter(StaticEmbedder):
    """Adapts any gensim KeyedVectors to the StaticEmbedder interface.

    Example:
        >>> kv = KeyedVectors.load("models/bel_ft.model.kv")
        >>> embedder = GensimAdapter(kv)
        >>> embedder.embed_word("кот")   # -> np.ndarray (100,)
        >>> embedder.most_similar("кот") # -> ["сабака", ...]
    """

    def __init__(self, keyed_vectors) -> None:
        self._kv = keyed_vectors

    @property
    def dim(self) -> int:
        return self._kv.vector_size

    def get_vector(self, word: str) -> np.ndarray:
        if word in self._kv:
            return np.array(self._kv[word], dtype=np.float32)
        return np.zeros(self.dim, dtype=np.float32)

    def most_similar(self, word: str, topn: int = 10) -> list[str]:
        results = self._kv.most_similar(word, topn=topn)
        return [w for w, _ in results]

    def most_similar_to_vector(
        self,
        vector: np.ndarray,
        topn: int = 10,
        exclude: set[str] | None = None,
    ) -> list[str]:
        exclude = exclude or set()
        results = self._kv.similar_by_vector(vector, topn=topn + len(exclude))
        return [w for w, _ in results if w not in exclude][:topn]

    def __contains__(self, word: str) -> bool:
        return word in self._kv