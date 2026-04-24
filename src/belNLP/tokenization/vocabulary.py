from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
import json

from tokenization.base import BaseTokenizer, BaseVocabulary



class SpecialTokens:
    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"

    @classmethod
    def all(cls) -> list[str]:
        return [cls.PAD, cls.UNK, cls.BOS, cls.EOS]
    


class Vocabulary(BaseVocabulary):

    def __init__(self):
        self._token2id: dict[str, int] = {}
        self._id2token: dict[int, str] = {}
        self._add_special_tokens()

    def _add_special_tokens(self) -> None:
        for token in SpecialTokens.all():
            self._add(token)

    def _add(self, token: str) -> None:
        if token not in self._token2id:
            idx = len(self._token2id)
            self._token2id[token] = idx
            self._id2token[idx] = token

    def build(self, tokens: list[str]) -> None:
        for token in tokens:
            self._add(token)

    def token2id(self, token: str) -> int:
        return self._token2id.get(token, self._token2id[SpecialTokens.UNK])

    def id2token(self, id: int) -> str:
        return self._id2token[id]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token2id(t) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id2token(i) for i in ids]

    def encode_char(self, word: str) -> list[int]:
        return [self.token2id(ch) for ch in word]

    def __len__(self) -> int:
        return len(self._token2id)

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._token2id, f, ensure_ascii=False, indent=2)

    def load(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self._token2id = json.load(f)
            self._id2token = {v: k for k, v in self._token2id.items()}



class FrequencyVocabBuilder:

    def __init__(self, min_freq: int = 1, max_size: int | None = None):
        self._min_freq = min_freq
        self._max_size = max_size

    def set_min_freq(self, n: int) -> "FrequencyVocabBuilder":
        self._min_freq = n
        return self

    def set_max_size(self, n: int) -> "FrequencyVocabBuilder":
        self._max_size = n
        return self

    def build(self, corpus: list[str], tokenizer: BaseTokenizer) -> Vocabulary:
        counter: Counter = Counter()
        for text in corpus:
            tokens = tokenizer.tokenize(text)
            counter.update(tokens)

        filtered = [
            token for token, freq in counter.most_common(self._max_size)
            if freq >= self._min_freq
        ]

        vocab = Vocabulary()
        vocab.build(filtered)
        return vocab