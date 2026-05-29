from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Base class for all tokenizers. Calls _preprocess → _tokenize → _postprocess."""

    def tokenize(self, text: str) -> list[str]:
        text = self._preprocess(text)
        tokens = self._tokenize(text)
        return self._postprocess(tokens)

    def __call__(self, text: str) -> list[str]:
        return self.tokenize(text)

    @abstractmethod
    def _tokenize(self, text: str) -> list[str]:
        pass

    def _preprocess(self, text: str) -> str:
        return text

    def _postprocess(self, tokens: list[str]) -> list[str]:
        return tokens


class BasePreprocessor(ABC):
    """Base class for text preprocessors."""

    def __call__(self, text: str) -> str:
        return self.process(text)

    @abstractmethod
    def process(self, text: str) -> str:
        pass


class BaseFilter(ABC):
    """Base class for token filters."""

    @abstractmethod
    def filter(self, tokens: list[str]) -> list[str]:
        pass

    def __call__(self, tokens: list[str]) -> list[str]:
        return self.filter(tokens)


class BaseVocabulary(ABC):
    """Base class for token vocabularies (token ↔ id mapping)."""

    @abstractmethod
    def token2id(self, token: str) -> int:
        pass

    @abstractmethod
    def id2token(self, id: int) -> str:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __contains__(self, token: str) -> bool:
        try:
            self.token2id(token)
            return True
        except KeyError:
            return False
