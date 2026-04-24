from abc import ABC, abstractmethod



"""

"""

"""
module pipeline:
  - preprocessing (e.g. LowercasePreprocessor)
  - tokenizing (e.g. WordTokenizer)
  - postprocessing (WIP)
"""


class BaseTokenizer(ABC):
    """
    Base interface for all tokenization models.
    """
    def tokenize(self, text: str) -> list[str]:
        text = self._preprocess(text)
        tokens = self._tokenize(text)
        return self._postprocess(tokens)

    @abstractmethod
    def _tokenize(self, text: str) -> list[str]:
        pass

    def _preprocess(self, text: str) -> str:
        return text

    def _postprocess(self, tokens: list[str]) -> list[str]:
        return tokens
    


class BasePreprocessor(ABC):
    """
    Base interface for all text preprocessors.
    """

    def __call__(self, text: str) -> str:
        return self.process(text)

    @abstractmethod
    def process(self, text: str) -> str:
        pass


class BaseFilter(ABC):
    """
    
    """
    @abstractmethod
    def filter(self, tokens: list[str]) -> list[str]:
        pass

    def __call__(self, tokens: list[str]) -> list[str]:
        return self.filter(tokens)
    


class BaseVocabulary(ABC):

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
        

