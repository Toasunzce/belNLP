from abc import ABC, abstractmethod



"""

"""



class BaseTokenizer(ABC):
    """
    Public interface for basic tokenizers
    """
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass
