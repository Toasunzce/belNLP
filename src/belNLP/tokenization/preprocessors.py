from tokenization.base import BasePreprocessor
import unicodedata
import re


"""
patterns:


"""


"""

"""

# TODO check BLREmbeddings project to list every needed preprocessings...


class LowercasePreprocessor(BasePreprocessor):
    """
    
    """
    def process(self, text: str) -> str:
        return text.lower()
    


class UnicodeNormalizer(BasePreprocessor):
    """
    
    """
    def __init__(self, form: str = "NFKC"):
        self._form = form

    def process(self, text: str) -> str:
        return unicodedata.normalize(self._form, text)  # ty:ignore[invalid-argument-type]
    


class WhitespaceNormalizer(BasePreprocessor):
    """
    
    """
    def __init__(self, strip: bool = True):
        self._strip = strip
        self._regex = re.compile(r"\s+")

    def process(self, text: str) -> str:
        text = self._regex.sub(" ", text)
        return text.strip() if self._strip else text
    

# chain of responsibility
class PreprocessorChain(BasePreprocessor):
    """
    
    """
    def __init__(self):
        self._chain: list[BasePreprocessor] = []

    def add(self, preprocessor: BasePreprocessor) -> "PreprocessorChain":
        self._chain.append(preprocessor)
        return self

    def process(self, text: str) -> str:
        for preprocessor in self._chain:
            text = preprocessor.process(text)
        return text