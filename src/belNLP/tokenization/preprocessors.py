import re
import unicodedata

from belNLP.tokenization.base import BasePreprocessor


class LowercasePreprocessor(BasePreprocessor):
    """Converts text to lowercase.

    Example:
        >>> LowercasePreprocessor().process("Прывет Свет") == "прывет свет"
    """

    def process(self, text: str) -> str:
        return text.lower()


class UnicodeNormalizer(BasePreprocessor):
    """Normalizes unicode to a given form (default NFKC).

    Example:
        >>> UnicodeNormalizer().process("ﬁle") == "file"
    """

    def __init__(self, form: str = "NFKC") -> None:
        self._form = form

    def process(self, text: str) -> str:
        return unicodedata.normalize(self._form, text)


class WhitespaceNormalizer(BasePreprocessor):
    """Collapses multiple whitespace characters into a single space.

    Example:
        >>> WhitespaceNormalizer().process("а  б   в") == "а б в"
    """

    def __init__(self, strip: bool = True) -> None:
        self._strip = strip
        self._regex = re.compile(r"\s+")

    def process(self, text: str) -> str:
        text = self._regex.sub(" ", text)
        return text.strip() if self._strip else text


class PreprocessorChain(BasePreprocessor):
    """Chains multiple preprocessors in sequence (Chain of Responsibility).

    Example:
        >>> chain = PreprocessorChain().add(LowercasePreprocessor()).add(WhitespaceNormalizer())
        >>> chain.process("  Прывет  Свет  ") == "прывет свет"
    """

    def __init__(self) -> None:
        self._chain: list[BasePreprocessor] = []

    def add(self, preprocessor: BasePreprocessor) -> "PreprocessorChain":
        self._chain.append(preprocessor)
        return self

    def process(self, text: str) -> str:
        for preprocessor in self._chain:
            text = preprocessor.process(text)
        return text
