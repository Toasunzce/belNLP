from tokenization.base import BaseFilter
import re


class StopWordFilter(BaseFilter):
    """
    
    """
    def __init__(self, words: set[str]):
        self._words = words

    def filter(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in self._words]



class PunctuationFilter(BaseFilter):
    """
    
    """
    def __init__(self):
        self._regex = re.compile(r"^[^\w\s]+$")

    def filter(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if not self._regex.match(t)]



class LengthFilter(BaseFilter):
    """
    
    """
    def __init__(self, min_len: int = 1, max_len: int = 100):
        self._min = min_len
        self._max = max_len

    def filter(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if self._min <= len(t) <= self._max]



class RegexFilter(BaseFilter):
    """
    
    """
    def __init__(self, pattern: str):
        self._regex = re.compile(pattern)

    def filter(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if not self._regex.match(t)]



class CompositeFilter(BaseFilter):
    """
    
    """
    def __init__(self):
        self._filters: list[BaseFilter] = []

    def add(self, f: BaseFilter) -> "CompositeFilter":
        self._filters.append(f)
        return self

    def filter(self, tokens: list[str]) -> list[str]:
        raise NotImplementedError



class AndFilter(CompositeFilter):
    """

    """
    def filter(self, tokens: list[str]) -> list[str]:
        for f in self._filters:
            tokens = f.filter(tokens)
        return tokens



class OrFilter(CompositeFilter):
    """

    """
    def filter(self, tokens: list[str]) -> list[str]:
        result = set()
        for f in self._filters:
            result.update(f.filter(tokens))
        return [t for t in tokens if t in result]