import re

from belNLP.tokenization.base import BaseFilter


class StopWordFilter(BaseFilter):
    """Removes tokens that are in the stop-word set.

    Example:
        >>> f = StopWordFilter({"і", "ў", "на"})
        >>> f.filter(["кот", "і", "сабака"]) == ["кот", "сабака"]
    """

    def __init__(self, words: set[str]) -> None:
        self._words = words

    def filter(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in self._words]


class PunctuationFilter(BaseFilter):
    """Removes tokens that consist entirely of punctuation/symbols.

    Example:
        >>> PunctuationFilter().filter(["кот", ",", "!"]) == ["кот"]
    """

    def __init__(self) -> None:
        self._regex = re.compile(r"^[^\w\s]+$", re.UNICODE)

    def filter(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if not self._regex.match(t)]


class LengthFilter(BaseFilter):
    """Keeps only tokens whose length is within [min_len, max_len].

    Example:
        >>> LengthFilter(min_len=2, max_len=5).filter(["я", "іду", "дадому"]) == ["іду"]
    """

    def __init__(self, min_len: int = 1, max_len: int = 100) -> None:
        self._min = min_len
        self._max = max_len

    def filter(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if self._min <= len(t) <= self._max]


class RegexFilter(BaseFilter):
    """Removes tokens that match the given regex pattern.

    Example:
        >>> RegexFilter(r"\\d+").filter(["слова", "123", "яшчэ"]) == ["слова", "яшчэ"]
    """

    def __init__(self, pattern: str) -> None:
        self._regex = re.compile(pattern)

    def filter(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if not self._regex.match(t)]


class CompositeFilter(BaseFilter):
    """Base class for filters that combine multiple child filters."""

    def __init__(self) -> None:
        self._filters: list[BaseFilter] = []

    def add(self, f: BaseFilter) -> "CompositeFilter":
        self._filters.append(f)
        return self

    def filter(self, tokens: list[str]) -> list[str]:
        raise NotImplementedError


class AndFilter(CompositeFilter):
    """Applies all child filters in sequence (token must pass every filter).

    Example:
        >>> f = AndFilter().add(LengthFilter(min_len=2)).add(PunctuationFilter())
        >>> f.filter(["я", "іду", ","]) == ["іду"]
    """

    def filter(self, tokens: list[str]) -> list[str]:
        for f in self._filters:
            tokens = f.filter(tokens)
        return tokens


class OrFilter(CompositeFilter):
    """Keeps tokens that survive at least one child filter.

    Example:
        >>> f = OrFilter().add(StopWordFilter({"і"})).add(LengthFilter(max_len=2))
        >>> f.filter(["я", "і", "іду"]) == ["я", "і"]
    """

    def filter(self, tokens: list[str]) -> list[str]:
        result = set()
        for f in self._filters:
            result.update(f.filter(tokens))
        return [t for t in tokens if t in result]
