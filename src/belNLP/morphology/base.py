from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

T_in  = TypeVar("T_in")
T_out = TypeVar("T_out")


@dataclass
class MorphToken:
    """A token with optional morphological annotations.

    Attributes:
        text:  The token string.
        lemma: Dictionary form (filled by Lemmatizer).
        pos:   Part-of-speech tag (filled by POSTagger).
        morph: Extra metadata (e.g. model confidence).
    """
    text:  str
    lemma: str | None            = None
    pos:   str | None            = None
    morph: dict[str, str] | None = None


class BaseAnnotator(ABC, Generic[T_in, T_out]):
    """Base class for annotators that enrich a list of tokens.

    T_in  — input token type (str for POSTagger, MorphToken for Lemmatizer).
    T_out — output token type (MorphToken for both).
    """

    @abstractmethod
    def annotate(self, tokens: list[T_in]) -> list[T_out]:
        pass

    def __call__(self, tokens: list[T_in]) -> list[T_out]:
        return self.annotate(tokens)
