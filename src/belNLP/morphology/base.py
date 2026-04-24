from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import LiteralString

@dataclass
class MorphToken:
    """Single token enriched with morphological annotations."""
    text: LiteralString
    lemma: LiteralString | None = None
    pos: LiteralString | None = None
    morph: dict[LiteralString, LiteralString] | None = None


class BaseAnnotator(ABC):
    """Base interface for all morphological annotators."""

    @abstractmethod
    def annotate(self, tokens: list[LiteralString]) -> list[MorphToken]:
        pass

    def __call__(self, tokens: list[LiteralString]) -> list[MorphToken]:
        return self.annotate(tokens)