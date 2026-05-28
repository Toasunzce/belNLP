from decimal import Decimal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar


T_in  = TypeVar("T_in")
T_out = TypeVar("T_out")



@dataclass
class MorphToken:
    """Single token enriched with morphological annotations."""
    text:  str
    lemma: str | None = None
    pos:   str | None = None
    morph: dict[str, str] | None = None



class BaseAnnotator(ABC, Generic[T_in, T_out]):
    """"""
    @abstractmethod
    def annotate(self, tokens: list[T_in]) -> list[T_out]:
        pass

    def __call__(self, tokens: list[T_in]) -> list[T_out]:
        return self.annotate(tokens)