from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from belNLP.morphology.base import MorphToken
    from belNLP.embeddings.base import EmbeddingResult


@dataclass
class Annotation:
    """Carries NLP results through all pipeline steps.

    Fields are None until the corresponding step fills them:
        raw_text       — always present (original input, never changed)
        processed_text — set by PreprocessStep
        tokens         — set by TokenizeStep, synced by FilterMorphStep
        morph_tokens   — set by POSTagStep, updated by LemmatizeStep
        embeddings     — set by EmbedStep
        metadata       — free-form dict for extra data between steps

    Example:
        >>> ann = Annotation("Кот бяжыць!")
        >>> ann.text        # -> "Кот бяжыць!"  (processed_text is None → raw_text)
    """

    raw_text:       str
    processed_text: str | None                = None
    tokens:         list[str] | None          = None
    morph_tokens:   list["MorphToken"] | None = None
    embeddings:     "EmbeddingResult" | None  = None
    metadata:       dict                      = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Processed text if available, otherwise raw_text."""
        return self.processed_text if self.processed_text is not None else self.raw_text

    def is_tokenized(self) -> bool:
        return self.tokens is not None

    def is_pos_tagged(self) -> bool:
        return self.morph_tokens is not None and any(t.pos is not None for t in self.morph_tokens)

    def is_lemmatized(self) -> bool:
        return self.morph_tokens is not None and any(t.lemma is not None for t in self.morph_tokens)

    def is_embedded(self) -> bool:
        return self.embeddings is not None

    def __repr__(self) -> str:
        flags = []
        if self.processed_text is not None: flags.append("preprocessed")
        if self.tokens is not None:         flags.append(f"{len(self.tokens)} tokens")
        if self.morph_tokens is not None:   flags.append(f"{len(self.morph_tokens)} morph")
        if self.embeddings is not None:     flags.append("embedded")
        return f"Annotation({', '.join(flags) or 'raw'!r}, text={self.text[:40]!r})"
