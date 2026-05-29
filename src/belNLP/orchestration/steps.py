from __future__ import annotations

from abc import ABC, abstractmethod

from belNLP.orchestration.annotation import Annotation
from belNLP.tokenization.base import BaseTokenizer, BasePreprocessor, BaseFilter
from belNLP.embeddings.base import BaseEmbedder


class BasePipelineStep(ABC):
    """Base class for a single pipeline step."""

    @abstractmethod
    def process(self, annotation: Annotation) -> Annotation:
        pass

    def __call__(self, annotation: Annotation) -> Annotation:
        return self.process(annotation)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}()"


class PreprocessStep(BasePipelineStep):
    """Applies a preprocessor to annotation.text → stores result in annotation.processed_text.

    Example:
        >>> PreprocessStep(LowercasePreprocessor())
    """

    def __init__(self, preprocessor: BasePreprocessor) -> None:
        self._preprocessor = preprocessor

    def process(self, annotation: Annotation) -> Annotation:
        annotation.processed_text = self._preprocessor.process(annotation.text)
        return annotation

    def __repr__(self) -> str:
        return f"PreprocessStep({self._preprocessor.__class__.__name__})"


class TokenizeStep(BasePipelineStep):
    """Tokenizes annotation.text → fills annotation.tokens.

    Example:
        >>> TokenizeStep(WordTokenizer())
    """

    def __init__(self, tokenizer: BaseTokenizer) -> None:
        self._tokenizer = tokenizer

    def process(self, annotation: Annotation) -> Annotation:
        annotation.tokens = self._tokenizer.tokenize(annotation.text)
        return annotation

    def __repr__(self) -> str:
        return f"TokenizeStep({self._tokenizer.__class__.__name__})"


class FilterStep(BasePipelineStep):
    """Filters annotation.tokens in-place. Requires prior TokenizeStep.

    Example:
        >>> FilterStep(PunctuationFilter())
    """

    def __init__(self, filter: BaseFilter) -> None:
        self._filter = filter

    def process(self, annotation: Annotation) -> Annotation:
        if annotation.tokens is None:
            raise RuntimeError(f"{self.name}: annotation.tokens is None — add TokenizeStep first.")
        annotation.tokens = self._filter.filter(annotation.tokens)
        return annotation

    def __repr__(self) -> str:
        return f"FilterStep({self._filter.__class__.__name__})"


class FilterMorphStep(BasePipelineStep):
    """Filters annotation.morph_tokens and syncs annotation.tokens. Requires prior POSTagStep.

    Used to strip punctuation after POS tagging (so the tagger still got full context),
    before lemmatization. Also updates annotation.tokens to match the filtered list.

    Example:
        >>> FilterMorphStep(PunctuationFilter())
    """

    def __init__(self, filter: BaseFilter) -> None:
        self._filter = filter

    def process(self, annotation: Annotation) -> Annotation:
        if annotation.morph_tokens is None:
            raise RuntimeError(f"{self.name}: annotation.morph_tokens is None — add POSTagStep first.")
        annotation.morph_tokens = [
            t for t in annotation.morph_tokens if self._filter.filter([t.text])
        ]
        annotation.tokens = [t.text for t in annotation.morph_tokens]
        return annotation

    def __repr__(self) -> str:
        return f"FilterMorphStep({self._filter.__class__.__name__})"


class POSTagStep(BasePipelineStep):
    """Runs POS tagger on annotation.tokens → fills annotation.morph_tokens. Requires TokenizeStep.

    Passes ALL tokens to the tagger (including punctuation) for full BiLSTM context.

    Example:
        >>> POSTagStep(POSTagger.load("models/POSTagger.pt"))
    """

    def __init__(self, tagger) -> None:
        self._tagger = tagger

    def process(self, annotation: Annotation) -> Annotation:
        if annotation.tokens is None:
            raise RuntimeError(f"{self.name}: annotation.tokens is None — add TokenizeStep first.")
        annotation.morph_tokens = self._tagger.annotate(annotation.tokens)
        return annotation

    def __repr__(self) -> str:
        return f"POSTagStep({self._tagger.__class__.__name__})"


class LemmatizeStep(BasePipelineStep):
    """Runs lemmatizer on annotation.morph_tokens, filling .lemma on each token.

    Example:
        >>> LemmatizeStep(Lemmatizer.load("models/Lemmatizer.pt"))
    """

    def __init__(self, lemmatizer) -> None:
        self._lemmatizer = lemmatizer

    def process(self, annotation: Annotation) -> Annotation:
        if annotation.morph_tokens is None:
            raise RuntimeError(f"{self.name}: annotation.morph_tokens is None — add POSTagStep first.")
        annotation.morph_tokens = self._lemmatizer.annotate(annotation.morph_tokens)
        return annotation

    def __repr__(self) -> str:
        return f"LemmatizeStep({self._lemmatizer.__class__.__name__})"


class EmbedStep(BasePipelineStep):
    """Embeds annotation.tokens → fills annotation.embeddings. Requires TokenizeStep.

    Example:
        >>> EmbedStep(GensimAdapter(kv))
    """

    def __init__(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder

    def process(self, annotation: Annotation) -> Annotation:
        if annotation.tokens is None:
            raise RuntimeError(f"{self.name}: annotation.tokens is None — add TokenizeStep first.")
        if annotation.tokens:
            annotation.embeddings = self._embedder.embed(annotation.tokens)
        return annotation

    def __repr__(self) -> str:
        return f"EmbedStep({self._embedder.__class__.__name__})"
