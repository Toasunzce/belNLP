from __future__ import annotations

from belNLP.orchestration.annotation import Annotation
from belNLP.orchestration.steps import (
    BasePipelineStep,
    PreprocessStep, TokenizeStep,
    FilterStep, FilterMorphStep,
    POSTagStep, LemmatizeStep, EmbedStep,
)
from belNLP.tokenization.base import BaseTokenizer, BasePreprocessor, BaseFilter
from belNLP.embeddings.base import BaseEmbedder


class Pipeline:
    """Executes a sequence of steps over an Annotation.

    Example:
        >>> result = pipeline.run("Я іду дадому")
        >>> result.tokens       # -> ["я", "іду", "дадому"]
        >>> result.morph_tokens # -> [MorphToken(...), ...]
        >>> result.embeddings   # -> EmbeddingResult(...)
    """

    def __init__(self, steps: list[BasePipelineStep]) -> None:
        if not steps:
            raise ValueError("Pipeline must contain at least one step.")
        self._steps = list(steps)

    def run(self, input: str | Annotation) -> Annotation:
        """Run the pipeline. Accepts a raw string or an existing Annotation."""
        annotation = Annotation(raw_text=input) if isinstance(input, str) else input
        for step in self._steps:
            annotation = step.process(annotation)
        return annotation

    def __call__(self, input: str | Annotation) -> Annotation:
        return self.run(input)

    @property
    def steps(self) -> list[BasePipelineStep]:
        return list(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return f"Pipeline[{' → '.join(s.name for s in self._steps)}]"


class PipelineBuilder:
    """Fluent builder for constructing Pipelines.

    Example:
        >>> pipeline = (
        ...     PipelineBuilder()
        ...     .add_preprocessor(LowercasePreprocessor())
        ...     .add_tokenizer(WordTokenizer())
        ...     .add_pos_tagger(tagger)
        ...     .add_lemmatizer(lemmatizer)
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        self._steps: list[BasePipelineStep] = []

    def add_preprocessor(self, preprocessor: BasePreprocessor) -> "PipelineBuilder":
        self._steps.append(PreprocessStep(preprocessor))
        return self

    def add_tokenizer(self, tokenizer: BaseTokenizer) -> "PipelineBuilder":
        self._steps.append(TokenizeStep(tokenizer))
        return self

    def add_filter(self, filter: BaseFilter) -> "PipelineBuilder":
        """Filter annotation.tokens."""
        self._steps.append(FilterStep(filter))
        return self

    def add_morph_filter(self, filter: BaseFilter) -> "PipelineBuilder":
        """Filter annotation.morph_tokens and sync tokens (use after POSTagStep)."""
        self._steps.append(FilterMorphStep(filter))
        return self

    def add_pos_tagger(self, tagger) -> "PipelineBuilder":
        self._steps.append(POSTagStep(tagger))
        return self

    def add_lemmatizer(self, lemmatizer) -> "PipelineBuilder":
        self._steps.append(LemmatizeStep(lemmatizer))
        return self

    def add_embedder(self, embedder: BaseEmbedder) -> "PipelineBuilder":
        self._steps.append(EmbedStep(embedder))
        return self

    def add_annotator(self, annotator) -> "PipelineBuilder":
        """Auto-detect POSTagger or Lemmatizer and add the correct step."""
        from belNLP.morphology.pos_tagger import POSTagger
        from belNLP.morphology.lemmatizer import Lemmatizer
        if isinstance(annotator, POSTagger):
            return self.add_pos_tagger(annotator)
        if isinstance(annotator, Lemmatizer):
            return self.add_lemmatizer(annotator)
        raise TypeError(
            f"Unknown annotator type: {type(annotator).__name__}. "
            "Use add_pos_tagger() or add_lemmatizer() explicitly."
        )

    def add_step(self, step: BasePipelineStep) -> "PipelineBuilder":
        """Add a custom step directly."""
        if not isinstance(step, BasePipelineStep):
            raise TypeError(f"Expected BasePipelineStep, got {type(step).__name__}")
        self._steps.append(step)
        return self

    def build(self) -> Pipeline:
        return Pipeline(list(self._steps))

    def __repr__(self) -> str:
        steps = " → ".join(s.name for s in self._steps)
        return f"PipelineBuilder[{steps or 'empty'}]"
