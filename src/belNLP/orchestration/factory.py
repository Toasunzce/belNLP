from __future__ import annotations

from belNLP.orchestration.pipeline import Pipeline, PipelineBuilder
from belNLP.tokenization.tokenizers import WordTokenizer
from belNLP.tokenization.preprocessors import LowercasePreprocessor
from belNLP.tokenization.filters import PunctuationFilter


class PipelineFactory:
    """Creates pre-configured pipelines for common tasks.

    Presets:
        "morphological" — preprocess → tokenize → POS tag → filter punct → lemmatize
        "embed"         — preprocess → tokenize → filter punct → embed
        "full"          — preprocess → tokenize → POS tag → filter punct → lemmatize → embed

    Example:
        >>> pipeline = PipelineFactory.create("morphological", tagger=tagger, lemmatizer=lemmatizer)
        >>> ann = pipeline.run("Кот бяжыць хутка!")
        >>> [(t.text, t.pos, t.lemma) for t in ann.morph_tokens]

        >>> pipeline = PipelineFactory.morphological(tagger, lemmatizer)  # same thing
    """

    @classmethod
    def create(cls, preset: str, **components) -> Pipeline:
        """Create a pipeline by preset name. Pass model components as kwargs."""
        dispatch = {
            "morphological": cls.morphological,
            "embed":         cls.embedding,
            "full":          cls.full,
        }
        if preset not in dispatch:
            raise ValueError(f"Unknown preset {preset!r}. Available: {sorted(dispatch)}")
        return dispatch[preset](**components)

    @classmethod
    def morphological(
        cls, tagger, lemmatizer, *,
        tokenizer=None, preprocessor=None, punct_filter=None,
    ) -> Pipeline:
        """preprocess → tokenize → POS tag (all tokens) → filter punct → lemmatize."""
        return (
            PipelineBuilder()
            .add_preprocessor(preprocessor or LowercasePreprocessor())
            .add_tokenizer(tokenizer or WordTokenizer())
            .add_pos_tagger(tagger)
            .add_morph_filter(punct_filter or PunctuationFilter())
            .add_lemmatizer(lemmatizer)
            .build()
        )

    @classmethod
    def embedding(
        cls, embedder, *,
        tokenizer=None, preprocessor=None, punct_filter=None,
    ) -> Pipeline:
        """preprocess → tokenize → filter punct → embed."""
        return (
            PipelineBuilder()
            .add_preprocessor(preprocessor or LowercasePreprocessor())
            .add_tokenizer(tokenizer or WordTokenizer())
            .add_filter(punct_filter or PunctuationFilter())
            .add_embedder(embedder)
            .build()
        )

    @classmethod
    def full(
        cls, tagger, lemmatizer, embedder, *,
        tokenizer=None, preprocessor=None, punct_filter=None,
    ) -> Pipeline:
        """preprocess → tokenize → POS tag → filter punct → lemmatize → embed."""
        return (
            PipelineBuilder()
            .add_preprocessor(preprocessor or LowercasePreprocessor())
            .add_tokenizer(tokenizer or WordTokenizer())
            .add_pos_tagger(tagger)
            .add_morph_filter(punct_filter or PunctuationFilter())
            .add_lemmatizer(lemmatizer)
            .add_embedder(embedder)
            .build()
        )
