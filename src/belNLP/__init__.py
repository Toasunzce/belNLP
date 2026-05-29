"""
belNLP
~~~~~~
NLP library for Belarusian text — tokenization, morphology, embeddings, pipelines.

Quick start:
    from belNLP import Pipeline, PipelineBuilder, PipelineFactory, Annotation

    # Use a preset pipeline
    pipeline = PipelineFactory.morphological(tagger, lemmatizer)
    ann = pipeline.run("Кот бяжыць хутка!")
    for t in ann.morph_tokens:
        print(t.text, t.pos, t.lemma)

    # Build a custom pipeline
    pipeline = (
        PipelineBuilder()
        .add_preprocessor(LowercasePreprocessor())
        .add_tokenizer(WordTokenizer())
        .add_pos_tagger(tagger)
        .add_lemmatizer(lemmatizer)
        .add_embedder(embedder)
        .build()
    )

Submodules:
    belNLP.tokenization  — tokenizers, preprocessors, filters, vocabulary
    belNLP.morphology    — POSTagger, Lemmatizer, MorphToken
    belNLP.embeddings    — GensimAdapter, analytics (cosine, analogy, PCA)
    belNLP.orchestration — Pipeline, PipelineBuilder, PipelineFactory, Annotation
"""

from belNLP.orchestration import (
    Annotation,
    Pipeline,
    PipelineBuilder,
    PipelineFactory,
)

__all__ = [
    "Annotation",
    "Pipeline",
    "PipelineBuilder",
    "PipelineFactory",
]
