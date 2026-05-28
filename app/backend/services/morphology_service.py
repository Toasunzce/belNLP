from __future__ import annotations

from services.base import BaseService
from core.model_registry import registry
from schemas.morphology import (
    AnalyzeRequest, AnalyzeResponse, TokenResult,
    LemmatizeResponse,
)


class MorphologyService(BaseService):
    """
    Facade over POSTagger and CharSeq2SeqLemmatizer.
    Orchestrates the two-step annotation pipeline:
    tokenize -> POS tag -> lemmatize.
    """

    def ready(self) -> bool:
        return registry._initialized

    def analyze(self, req: AnalyzeRequest) -> AnalyzeResponse:
        tokens = req.text.strip().split()
        if not tokens:
            return AnalyzeResponse(tokens=[])

        morph = registry.pos_tagger.annotate(tokens)
        morph = registry.lemmatizer.annotate(morph)

        return AnalyzeResponse(tokens=[
            TokenResult(
                text=t.text,
                pos=t.pos,
                lemma=t.lemma,
            )
            for t in morph
        ])

    def lemmatize(self, word: str, pos: str) -> LemmatizeResponse:
        lemma = registry.lemmatizer.lemmatize(word, pos)
        return LemmatizeResponse(word=word, lemma=lemma, pos=pos)