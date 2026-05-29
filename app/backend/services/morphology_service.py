from __future__ import annotations

from services.base import BaseService
from core.model_registry import registry
from schemas.morphology import (
    AnalyzeRequest, AnalyzeResponse, TokenResult,
    LemmatizeResponse,
)
from belNLP.tokenization.tokenizers import WordTokenizer
from belNLP.tokenization.preprocessors import LowercasePreprocessor
from belNLP.tokenization.filters import PunctuationFilter

# Compiled once at module level
_tokenizer  = WordTokenizer()
_lowercase  = LowercasePreprocessor()
_punct      = PunctuationFilter()


def _is_symbol(token: str) -> bool:
    """True if the token is pure punctuation / symbol (filtered out by PunctuationFilter)."""
    return not _punct.filter([token])


class MorphologyService(BaseService):
    """
    Facade over POSTagger and CharSeq2SeqLemmatizer.
    Orchestrates the two-step annotation pipeline:
    preprocess -> tokenize -> POS tag -> lemmatize.

    Input text is lowercased automatically.
    WordTokenizer separates punctuation from words so that
    symbols participate in model predictions (context) but
    are excluded from the returned token list.
    """

    def ready(self) -> bool:
        return registry._initialized

    def analyze(self, req: AnalyzeRequest) -> AnalyzeResponse:
        # 1. Lowercase + basic normalisation
        text = _lowercase.process(req.text.strip())
        # 2. Tokenize: separates words from punctuation/symbols
        tokens = _tokenizer.tokenize(text)
        if not tokens:
            return AnalyzeResponse(tokens=[])

        # 3. POS-tag ALL tokens (symbols give sentence-level context to the BiLSTM)
        morph = registry.pos_tagger.annotate(tokens)

        # 4. Lemmatize only word tokens — the lemmatizer is per-token and
        #    context-free, so passing symbols is wasteful and can confuse it.
        word_tokens = [t for t in morph if not _is_symbol(t.text)]
        word_tokens = registry.lemmatizer.annotate(word_tokens)

        # 5. Fallback: if the model returned an empty lemma, keep the word itself
        result = []
        for t in word_tokens:
            lemma = t.lemma if t.lemma else t.text
            result.append(TokenResult(text=t.text, pos=t.pos, lemma=lemma))

        return AnalyzeResponse(tokens=result)

    def lemmatize(self, word: str, pos: str) -> LemmatizeResponse:
        word = _lowercase.process(word.strip())
        lemma = registry.lemmatizer.lemmatize(word, pos) or word
        return LemmatizeResponse(word=word, lemma=lemma, pos=pos)