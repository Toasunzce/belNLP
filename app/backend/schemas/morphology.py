from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    text: str


class TokenResult(BaseModel):
    text: str
    pos: str | None = None
    lemma: str | None = None


class AnalyzeResponse(BaseModel):
    tokens: list[TokenResult]


class LemmatizeRequest(BaseModel):
    word: str
    pos: str = "NOUN"


class LemmatizeResponse(BaseModel):
    word: str
    lemma: str
    pos: str