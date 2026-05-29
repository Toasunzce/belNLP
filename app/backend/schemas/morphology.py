from pydantic import BaseModel, field_validator


class AnalyzeRequest(BaseModel):
    text: str

    @field_validator("text", mode="before")
    @classmethod
    def lowercase_text(cls, v: str) -> str:
        return v.lower()


class TokenResult(BaseModel):
    text: str
    pos: str | None = None
    lemma: str | None = None


class AnalyzeResponse(BaseModel):
    tokens: list[TokenResult]


class LemmatizeRequest(BaseModel):
    word: str
    pos: str = "NOUN"

    @field_validator("word", mode="before")
    @classmethod
    def lowercase_word(cls, v: str) -> str:
        return v.lower()


class LemmatizeResponse(BaseModel):
    word: str
    lemma: str
    pos: str