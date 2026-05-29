from pydantic import BaseModel, field_validator


class EmbedRequest(BaseModel):
    words: list[str]
    reduction: str = "pca"        # "pca" | "tsne" | "umap"
    n_components: int = 3         # 2 or 3

    @field_validator("words", mode="before")
    @classmethod
    def lowercase_words(cls, v: list[str]) -> list[str]:
        return [w.lower() for w in v]


class EmbedPoint(BaseModel):
    word: str
    x: float
    y: float
    z: float


class EmbedResponse(BaseModel):
    points: list[EmbedPoint]
    dim: int


class SimilarityRequest(BaseModel):
    word_a: str
    word_b: str

    @field_validator("word_a", "word_b", mode="before")
    @classmethod
    def lowercase_words(cls, v: str) -> str:
        return v.lower()


class SimilarityResponse(BaseModel):
    cosine: float
    euclidean: float


class AnalogyRequest(BaseModel):
    a: str
    b: str
    c: str
    topn: int = 5

    @field_validator("a", "b", "c", mode="before")
    @classmethod
    def lowercase_words(cls, v: str) -> str:
        return v.lower()


class AnalogyResponse(BaseModel):
    results: list[str]


class NearestRequest(BaseModel):
    word: str
    topn: int = 10

    @field_validator("word", mode="before")
    @classmethod
    def lowercase_word(cls, v: str) -> str:
        return v.lower()


class NearestResponse(BaseModel):
    neighbours: list[str]