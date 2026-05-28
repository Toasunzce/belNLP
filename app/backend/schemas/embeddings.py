from pydantic import BaseModel


class EmbedRequest(BaseModel):
    words: list[str]
    reduction: str = "pca"        # "pca" | "tsne" | "umap"
    n_components: int = 3         # 2 or 3


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


class SimilarityResponse(BaseModel):
    cosine: float
    euclidean: float


class AnalogyRequest(BaseModel):
    a: str
    b: str
    c: str
    topn: int = 5


class AnalogyResponse(BaseModel):
    results: list[str]


class NearestRequest(BaseModel):
    word: str
    topn: int = 10


class NearestResponse(BaseModel):
    neighbours: list[str]