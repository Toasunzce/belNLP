from fastapi import APIRouter, HTTPException, Depends

from services.embedding_service import EmbeddingService
from schemas.embeddings import (
    EmbedRequest, EmbedResponse,
    SimilarityRequest, SimilarityResponse,
    AnalogyRequest, AnalogyResponse,
    NearestRequest, NearestResponse,
)

router = APIRouter()
_service = EmbeddingService()


def get_service() -> EmbeddingService:
    if not _service.ready():
        raise HTTPException(503, "Models not loaded yet")
    return _service


@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest, svc: EmbeddingService = Depends(get_service)):
    return svc.embed(req)


@router.post("/similarity", response_model=SimilarityResponse)
def similarity(req: SimilarityRequest, svc: EmbeddingService = Depends(get_service)):
    return svc.similarity(req.word_a, req.word_b)


@router.post("/analogy", response_model=AnalogyResponse)
def analogy(req: AnalogyRequest, svc: EmbeddingService = Depends(get_service)):
    return svc.analogy(req.a, req.b, req.c, req.topn)


@router.post("/nearest", response_model=NearestResponse)
def nearest(req: NearestRequest, svc: EmbeddingService = Depends(get_service)):
    return svc.nearest(req.word, req.topn)