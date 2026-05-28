from fastapi import APIRouter, HTTPException, Depends

from services.morphology_service import MorphologyService
from schemas.morphology import (
    AnalyzeRequest, AnalyzeResponse,
    LemmatizeRequest, LemmatizeResponse,
)

router = APIRouter()
_service = MorphologyService()


def get_service() -> MorphologyService:
    if not _service.ready():
        raise HTTPException(503, "Models not loaded yet")
    return _service


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest, svc: MorphologyService = Depends(get_service)):
    return svc.analyze(req)


@router.post("/lemmatize", response_model=LemmatizeResponse)
def lemmatize(req: LemmatizeRequest, svc: MorphologyService = Depends(get_service)):
    return svc.lemmatize(req.word, req.pos)