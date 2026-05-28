from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.model_registry import registry
from api import embeddings, morphology


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.init()
    yield


app = FastAPI(title="belNLP API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(embeddings.router, prefix="/api/embeddings", tags=["embeddings"])
app.include_router(morphology.router, prefix="/api/morphology", tags=["morphology"])


@app.get("/api/health")
def health():
    return {"status": "ok", "models_ready": registry._initialized}