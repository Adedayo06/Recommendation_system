"""
FastAPI application for the recommendation service.

Loads the trained model once on startup and exposes it over HTTP. Run from the
project root with:

    uvicorn api.main:app --reload

Interactive docs (Swagger) are then at http://127.0.0.1:8000/docs.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .routers import recommendations
from .services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model before the server starts accepting requests, so the first
    # request isn't the one paying the ~seconds of load time.
    model_service.load()
    yield
    # nothing to tear down — the model is just in-memory


app = FastAPI(
    title="E-commerce Recommendation API",
    version="0.1.0",
    description="Item-based collaborative filtering with a popularity fallback.",
    lifespan=lifespan,
)

app.include_router(recommendations.router)


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "recommendation-api",
        "docs": "/docs",
        "endpoints": ["/health", "/recommendations", "/items/{item_id}/similar"],
    }


@app.get("/health", tags=["meta"])
def health():
    """Readiness probe — reports whether the model is loaded and how big its
    catalog is. A deploy/orchestrator should wait for `ready: true`."""
    return {
        "status": "ok" if model_service.ready else "loading",
        "ready": model_service.ready,
        "catalog_size": model_service.catalog_size,
        "model_loaded_at": model_service.loaded_at,
        "model_load_seconds": (
            round(model_service.load_seconds, 2) if model_service.load_seconds else None
        ),
    }
