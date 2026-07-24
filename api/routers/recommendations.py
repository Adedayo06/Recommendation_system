"""Recommendation endpoints."""

from fastapi import APIRouter, HTTPException, Path, Query

from ..schemas.recommendations import (
    RecommendationRequest,
    RecommendationResponse,
    SimilarItemsResponse,
)
from ..services.model_service import model_service

router = APIRouter(tags=["recommendations"])


def _require_ready():
    if not model_service.ready:
        raise HTTPException(status_code=503, detail="model is not loaded yet")


@router.post("/recommendations", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    """
    Recommend items for a customer given their recent activity.

    Pass the customer's history (oldest-first, or with timestamps). A warm
    customer gets personalized item-CF results; a brand-new visitor (empty
    history) gets a popularity fallback, scoped to `category` if given.
    """
    _require_ready()
    return model_service.recommend(request.history, n=request.n, category=request.category)


@router.get("/items/{item_id}/similar", response_model=SimilarItemsResponse)
def similar_items(
    item_id: int = Path(description="an item the model knows about"),
    n: int = Query(default=10, ge=1, le=100),
):
    """
    "Customers who viewed this also viewed…" — items most similar to a given
    item, straight from the precomputed similarity matrix. Needs no customer
    history, so it's ideal for a product page. 404 if the item is unknown to
    the model (too new or too sparse to have neighbours).
    """
    _require_ready()
    if not model_service.is_known_item(item_id):
        raise HTTPException(
            status_code=404,
            detail=f"item {item_id} is not in the model's catalog",
        )
    return model_service.similar_items(item_id, n=n)
