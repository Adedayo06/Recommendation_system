"""Request/response schemas for the recommendation endpoints."""

from typing import List, Optional

from pydantic import BaseModel, Field


class Interaction(BaseModel):
    """One thing a customer did. Supply either an `event` (which maps to a
    weight the way the model was trained) or an explicit `weight`. Order your
    interactions oldest-first, or include `timestamp` and they'll be sorted."""

    item_id: int
    event: Optional[str] = Field(
        default=None, description="view | addtocart | transaction"
    )
    weight: Optional[float] = Field(
        default=None, description="explicit interaction strength; overrides event"
    )
    timestamp: Optional[int] = Field(
        default=None, description="epoch ms; used only to order the history"
    )

    model_config = {
        "json_schema_extra": {
            "example": {"item_id": 355908, "event": "view", "timestamp": 1433221332117}
        }
    }


class RecommendationRequest(BaseModel):
    history: List[Interaction] = Field(
        default_factory=list,
        description="the customer's recent activity; empty for a brand-new visitor",
    )
    n: int = Field(default=10, ge=1, le=100, description="how many items to return")
    category: Optional[int] = Field(
        default=None, description="optional category to scope a cold-start fallback"
    )


class RecommendedItem(BaseModel):
    item_id: int
    score: float
    source: str = Field(description="item_cf | popularity_category | popularity_global")
    category: Optional[int] = None


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendedItem]
    count: int
    strategy: str = Field(description="personalized | cold_start")
    known_history_items: int = Field(
        description="how many of the supplied history items the model actually knows"
    )


class SimilarItem(BaseModel):
    item_id: int
    similarity: float
    category: Optional[int] = None


class SimilarItemsResponse(BaseModel):
    item_id: int
    category: Optional[int] = None
    similar: List[SimilarItem]
    count: int
