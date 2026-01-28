from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class HistoryPlace(BaseModel):
    place_id: int = Field(alias="placeId")
    category: str
    province: str


class AnchorRef(BaseModel):
    centers: List[List[float]]
    source: Optional[str] = None
    radius_by_intent: Dict[str, float] = Field(default_factory=dict)


class PlaceRef(BaseModel):
    place: Optional[str] = None
    area: Optional[str] = None
    point: Optional[str] = None


class ChatContext(BaseModel):
    last_anchor: Optional[AnchorRef] = None
    last_radius_km: Optional[float] = None
    last_mode: Optional[str] = None
    last_query: Optional[str] = None
    last_resolved_name: Optional[str] = None
    last_place: Optional[PlaceRef] = None
    last_filter_applied: Optional[bool] = None


class ChatRequest(BaseModel):
    query: str
    messages: List[ChatMessage] = Field(default_factory=list)
    history_places: List[HistoryPlace] = Field(default_factory=list, alias="historyPlaces")
    top_k: Optional[int] = None
    debug: Optional[bool] = False
    context: Optional[ChatContext] = None

    preferred_themes: Optional[List[str]] = None
    preferred_moods: Optional[List[str]] = None
    preferred_restaurant_types: Optional[List[str]] = None
    preferred_cafe_types: Optional[List[str]] = None
    avoid: Optional[List[str]] = None
    activity_level: Optional[str] = None
