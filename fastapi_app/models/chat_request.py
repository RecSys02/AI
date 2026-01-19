from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class HistoryPlace(BaseModel):
    place_id: int = Field(alias="placeId")
    category: str
    province: str


class ChatRequest(BaseModel):
    user_id: int = Field(alias="userId")
    query: str
    messages: List[ChatMessage] = Field(default_factory=list)
    history_places: List[HistoryPlace] = Field(default_factory=list, alias="historyPlaces")
    top_k: Optional[int] = None
    debug: Optional[bool] = False

    preferred_themes: Optional[List[str]] = None
    preferred_moods: Optional[List[str]] = None
    preferred_restaurant_types: Optional[List[str]] = None
    preferred_cafe_types: Optional[List[str]] = None
    avoid: List[str] = Field(default_factory=list)
    activity_level: Optional[str] = None
