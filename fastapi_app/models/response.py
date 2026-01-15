# fastapi_app/models/response.py
from typing import List
from pydantic import BaseModel, Field


class PoiItem(BaseModel):
    """API 응답에 포함될 개별 POI 아이템의 스키마"""

    place_id: int
    name: str = Field(..., description="장소 이름")
    category: str = Field(..., description="장소 카테고리 (tourspot, cafe, restaurant)")
    address: str | None = Field(None, description="주소")
    rating: float | None = Field(None, description="평점")
    review_count: int | None = Field(None, description="리뷰 수")

    class Config:
        # Pydantic V2, from_attributes=True is the new name for orm_mode=True
        from_attributes = True


class RecommendationCategory(BaseModel):
    """카테고리별 추천 목록"""

    category: str
    items: List[PoiItem]


class RecommendationResponse(BaseModel):
    """최종 /recommend API 응답 스키마"""

    recommendations: List[RecommendationCategory]
