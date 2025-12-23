# models/user_input.py
from pydantic import BaseModel
from typing import List, Optional
from models.poi_ref import PoiRef


class UserInput(BaseModel):
    user_id: int

    # 회원가입 시 받은 정보
    preferred_themes: List[str]
    preferred_moods: List[str]
    preferred_restaurant_types: List[str]
    preferred_cafe_types: List[str]
    avoid: List[str] = []
    activity_level: str  # "거의 안걷기" | "적당히" | "오래 걸어도" | "오래 걷는것 선호"

    # 일정 생성 시 입력
    city: str
    companion_type: List[str]
    budget: Optional[str] = None

    # ✅ 방문 이력 (object list)
    visit_cafe: Optional[List[PoiRef]] = None
    visit_restaurant: Optional[List[PoiRef]] = None
    visit_tourspot: Optional[List[PoiRef]] = None

    # 마지막 선택된 장소들 (신호 강화)
    last_selected_pois: Optional[List[PoiRef]] = None
