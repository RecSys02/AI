# models/user_input.py
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from models.poi_ref import PoiRef


def to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class UserInput(BaseModel):
    # camelCase(프론트)와 snake_case(백엔드) 모두 허용
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    user_id: int = Field(alias="userId")

    # 회원가입 시 받은 정보
    preferred_themes: Optional[List[str]] = None
    preferred_moods: Optional[List[str]] = None
    preferred_restaurant_types: Optional[List[str]] = None
    preferred_cafe_types: Optional[List[str]] = None
    avoid: List[str] = Field(default_factory=list)
    activity_level: Optional[str] = None  # "거의 안걷기" | "적당히" | "오래 걸어도" | "오래 걷는것 선호"

    # 일정 생성 시 입력
    region: str
    companion: Optional[List[str]] = None
    budget: Optional[str] = None

    # ✅ 방문 이력 (object list)
    history_places: Optional[List[PoiRef]] = Field(default=None, alias="historyPlaces")

    # 마지막 선택된 장소들 (신호 강화)
    selected_places: Optional[List[PoiRef]] = Field(default=None, alias="selectedPlaces")
    