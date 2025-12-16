# models/user_input.py
from pydantic import BaseModel
from typing import List, Optional


class UserInput(BaseModel):
    user_id: Optional[str] = None

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

    # 전에 방문했던 장소
    visit_cafe: List[str] = None
    visit_rest: List[str] = None
    visit_attraction: List[str] = None

    # 클릭했던 곳

    # 나왔지만 클릭안했던 곳 

    # 지금 마지막으로 선택된 곳 위치
    last_selected_poi: Optional[str] = None