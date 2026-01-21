import json
from typing import Dict

from services.chat_nodes.callbacks import build_callbacks_config
from services.chat_nodes.llm_clients import detect_llm


async def llm_extract_place(query: str, callbacks: list | None = None) -> Dict | None:
    """LLM으로 장소 후보 키워드를 최대한 너그럽게 추출한다."""
    config = build_callbacks_config(callbacks)
    messages = [
        (
            "system",
            "너는 사용자의 의도에서 '지리적 위치(지명)'만 추출하는 전문가야.\n"
            "다음 JSON 형식으로만 답하라: {\"area\": \"...\", \"point\": \"...\"}\n"
            "규칙:\n"
            "1. '음식 메뉴(김밥, 파스타, 떡볶이 등)'나 '장소의 종류(맛집, 카페, 놀거리)'는 절대 지명으로 추출하지 마라.\n"
            "2. area는 행정구역/지역명(예: 강남구, 신림동, 여의도), point는 구체 지점(역/대학교/아파트/빌딩/랜드마크/몰)로 분리하라.\n"
            "3. 둘 다 있으면 area와 point 모두 채워라. point가 없다면 point는 null로 두어라.\n"
            "4. 오타가 있더라도 문맥상 '지역/지점'이면 추출하되(예: 걍남 -> 걍남), 메뉴 이름은 무조건 배제하라.\n"
            "5. 지명이 없으면 반드시 {\"area\": null, \"point\": null}을 반환하라."
        ),
        ("user", f"입력 문장: {query}\n추출 결과: "),
    ]
    try:
        resp = await detect_llm.ainvoke(messages, max_tokens=40, config=config)
        raw = (resp.content or "").strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        area = data.get("area")
        point = data.get("point")
        area_val = area.strip() if isinstance(area, str) and area.strip() else None
        point_val = point.strip() if isinstance(point, str) and point.strip() else None
        if not area_val and not point_val:
            return None
        return {"area": area_val, "point": point_val}
    except Exception:
        return None


async def llm_correct_place(
    query: str,
    area: str | None,
    point: str | None,
    callbacks: list | None = None,
) -> Dict | None:
    config = build_callbacks_config(callbacks)
    messages = [
        (
            "system",
            "너는 지명 오타 보정 전문가야.\n"
            "입력된 area/point에서 오타로 보이는 부분만 표준 지명으로 고쳐라.\n"
            "오타가 아니면 원문을 그대로 유지하고, 변경이 없으면 changed=false로 표시하라.\n"
            "JSON 형식만 반환: {\"area\": \"...\", \"point\": \"...\", \"changed\": true/false}",
        ),
        (
            "user",
            f"문장: {query}\narea: {area}\npoint: {point}\n보정 결과:",
        ),
    ]
    try:
        resp = await detect_llm.ainvoke(messages, max_tokens=60, config=config)
        raw = (resp.content or "").strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        changed = bool(data.get("changed"))
        area_val = data.get("area")
        point_val = data.get("point")
        area_out = area_val.strip() if isinstance(area_val, str) and area_val.strip() else None
        point_out = point_val.strip() if isinstance(point_val, str) and point_val.strip() else None
        return {"area": area_out, "point": point_out, "changed": changed}
    except Exception:
        return None
