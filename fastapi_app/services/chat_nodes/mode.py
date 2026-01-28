from services.chat_nodes.callbacks import build_callbacks_config
from services.chat_nodes.llm_clients import detect_llm, max_tokens_kwargs


def detect_mode(mode_hint: str | None, query: str) -> str:
    # 직접적으로 모드가 주어지면 우선 사용
    if mode_hint in {"tourspot", "cafe", "restaurant"}:
        return mode_hint
    # 힌트가 없으면 쿼리 기반으로 추론
    q = query.lower()
    cafe_terms = ["카페", "커피", "디저트", "브런치", "빵", "라떼", "tea", "티룸"]
    restaurant_terms = [
        "맛집",
        "식당",
        "레스토랑",
        "밥",
        "점심",
        "저녁",
        "고기",
        "파스타",
        "스테이크",
        "스시",
        "초밥",
        "회",
        "중식",
        "한식",
        "양식",
        "분식",
        "라멘",
        "라면",
        "피자",
        "버거",
        "삼겹살",
        "술집",
        "포차",
        "안주",
        "뷔페",
    ]
    # 기본값은 unknown, 카페/맛집 키워드가 있으면 해당 모드로 변경
    if any(t in q for t in cafe_terms):
        return "cafe"
    if any(t in q for t in restaurant_terms):
        return "restaurant"
    return "unknown"


async def llm_detect_mode(query: str, callbacks: list | None = None) -> str:
    """LLM으로 모드 분류 (tourspot/cafe/restaurant/unknown 중 하나만 반환)."""
    config = build_callbacks_config(callbacks)
    messages = [
        (
            "system",
            "다음 사용자 질문이 관광지(tourspot), 카페(cafe), 식당/맛집(restaurant) 중 어느 카테고리에 해당하는지 "
            "정확히 하나의 단어만 소문자로 답하라. 해당이 없으면 unknown만 답하라.",
        ),
        ("user", query),
    ]
    try:
        resp = await detect_llm.ainvoke(messages, **max_tokens_kwargs(5), config=config)
        mode = (resp.content or "").strip().lower()
        return mode if mode in {"tourspot", "cafe", "restaurant"} else "unknown"
    except Exception:
        return "unknown"
