import re
from typing import Dict

from services.chat_nodes.intent import is_expand_query
from services.chat_nodes.state import GraphState
from utils.geo import append_node_trace_result, normalize_text

CAFE_TERMS = ("카페", "커피", "디저트", "브런치", "빵", "라떼", "tea", "티룸")
RESTAURANT_TERMS = (
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
)


def _detect_explicit_mode(query: str) -> str | None:
    q = (query or "").lower()
    if any(term in q for term in CAFE_TERMS):
        return "cafe"
    if any(term in q for term in RESTAURANT_TERMS):
        return "restaurant"
    return None


def _has_place(place: dict | None) -> bool:
    if not isinstance(place, dict):
        return False
    return any(str(place.get(key) or "").strip() for key in ("area", "point", "place"))


def _context_place(context: dict) -> dict | None:
    if not isinstance(context, dict):
        return None
    last_place = context.get("last_place") or {}
    if _has_place(last_place):
        return dict(last_place)
    resolved_name = str(context.get("last_resolved_name") or "").strip()
    if resolved_name:
        return {"place": resolved_name}
    return None


def _is_short_category_followup(query: str, explicit_mode: str | None) -> bool:
    if not explicit_mode:
        return False
    compact = re.sub(r"\s+", "", re.sub(r"[?!.,]+$", "", str(query or "")))
    if len(compact) > 8:
        return False
    words = re.findall(r"[0-9A-Za-z가-힣]+", str(query or ""))
    return len(words) <= 2


async def context_manager_node(state: GraphState) -> Dict:
    """Decide whether to flush, keep, or clarify the location context."""
    query = str(state.get("query") or "").strip()
    context = state.get("context") or {}
    extracted_place = state.get("place") or {}
    place_original = state.get("place_original") or {}
    anaphora_detected = bool(state.get("anaphora_detected"))
    place_confidence = float(state.get("place_confidence") or 0.0)
    explicit_mode = _detect_explicit_mode(query)
    context_place = _context_place(context)
    expand_request = is_expand_query(query)

    needs_clarification = False
    clarification_reason = None
    context_action = "none"
    active_place = None
    resolved_name = None

    if not normalize_text(query):
        needs_clarification = True
        clarification_reason = "empty_query"
    elif _has_place(extracted_place):
        context_action = "flush"
        active_place = dict(extracted_place)
    elif anaphora_detected or expand_request:
        if context_place:
            context_action = "keep"
            active_place = context_place
            resolved_name = context.get("last_resolved_name")
            if not place_confidence:
                place_confidence = 0.9
        else:
            needs_clarification = True
            clarification_reason = "missing_reference"
    elif context_place and _is_short_category_followup(query, explicit_mode):
        context_action = "keep"
        active_place = context_place
        resolved_name = context.get("last_resolved_name")
        if not place_confidence:
            place_confidence = 0.75
    else:
        needs_clarification = True
        clarification_reason = "missing_place"

    result = {
        "explicit_mode": explicit_mode,
        "context_action": context_action,
        "place": active_place,
        "place_original": place_original or None,
        "resolved_name": resolved_name,
        "needs_clarification": needs_clarification,
        "clarification_reason": clarification_reason,
        "place_confidence": place_confidence,
        "has_explicit_place": bool(_has_place(extracted_place)),
    }
    append_node_trace_result(query, "context_manager", result)
    return result
