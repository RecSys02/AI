from typing import Dict

from services.chat_nodes.place_llm import llm_correct_place, llm_extract_place
from services.chat_nodes.state import GraphState
from utils.geo import append_place_debug, append_node_trace_result

ANAPHORA_HINTS = (
    "거기",
    "거긴",
    "거기서",
    "그근처",
    "그 근처",
    "그주변",
    "그 주변",
    "그쪽",
    "그곳",
)


def _has_anaphora(query: str) -> bool:
    compact = "".join(str(query or "").lower().split())
    normalized_hints = [hint.replace(" ", "") for hint in ANAPHORA_HINTS]
    return any(hint in compact for hint in normalized_hints)


def _place_confidence(place: dict | None, corrected: dict | None = None) -> float:
    if not isinstance(place, dict):
        return 0.0
    score = 0.0
    if str(place.get("point") or "").strip():
        score += 0.65
    if str(place.get("area") or "").strip():
        score += 0.2
    if corrected and corrected.get("changed"):
        score += 0.05
    if str(place.get("place") or "").strip():
        score += 0.1
    return min(score, 0.95)


async def extract_place_node(state: GraphState) -> Dict:
    """Extract a place candidate from the raw user query before any rewrite."""
    query = state.get("query", "")
    callbacks = state.get("callbacks")
    place_original = await llm_extract_place(query, callbacks=callbacks)
    corrected = None
    place = place_original
    if place_original:
        corrected = await llm_correct_place(
            query,
            place_original.get("area"),
            place_original.get("point"),
            callbacks=callbacks,
        )
        if corrected and (corrected.get("area") or corrected.get("point")):
            place = {
                "area": corrected.get("area"),
                "point": corrected.get("point"),
            }
    place = place or None
    confidence = _place_confidence(place, corrected=corrected)
    anaphora_detected = _has_anaphora(query)
    append_place_debug(
        {
            "query": query,
            "place_original": place_original,
            "place": place,
            "corrected": corrected,
            "anaphora_detected": anaphora_detected,
            "place_confidence": confidence,
        }
    )
    result = {
        "place": place,
        "place_original": place_original,
        "anaphora_detected": anaphora_detected,
        "place_confidence": confidence,
    }
    append_node_trace_result(query, "extract_place", result)
    return result
