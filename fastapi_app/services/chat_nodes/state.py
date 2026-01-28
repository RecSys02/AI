from typing import Dict, List, TypedDict


class GraphState(TypedDict):
    query: str
    normalized_query: str | None
    mode: str | None
    mode_detected: str | None
    mode_unknown: bool | None
    place: Dict | None
    place_original: Dict | None
    anchor: Dict | None
    anchor_failed: bool | None
    input_place: str | None
    resolved_name: str | None
    last_radius_km: float | None
    top_k: int | None
    history_place_ids: List[int]
    intent: str
    expand_request: bool | None
    expand_failed: bool | None
    retrievals: List[dict]
    final: str
    messages: List[dict]
    debug: bool | None
    context: Dict | None
    callbacks: List[object] | None


def slim_retrievals(items: List[dict]) -> List[dict]:
    slim = []
    for r in items:
        meta = r.get("meta") or {}
        slim.append(
            {
                "place_id": r.get("place_id"),
                "category": r.get("category"),
                "province": meta.get("province"),
                "name": meta.get("name") or meta.get("title"),
                "score": r.get("score"),
            }
        )
    return slim


def build_context(state: GraphState) -> Dict:
    anchor = state.get("anchor")
    return {
        "last_anchor": anchor or None,
        "last_radius_km": state.get("last_radius_km"),
        "last_mode": state.get("mode"),
        "last_query": state.get("query"),
        "last_resolved_name": state.get("resolved_name"),
        "last_place": state.get("place"),
        "last_filter_applied": bool(anchor),
    }
