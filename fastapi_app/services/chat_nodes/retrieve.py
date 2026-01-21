from typing import Dict

from services.chat_nodes.config import RETRIEVE_K
from services.chat_nodes.intent import augment_query_for_date, is_date_query
from services.chat_nodes.mode import detect_mode, llm_detect_mode
from services.chat_nodes.state import GraphState, slim_retrievals
from services.retriever import retrieve
from utils.geo import append_node_trace_result


async def retrieve_node(state: GraphState) -> Dict:
    """Retrieve candidates using embeddings/BM25 and optional location constraints."""
    query = state.get("normalized_query") or state.get("query", "")
    # Date-oriented requests get additional intent keywords for better recall.
    query_for_retrieve = augment_query_for_date(query) if is_date_query(query) else query
    # Mode detection chooses which index to query (restaurant/cafe/tourspot).
    mode_raw = detect_mode(state.get("mode"), query)
    if mode_raw == "unknown":
        mode_raw = await llm_detect_mode(query, callbacks=state.get("callbacks"))
    mode_unknown = mode_raw == "unknown"
    mode_used = "tourspot" if mode_unknown else mode_raw
    # 이전 방문 기록 아직 없음. 어떻게 뽑아올지 정해야함(기존 추천시스템에서 방문한 이력들 DB에서 뽑아오도록 해야할듯?)
    history_ids = state.get("history_place_ids") or []

    debug_flag = bool(state.get("debug"))
    # Use location constraints if an anchor/admin term is available.
    anchor = state.get("anchor") or {}
    admin_term = state.get("admin_term")
    centers = anchor.get("centers") or []
    radius_by_intent = anchor.get("radius_by_intent") or {}
    radius_km = float(radius_by_intent.get(mode_used, 2.0)) if centers else None
    hits = retrieve(
        query=query_for_retrieve,
        mode=mode_used,
        top_k=RETRIEVE_K,
        history_place_ids=history_ids,
        debug=debug_flag,
        anchor_centers=centers or None,
        anchor_radius_km=radius_km,
        admin_term=admin_term,
    )
    result = {
        "retrievals": hits,
        "mode": mode_used,
        "mode_detected": mode_raw,
        "mode_unknown": mode_unknown,
        "last_radius_km": radius_km,
    }
    append_node_trace_result(
        state.get("query", ""),
        "retrieve",
        {**result, "retrievals": slim_retrievals(result["retrievals"])},
    )
    return result
