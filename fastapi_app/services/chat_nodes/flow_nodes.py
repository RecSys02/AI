import asyncio
import re
from typing import Dict, List

from services.chat_nodes.answer import answer_node
from services.chat_nodes.apply_location_filter import apply_location_filter_node
from services.chat_nodes.callbacks import build_callbacks_config
from services.chat_nodes.clarify_query import clarify_query_node
from services.chat_nodes.context_manager import context_manager_node
from services.chat_nodes.expand_radius import expand_radius_node
from services.chat_nodes.extract_place import extract_place_node
from services.chat_nodes.general_answer import general_answer_node
from services.chat_nodes.intent import detect_intent, is_expand_query, is_nearby_query
from services.chat_nodes.llm_clients import LLMRateLimitError, llm, rate_limit_fallback_text
from services.chat_nodes.message_utils import normalize_messages
from services.chat_nodes.mode import detect_mode, llm_detect_mode
from services.chat_nodes.recommend_cache import build_recommend_cache_key, get_recommend_cache
from services.chat_nodes.resolve_anchor import resolve_anchor_node
from services.chat_nodes.retrieve import retrieve_node
from services.chat_nodes.rewrite_query import rewrite_query_node
from services.chat_nodes.state import GraphState, build_context, slim_retrievals
from services.retriever import retrieve
from utils.geo import append_node_trace_result

DETAIL_TERMS = (
    "어때",
    "어떤 곳",
    "어떤곳",
    "설명",
    "소개",
    "정보",
    "분위기",
    "주소",
    "위치",
    "영업",
    "운영",
    "시간",
    "주차",
    "가격",
    "메뉴",
    "후기",
    "리뷰",
    "평점",
)

FOLLOWUP_TERMS = (
    "다른",
    "말고",
    "또",
    "더",
    "비슷한",
    "그중",
    "거기",
    "그곳",
    "그 주변",
    "그 근처",
)

CATEGORY_HINTS = (
    "맛집",
    "식당",
    "레스토랑",
    "카페",
    "커피",
    "디저트",
    "관광지",
    "명소",
    "놀거리",
    "여행지",
    "코스",
    "추천",
)

MODE_ORDER = ("tourspot", "restaurant", "cafe")
NO_RESULT_LABELS = {
    "tourspot": "놀거리",
    "restaurant": "맛집",
    "cafe": "카페",
}


def _has_place(place: dict | None) -> bool:
    if not isinstance(place, dict):
        return False
    return any(str(place.get(key) or "").strip() for key in ("area", "point", "place"))


def _place_to_label(place: dict | None) -> str:
    if not isinstance(place, dict):
        return ""
    parts = [str(place.get(key) or "").strip() for key in ("area", "point", "place")]
    parts = [part for part in parts if part]
    return " ".join(dict.fromkeys(parts))


def _extract_names(items: List[dict]) -> List[str]:
    names: List[str] = []
    for item in items:
        meta = item.get("meta") or {}
        name = meta.get("name") or meta.get("title")
        if not name:
            continue
        text = str(name).strip()
        if text and text not in names:
            names.append(text)
    return names


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in str(text or "") for term in terms)


def _mentions_last_recommended(query: str, context: dict) -> bool:
    names = context.get("last_recommended_names") or []
    return any(name and str(name) in query for name in names)


def _has_recommend_context(context: dict) -> bool:
    return bool(
        (context.get("last_anchor") or {}).get("centers")
        or context.get("last_place")
        or context.get("last_recommended_names")
        or context.get("last_resolved_name")
    )


def _is_followup_recommend_query(state: GraphState, query: str) -> bool:
    context = state.get("context") or {}
    if not _has_recommend_context(context):
        return False
    if is_expand_query(query):
        return True
    if bool(state.get("anaphora_detected")) and detect_intent(query) == "recommend":
        return True
    if _mentions_last_recommended(query, context) and detect_intent(query) == "recommend":
        return True
    compact = re.sub(r"\s+", "", str(query or ""))
    if (
        detect_intent(query) == "recommend"
        and not _has_place(state.get("place"))
        and (len(compact) <= 18 or _contains_any(query, FOLLOWUP_TERMS))
        and _contains_any(query, CATEGORY_HINTS + FOLLOWUP_TERMS)
    ):
        return True
    return False


def _is_place_detail_query(state: GraphState, query: str) -> bool:
    if detect_intent(query) == "recommend":
        return False
    context = state.get("context") or {}
    if _contains_any(query, DETAIL_TERMS):
        if _has_place(state.get("place")):
            return True
        if bool(state.get("anaphora_detected")) and _has_recommend_context(context):
            return True
        if _mentions_last_recommended(query, context):
            return True
        if context.get("last_resolved_name"):
            return True
    return False


def _build_no_result_text(state: GraphState) -> str:
    anchor = state.get("anchor") or {}
    if anchor:
        location_name = state.get("resolved_name") or state.get("input_place") or "해당 지역"
        mode_used = state.get("mode") or "tourspot"
        category_label = NO_RESULT_LABELS.get(mode_used, "추천")
        return (
            f"{location_name} 근처에는 조건에 맞는 결과가 없어요. "
            f"반경을 넓혀서 다시 찾아볼까요, 아니면 {location_name}의 다른 {category_label}로 추천해드릴까요?"
        )
    return "조건에 맞는 추천 결과를 찾지 못했어요. 지역이나 카테고리를 조금 바꿔서 다시 말씀해 주세요."


def _build_place_detail_query(state: GraphState) -> str:
    raw_query = str(state.get("normalized_query") or state.get("query") or "").strip()
    if raw_query and not bool(state.get("anaphora_detected")):
        return raw_query
    place_label = _place_to_label(state.get("place"))
    if place_label:
        return f"{place_label} 정보"
    context = state.get("context") or {}
    names = context.get("last_recommended_names") or []
    if names:
        return f"{names[0]} 정보"
    resolved = str(context.get("last_resolved_name") or "").strip()
    if resolved:
        return f"{resolved} 정보"
    return raw_query


def _detail_sort_key(query: str, hit: dict, mentioned_names: list[str]) -> tuple[float, float]:
    meta = hit.get("meta") or {}
    name = str(meta.get("name") or meta.get("title") or "").strip()
    score = float(hit.get("score") or 0.0)
    bonus = 0.0
    if name and name in query:
        bonus += 3.0
    if name and name in mentioned_names:
        bonus += 2.0
    if name and query.startswith(name):
        bonus += 1.0
    return (bonus, score)


async def query_understanding_node(state: GraphState) -> Dict:
    result = await extract_place_node(state)
    append_node_trace_result(
        state.get("query", ""),
        "query_understanding",
        {
            "place": result.get("place"),
            "place_confidence": result.get("place_confidence"),
            "anaphora_detected": result.get("anaphora_detected"),
        },
    )
    return result


async def route_intent_node(state: GraphState) -> Dict:
    query = str(state.get("query") or "").strip()
    if not query:
        result = {"intent": "general", "route_intent": "general_chat", "expand_request": False}
        append_node_trace_result(query, "route_intent", result)
        return result

    base_intent = detect_intent(query)
    if _is_place_detail_query(state, query):
        route_intent = "place_detail"
    elif base_intent == "recommend" and _is_followup_recommend_query(state, query):
        route_intent = "followup_recommend"
    elif base_intent == "recommend":
        route_intent = "recommendation"
    else:
        route_intent = "general_chat"

    result = {
        "intent": base_intent,
        "route_intent": route_intent,
        "expand_request": is_expand_query(query),
    }
    append_node_trace_result(query, "route_intent", result)
    return result


async def extract_constraints_node(state: GraphState) -> Dict:
    result = await context_manager_node(state)
    append_node_trace_result(
        state.get("query", ""),
        "extract_constraints",
        {
            "place": result.get("place"),
            "context_action": result.get("context_action"),
            "explicit_mode": result.get("explicit_mode"),
            "needs_clarification": result.get("needs_clarification"),
            "clarification_reason": result.get("clarification_reason"),
        },
    )
    return result


async def merge_with_previous_context_node(state: GraphState) -> Dict:
    merged = await context_manager_node(state)
    combined_state = {**state, **merged}
    if bool(state.get("expand_request")):
        expanded = await expand_radius_node({**combined_state, "expand_request": True})
        merged.update(expanded)
        if expanded.get("expand_failed"):
            merged["needs_clarification"] = True
            merged["clarification_reason"] = "missing_reference"
    append_node_trace_result(
        state.get("query", ""),
        "merge_with_previous_context",
        {
            "place": merged.get("place"),
            "resolved_name": merged.get("resolved_name"),
            "anchor": merged.get("anchor"),
            "needs_clarification": merged.get("needs_clarification"),
            "clarification_reason": merged.get("clarification_reason"),
        },
    )
    return merged


async def normalize_constraints_node(state: GraphState) -> Dict:
    updates: Dict = {}
    rewrite = await rewrite_query_node(state)
    updates.update(rewrite)
    combined_state = {**state, **updates}
    if not combined_state.get("anchor") and _has_place(combined_state.get("place")):
        updates.update(await resolve_anchor_node(combined_state))
    append_node_trace_result(
        state.get("query", ""),
        "normalize_constraints",
        {
            "normalized_query": updates.get("normalized_query"),
            "resolved_name": updates.get("resolved_name"),
            "anchor": updates.get("anchor"),
            "anchor_failed": updates.get("anchor_failed"),
        },
    )
    return updates


async def check_required_constraints_node(state: GraphState) -> Dict:
    query = str(state.get("normalized_query") or state.get("query") or "").strip()
    clarification_reason = state.get("clarification_reason")
    needs_followup = bool(state.get("needs_clarification"))
    if state.get("expand_failed"):
        needs_followup = True
        clarification_reason = "missing_reference"
    if state.get("anchor_failed"):
        needs_followup = True
        clarification_reason = "anchor_unresolved"
    if is_nearby_query(query) and not state.get("anchor"):
        needs_followup = True
        clarification_reason = clarification_reason or "missing_reference"
    result = {
        "requirements_ok": not needs_followup,
        "needs_followup": needs_followup,
        "clarification_reason": clarification_reason,
    }
    append_node_trace_result(state.get("query", ""), "check_required_constraints", result)
    return result


async def check_recommend_cache_node(state: GraphState) -> Dict:
    cache_key = build_recommend_cache_key(state)
    cached = get_recommend_cache().get(cache_key)
    result: Dict = {
        "recommend_cache_key": cache_key,
        "recommend_cache_hit": bool(cached),
    }
    if cached:
        result.update(
            {
                "retrievals": cached.get("retrievals") or [],
                "final": cached.get("final") or "",
                "mode": cached.get("mode"),
                "mode_detected": cached.get("mode_detected"),
                "mode_unknown": cached.get("mode_unknown"),
                "anchor": cached.get("anchor"),
                "resolved_name": cached.get("resolved_name"),
                "place": cached.get("place"),
                "last_radius_km": cached.get("last_radius_km"),
            }
        )
    append_node_trace_result(
        state.get("query", ""),
        "check_recommend_cache",
        {
            "recommend_cache_hit": result["recommend_cache_hit"],
            "recommend_cache_key": cache_key,
            "retrievals": slim_retrievals(result.get("retrievals") or []),
        },
    )
    return result


async def retrieve_candidates_node(state: GraphState) -> Dict:
    result = await retrieve_node(state)
    append_node_trace_result(
        state.get("query", ""),
        "retrieve_candidates",
        {**result, "retrievals": slim_retrievals(result.get("retrievals") or [])},
    )
    return result


async def filter_candidates_node(state: GraphState) -> Dict:
    result = await apply_location_filter_node(state)
    retrievals = result.get("retrievals")
    if retrievals is None:
        retrievals = state.get("retrievals") or []
        result = {"retrievals": retrievals}
    append_node_trace_result(
        state.get("query", ""),
        "filter_candidates",
        {"retrievals": slim_retrievals(result.get("retrievals") or [])},
    )
    return result


async def rank_candidates_node(state: GraphState) -> Dict:
    retrievals = list(state.get("retrievals") or [])
    retrievals.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    result = {"retrievals": retrievals}
    append_node_trace_result(
        state.get("query", ""),
        "rank_candidates",
        {"retrievals": slim_retrievals(retrievals)},
    )
    return result


async def check_results_node(state: GraphState) -> Dict:
    retrievals = state.get("retrievals") or []
    result = {"has_results": bool(retrievals), "result_count": len(retrievals)}
    append_node_trace_result(state.get("query", ""), "check_results", result)
    return result


async def ask_followup_node(state: GraphState):
    raw_query = str(state.get("query") or "").strip()
    reason = state.get("clarification_reason")
    if reason == "anchor_unresolved":
        final_text = "말씀하신 위치를 정확히 찾지 못했어요. 역, 건물명, 동 이름처럼 더 구체적인 기준 위치를 알려주세요."
        append_node_trace_result(raw_query, "ask_followup", {"final": final_text})
        yield {"final": final_text}
        yield {"context": build_context(state)}
        return
    async for chunk in clarify_query_node(state):
        if "final" in chunk:
            append_node_trace_result(raw_query, "ask_followup", {"final": chunk["final"]})
        yield chunk


async def fallback_no_result_node(state: GraphState):
    final_text = _build_no_result_text(state)
    append_node_trace_result(state.get("query", ""), "fallback_no_result", {"final": final_text})
    yield {"final": final_text}
    yield {"context": build_context(state)}


async def generate_answer_node(state: GraphState):
    names = _extract_names(state.get("retrievals") or [])
    if names:
        yield {"last_recommended_names": names}
    async for chunk in answer_node(state):
        yield chunk


async def generate_answer_from_cache_node(state: GraphState):
    retrievals = state.get("retrievals") or []
    if state.get("debug") and retrievals:
        yield {"debug": retrievals}
    names = _extract_names(retrievals)
    if names:
        yield {"last_recommended_names": names}
    final_text = str(state.get("final") or "").strip()
    if not final_text:
        async for chunk in answer_node(state):
            yield chunk
        return
    append_node_trace_result(
        state.get("query", ""),
        "generate_answer_from_cache",
        {"cached": True, "final": final_text, "retrievals": slim_retrievals(retrievals)},
    )
    yield {"final": final_text}
    yield {"context": build_context(state)}


async def save_recommend_cache_node(state: GraphState) -> Dict:
    retrievals = state.get("retrievals") or []
    final_text = str(state.get("final") or "").strip()
    cache_key = state.get("recommend_cache_key") or build_recommend_cache_key(state)
    if not cache_key or not retrievals or not final_text:
        result = {"recommend_cache_saved": False}
        append_node_trace_result(state.get("query", ""), "save_recommend_cache", result)
        return result
    names = _extract_names(retrievals)
    get_recommend_cache().set(
        cache_key,
        {
            "retrievals": retrievals,
            "final": final_text,
            "mode": state.get("mode"),
            "mode_detected": state.get("mode_detected"),
            "mode_unknown": state.get("mode_unknown"),
            "anchor": state.get("anchor"),
            "resolved_name": state.get("resolved_name"),
            "place": state.get("place"),
            "last_radius_km": state.get("last_radius_km"),
            "last_recommended_names": names,
        },
    )
    result = {"recommend_cache_saved": True, "last_recommended_names": names}
    append_node_trace_result(state.get("query", ""), "save_recommend_cache", result)
    return result


async def update_context_node(state: GraphState) -> Dict:
    context = build_context(state)
    append_node_trace_result(state.get("query", ""), "update_context", {"context": context})
    return {"context": context}


async def lookup_place_node(state: GraphState) -> Dict:
    query = _build_place_detail_query(state)
    history_place_ids: List[int] = state.get("history_place_ids") or []
    mode_hint = state.get("explicit_mode") or state.get("mode")
    mode_raw = detect_mode(mode_hint, query)
    if mode_raw == "unknown":
        try:
            mode_raw = await llm_detect_mode(query, callbacks=state.get("callbacks"))
        except Exception:
            mode_raw = "unknown"
    modes = [mode_raw] if mode_raw in MODE_ORDER else list(MODE_ORDER)
    tasks = [
        retrieve(query=query, mode=mode, top_k=3, history_place_ids=history_place_ids, debug=bool(state.get("debug")))
        for mode in modes
    ]
    hits_by_mode = await asyncio.gather(*tasks)
    context = state.get("context") or {}
    mentioned_names = [str(name) for name in (context.get("last_recommended_names") or []) if name]
    merged_hits = [hit for hits in hits_by_mode for hit in hits]
    merged_hits.sort(key=lambda hit: _detail_sort_key(query, hit, mentioned_names), reverse=True)
    deduped: List[dict] = []
    seen: set[int] = set()
    for hit in merged_hits:
        pid = hit.get("place_id")
        if pid in seen:
            continue
        seen.add(pid)
        deduped.append(hit)
        if len(deduped) >= 3:
            break
    result = {
        "normalized_query": query,
        "retrievals": deduped,
        "mode": deduped[0]["category"] if deduped else (mode_raw if mode_raw in MODE_ORDER else None),
    }
    append_node_trace_result(
        state.get("query", ""),
        "lookup_place",
        {"normalized_query": query, "retrievals": slim_retrievals(deduped), "mode": result.get("mode")},
    )
    return result


async def generate_place_answer_node(state: GraphState):
    retrievals = state.get("retrievals") or []
    if not retrievals:
        final_text = "관련 장소 정보를 찾지 못했어요. 장소명을 조금 더 구체적으로 알려주세요."
        append_node_trace_result(state.get("query", ""), "generate_place_answer", {"final": final_text})
        yield {"final": final_text}
        yield {"context": build_context(state)}
        return

    if state.get("debug"):
        yield {"debug": retrievals}
    names = _extract_names(retrievals)
    if names:
        yield {"last_recommended_names": names}

    callbacks = state.get("callbacks")
    config = build_callbacks_config(callbacks)

    def _ctx(item: dict) -> str:
        meta = item.get("meta") or {}
        name = meta.get("name") or meta.get("title") or "장소"
        parts = [f"[장소명]: {name}"]
        for label, value in (
            ("설명", meta.get("summary_one_sentence") or meta.get("description") or meta.get("overview")),
            ("주소", meta.get("address") or meta.get("location", {}).get("addr1")),
            ("키워드", ", ".join(map(str, meta.get("keywords", []))) if meta.get("keywords") else ""),
            ("정보", " / ".join(
                part for part in (
                    f"평점 {meta.get('starts')}" if meta.get("starts") else "",
                    f"리뷰 {meta.get('counts')}" if meta.get("counts") else "",
                ) if part
            )),
        ):
            text = str(value or "").strip()
            if text:
                parts.append(f"[{label}]: {text}")
        return " | ".join(parts)

    context_str = "\n".join(f"- {_ctx(item)}" for item in retrievals[:3])
    messages = normalize_messages(
        [
            (
                "system",
                "너는 서울 장소 정보를 안내하는 챗봇이다. "
                "반드시 아래 후보 정보만 사용해서 답하라. 후보 밖 정보는 추측하지 마라. "
                "질문한 정보가 후보에 없으면 없다고 분명히 말하라. "
                "마크다운 금지: **, *, _, `, # 등 강조/코드/헤더 사용 금지.",
            ),
            ("system", f"후보 정보:\n{context_str}"),
            ("user", str(state.get("query") or "").strip()),
        ]
    )

    parts: List[str] = []
    try:
        async for chunk in llm.astream(messages, config=config):
            content = chunk.content
            if not content:
                continue
            parts.append(content)
            yield {"token": content}
        final_text = "".join(parts)
    except LLMRateLimitError:
        final_text = "".join(parts) if parts else rate_limit_fallback_text()

    append_node_trace_result(state.get("query", ""), "generate_place_answer", {"final": final_text})
    yield {"final": final_text}
    yield {"context": build_context(state)}


async def general_chat_node(state: GraphState):
    async for chunk in general_answer_node(state):
        yield chunk
