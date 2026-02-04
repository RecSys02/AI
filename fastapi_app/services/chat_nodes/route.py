from typing import Dict

from services.chat_nodes.callbacks import build_callbacks_config
from services.chat_nodes.intent import detect_intent, is_expand_query
from services.chat_nodes.llm_clients import detect_llm, max_tokens_kwargs, parse_json_response
from services.chat_nodes.state import GraphState
from utils.geo import append_node_trace_result


async def route_node(state: GraphState) -> Dict:
    """Decide the high-level intent and whether the user asked to expand the radius."""
    query = state.get("query", "")
    normalized_query = state.get("normalized_query")
    normalized_query = query if normalized_query is None else str(normalized_query).strip()
    query = str(query).strip()
    callbacks = state.get("callbacks")
    config = build_callbacks_config(callbacks)

    if not normalized_query:
        result = {"intent": "general", "expand_request": False, "empty_query": True}
        append_node_trace_result(query, "route", result)
        return result

    # Fallback to rule-based detection if the LLM response is missing or malformed.
    intent = detect_intent(normalized_query)
    expand_request = is_expand_query(query) or is_expand_query(normalized_query)

    messages = [
        (
            "system",
            "너는 사용자 질문의 상위 intent를 분류하는 라우터다.\n"
            "아래 규칙에 따라 JSON만 반환하라.\n"
            "1) intent는 recommend 또는 general 중 하나만 선택한다.\n"
            "2) expand_request는 사용자가 검색 범위/반경 확대를 명시적으로 요청했을 때만 true다.\n"
            "3) 애매하면 recommend보다 general을 선택한다.\n"
            "반환 형식: {\"intent\": \"recommend|general\", \"expand_request\": true|false}",
        ),
        ("user", f"query: {query}\nnormalized_query: {normalized_query}"),
    ]
    try:
        resp = await detect_llm.ainvoke(messages, **max_tokens_kwargs(50), config=config)
        data = parse_json_response((resp.content or "").strip())
        if isinstance(data, dict):
            raw_intent = str(data.get("intent") or "").strip().lower()
            if raw_intent in {"recommend", "general"}:
                intent = raw_intent
            raw_expand = data.get("expand_request")
            if isinstance(raw_expand, bool):
                expand_request = raw_expand
            elif isinstance(raw_expand, str):
                expand_val = raw_expand.strip().lower()
                if expand_val in {"true", "false"}:
                    expand_request = expand_val == "true"
    except Exception:
        pass
    result = {"intent": intent, "expand_request": expand_request}
    append_node_trace_result(query, "route", result)
    return result
