from typing import Dict

from services.chat_nodes.intent import detect_intent, is_expand_query
from services.chat_nodes.state import GraphState
from utils.geo import append_node_trace_result


async def route_node(state: GraphState) -> Dict:
    """Decide the high-level intent and whether the user asked to expand the radius."""
    query = state.get("query", "")
    normalized_query = state.get("normalized_query")
    normalized_query = query if normalized_query is None else str(normalized_query).strip()
    query = str(query).strip()
    if not normalized_query:
        result = {"intent": "general", "expand_request": False, "empty_query": True}
        append_node_trace_result(query, "route", result)
        return result

    intent = detect_intent(normalized_query)
    expand_request = is_expand_query(query) or is_expand_query(normalized_query)
    result = {"intent": intent, "expand_request": expand_request}
    append_node_trace_result(query, "route", result)
    return result
