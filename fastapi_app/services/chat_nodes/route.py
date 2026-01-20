from typing import Dict

from services.chat_nodes.intent import detect_intent, is_expand_query
from services.chat_nodes.state import GraphState
from utils.geo import append_node_trace_result


async def route_node(state: GraphState) -> Dict:
    query = state.get("query", "")
    intent = detect_intent(query)
    expand_request = is_expand_query(query)
    result = {"intent": intent, "expand_request": expand_request}
    append_node_trace_result(query, "route", result)
    return result
