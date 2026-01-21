from typing import Dict

from services.chat_nodes.intent import detect_intent, is_expand_query
from services.chat_nodes.state import GraphState
from utils.geo import append_node_trace_result


async def route_node(state: GraphState) -> Dict:
    """Decide the high-level intent and whether the user asked to expand the radius."""
    query = state.get("query", "")
    # Rule-based intent detection keeps routing deterministic and fast.
    intent = detect_intent(query)
    # "Expand" is a special control intent used to widen an existing anchor radius.
    expand_request = is_expand_query(query)
    result = {"intent": intent, "expand_request": expand_request}
    append_node_trace_result(query, "route", result)
    return result
