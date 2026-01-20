from typing import Dict

from services.chat_nodes.place_llm import llm_extract_place
from services.chat_nodes.state import GraphState
from utils.geo import append_place_debug, append_node_trace_result


async def extract_place_node(state: GraphState) -> Dict:
    query = state.get("normalized_query") or state.get("query", "")
    place = await llm_extract_place(query)
    append_place_debug(
        {
            "query": query,
            "place": place,
        }
    )
    result = {"place": place}
    append_node_trace_result(query, "extract_place", result)
    return result
