from typing import Dict

from services.chat_nodes.place_llm import llm_extract_place
from services.chat_nodes.state import GraphState
from utils.geo import append_place_debug, append_node_trace_result


async def extract_place_node(state: GraphState) -> Dict:
    """Extract a place candidate (area/point) from the normalized query."""
    query = state.get("normalized_query") or state.get("query", "")
    # LLM returns a strict JSON-like structure or None if no place is found.
    place = await llm_extract_place(query, callbacks=state.get("callbacks"))
    # Debug log captures what the model extracted for later inspection.
    append_place_debug(
        {
            "query": query,
            "place": place,
        }
    )
    result = {"place": place}
    # Node trace is used to replay and inspect the pipeline flow.
    append_node_trace_result(query, "extract_place", result)
    return result
