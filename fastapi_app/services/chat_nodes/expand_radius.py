from typing import Dict

from services.chat_nodes.state import GraphState
from utils.geo import append_node_trace_result


async def expand_radius_node(state: GraphState) -> Dict:
    """Increase the previous anchor radius when the user asks to widen the search."""
    query = state.get("query", "")
    if not state.get("expand_request"):
        result = {}
        append_node_trace_result(query, "expand_radius", result)
        return result

    ctx = state.get("context") or {}
    last_anchor = ctx.get("last_anchor") or {}
    centers = last_anchor.get("centers") or []
    if not centers:
        # Cannot expand without a prior anchor to expand from.
        result = {"expand_failed": True}
        append_node_trace_result(query, "expand_radius", result)
        return result

    last_mode = ctx.get("last_mode") or state.get("mode") or "tourspot"
    radius_by_intent = dict(last_anchor.get("radius_by_intent") or {})
    default_radius_by_intent = {"restaurant": 2.0, "cafe": 2.0, "tourspot": 3.0}
    base_radius = ctx.get("last_radius_km")
    if base_radius is None:
        base_radius = radius_by_intent.get(last_mode, default_radius_by_intent.get(last_mode, 2.0))
    new_radius = min(float(base_radius) + 1.0, 10.0)
    # Only expand the active mode radius to avoid surprising changes.
    radius_by_intent[last_mode] = new_radius

    anchor = {
        "centers": centers,
        "radius_by_intent": radius_by_intent,
        "source": last_anchor.get("source") or "context",
    }
    base_query = ctx.get("last_query") or state.get("query", "")
    result = {
        "query": base_query,
        "normalized_query": base_query,
        "anchor": anchor,
        "last_radius_km": new_radius,
        "mode": last_mode,
        "resolved_name": ctx.get("last_resolved_name"),
        "admin_term": ctx.get("last_admin_term"),
        "place": ctx.get("last_place"),
        "expand_failed": False,
    }
    append_node_trace_result(
        query,
        "expand_radius",
        {
            "anchor": {"centers": centers, "radius_by_intent": radius_by_intent, "source": anchor["source"]},
            "last_radius_km": new_radius,
            "mode": last_mode,
            "resolved_name": ctx.get("last_resolved_name"),
            "admin_term": ctx.get("last_admin_term"),
            "place": ctx.get("last_place"),
            "expand_failed": False,
        },
    )
    return result
