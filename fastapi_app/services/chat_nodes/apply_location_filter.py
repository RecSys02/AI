from typing import Dict

from services.chat_nodes.state import GraphState, slim_retrievals
from utils.geo import append_node_trace_result, distance_to_centers_km, get_lat_lng


async def apply_location_filter_node(state: GraphState) -> Dict:
    """Filter retrieved candidates by anchor radius."""
    retrievals = state.get("retrievals") or []
    if not retrievals:
        result = {}
        append_node_trace_result(state.get("query", ""), "apply_location_filter", result)
        return result

    anchor = state.get("anchor") or {}
    mode_used = state.get("mode") or "tourspot"

    if anchor:
        # Distance filter uses anchor center(s) and per-mode radius.
        centers = anchor.get("centers") or []
        radius_by_intent = anchor.get("radius_by_intent") or {}
        radius_km = float(radius_by_intent.get(mode_used, 2.0))
        filtered = []
        for r in retrievals:
            meta = r.get("meta") or {}
            lat, lng = get_lat_lng(meta)
            if lat is None or lng is None:
                continue
            dist_km = distance_to_centers_km(lat, lng, centers)
            if dist_km is not None and dist_km <= radius_km:
                filtered.append(r)
        result = {"retrievals": filtered}
        append_node_trace_result(
            state.get("query", ""),
            "apply_location_filter",
            {"retrievals": slim_retrievals(result["retrievals"])},
        )
        return result

    result = {}
    append_node_trace_result(state.get("query", ""), "apply_location_filter", result)
    return result
