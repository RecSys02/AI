from typing import Dict

from services.chat_nodes.state import GraphState, slim_retrievals
from utils.geo import append_node_trace_result, distance_to_centers_km, get_lat_lng


async def apply_location_filter_node(state: GraphState) -> Dict:
    """Filter retrieved candidates by anchor radius or admin-term address match."""
    retrievals = state.get("retrievals") or []
    if not retrievals:
        result = {}
        append_node_trace_result(state.get("query", ""), "apply_location_filter", result)
        return result

    anchor = state.get("anchor") or {}
    admin_term = state.get("admin_term")
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

    if admin_term:
        # Admin-term filter checks address fields for substring matches.
        term = str(admin_term).lower()
        filtered = []
        for r in retrievals:
            meta = r.get("meta") or {}
            addr_parts = [
                str(meta.get("city") or ""),
                str(meta.get("district") or ""),
                str(meta.get("dong") or ""),
                str(meta.get("road") or ""),
                str(meta.get("address") or meta.get("location", {}).get("addr1") or ""),
            ]
            addr_blob = " ".join(addr_parts).lower()
            if term in addr_blob:
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
