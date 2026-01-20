from typing import Dict

from services.chat_nodes.place_llm import llm_correct_place
from services.chat_nodes.state import GraphState
from utils.geo import (
    add_alias,
    append_node_trace_result,
    load_admin_aliases,
    load_keyword_aliases,
    save_admin_aliases,
    save_keyword_aliases,
)
from utils.google_place_autocomplete import autocomplete_places


async def correct_place_node(state: GraphState) -> Dict:
    query = state.get("normalized_query") or state.get("query", "")
    place = state.get("place") or {}
    area = place.get("area")
    point = place.get("point")
    if not area and not point:
        result = {}
        append_node_trace_result(query, "correct_place", result)
        return result

    corrected = await llm_correct_place(query, area, point)
    if not corrected or not corrected.get("changed"):
        result = {}
        append_node_trace_result(query, "correct_place", result)
        return result

    new_area = corrected.get("area") or area
    new_point = corrected.get("point") or point

    def _valid_autocomplete(value: str, types: str | None = "(regions)") -> bool:
        if not value:
            return False
        types_param = "" if types is None else types
        return bool(autocomplete_places(value, limit=1, types=types_param))

    # validate point correction first
    if new_point and new_point != point and not _valid_autocomplete(new_point, None):
        new_point = point

    # validate area correction
    if new_area and new_area != area:
        admin_suffixes = ("시", "구", "동", "가", "로", "길", "대로")
        if new_area.endswith(admin_suffixes):
            if not _valid_autocomplete(new_area, "(regions)"):
                new_area = area
        else:
            if not _valid_autocomplete(new_area, None):
                new_area = area

    if new_area == area and new_point == point:
        result = {}
        append_node_trace_result(query, "correct_place", result)
        return result

    # cache alias if correction is meaningful
    if area and new_area and area != new_area:
        if new_area.endswith(("시", "구", "동", "가", "로", "길", "대로")):
            admin_aliases = load_admin_aliases()
            if add_alias(admin_aliases, new_area, area):
                save_admin_aliases(admin_aliases)
        else:
            keyword_aliases = load_keyword_aliases()
            if add_alias(keyword_aliases, new_area, area):
                save_keyword_aliases(keyword_aliases)

    if point and new_point and point != new_point:
        keyword_aliases = load_keyword_aliases()
        if add_alias(keyword_aliases, new_point, point):
            save_keyword_aliases(keyword_aliases)

    updated_place = {"area": new_area, "point": new_point}
    result = {"place": updated_place, "place_original": place}
    append_node_trace_result(
        query,
        "correct_place",
        {"before": place, "after": updated_place},
    )
    return result
