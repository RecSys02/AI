from services.chat_nodes.state import GraphState, build_context
from utils.geo import append_node_trace_result


async def clarify_query_node(state: GraphState):
    """Ask the user for missing context instead of guessing blindly."""
    raw_query = str(state.get("query") or "").strip()
    reason = state.get("clarification_reason")
    if reason == "missing_reference":
        final_text = "어느 장소를 말씀하시는지 모르겠어요. 기준이 될 지역이나 지점, 역, 건물명을 알려주세요."
    elif reason == "missing_place":
        final_text = "어느 지역이나 장소를 기준으로 찾을지 알려주세요. 예: '강남 식당 추천해줘', '코엑스 주변 카페 추천해줘'"
    elif reason == "empty_query":
        final_text = "질문이 너무 짧아요. 원하는 지역이나 카테고리를 조금만 더 알려주세요."
    else:
        final_text = (
            "원하는 지역이나 카테고리를 조금만 더 알려주세요. "
            "예: '강남 식당 추천해줘', '성수 카페 추천해줘'"
        )
    append_node_trace_result(raw_query, "clarify_query", {"final": final_text})
    yield {"final": final_text}
    yield {"context": build_context(state)}
