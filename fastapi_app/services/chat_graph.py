from langgraph.graph import END, StateGraph

from services.chat_nodes.flow_nodes import (
    ask_followup_node,
    check_recommend_cache_node,
    check_required_constraints_node,
    check_results_node,
    extract_constraints_node,
    fallback_no_result_node,
    filter_candidates_node,
    general_chat_node,
    generate_answer_from_cache_node,
    generate_answer_node,
    generate_place_answer_node,
    lookup_place_node,
    merge_with_previous_context_node,
    normalize_constraints_node,
    query_understanding_node,
    rank_candidates_node,
    retrieve_candidates_node,
    route_intent_node,
    save_recommend_cache_node,
    update_context_node,
)
from services.chat_nodes.state import GraphState


def _route_intent(state: GraphState) -> str:
    return str(state.get("route_intent") or "general_chat")


def _needs_followup(state: GraphState) -> str:
    return "ask_followup" if state.get("needs_followup") else "check_recommend_cache"


def _recommend_cache_branch(state: GraphState) -> str:
    return "generate_answer_from_cache" if state.get("recommend_cache_hit") else "retrieve_candidates"


def _result_branch(state: GraphState) -> str:
    return "generate_answer" if state.get("has_results") else "fallback_no_result"


workflow = StateGraph(GraphState)
workflow.add_node("query_understanding", query_understanding_node)
workflow.add_node("route_intent", route_intent_node)
workflow.add_node("extract_constraints", extract_constraints_node)
workflow.add_node("merge_with_previous_context", merge_with_previous_context_node)
workflow.add_node("normalize_constraints", normalize_constraints_node)
workflow.add_node("check_required_constraints", check_required_constraints_node)
workflow.add_node("ask_followup", ask_followup_node)
workflow.add_node("check_recommend_cache", check_recommend_cache_node)
workflow.add_node("generate_answer_from_cache", generate_answer_from_cache_node)
workflow.add_node("retrieve_candidates", retrieve_candidates_node)
workflow.add_node("filter_candidates", filter_candidates_node)
workflow.add_node("rank_candidates", rank_candidates_node)
workflow.add_node("check_results", check_results_node)
workflow.add_node("fallback_no_result", fallback_no_result_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("save_recommend_cache", save_recommend_cache_node)
workflow.add_node("update_context", update_context_node)
workflow.add_node("lookup_place", lookup_place_node)
workflow.add_node("generate_place_answer", generate_place_answer_node)
workflow.add_node("general_chat", general_chat_node)

workflow.set_entry_point("query_understanding")
workflow.add_edge("query_understanding", "route_intent")

workflow.add_conditional_edges(
    "route_intent",
    _route_intent,
    {
        "recommendation": "extract_constraints",
        "followup_recommend": "merge_with_previous_context",
        "place_detail": "lookup_place",
        "general_chat": "general_chat",
    },
)

workflow.add_edge("extract_constraints", "normalize_constraints")
workflow.add_edge("merge_with_previous_context", "normalize_constraints")
workflow.add_edge("normalize_constraints", "check_required_constraints")
workflow.add_conditional_edges(
    "check_required_constraints",
    _needs_followup,
    {
        "ask_followup": "ask_followup",
        "check_recommend_cache": "check_recommend_cache",
    },
)
workflow.add_edge("ask_followup", END)

workflow.add_conditional_edges(
    "check_recommend_cache",
    _recommend_cache_branch,
    {
        "generate_answer_from_cache": "generate_answer_from_cache",
        "retrieve_candidates": "retrieve_candidates",
    },
)
workflow.add_edge("generate_answer_from_cache", "update_context")
workflow.add_edge("retrieve_candidates", "filter_candidates")
workflow.add_edge("filter_candidates", "rank_candidates")
workflow.add_edge("rank_candidates", "check_results")
workflow.add_conditional_edges(
    "check_results",
    _result_branch,
    {
        "generate_answer": "generate_answer",
        "fallback_no_result": "fallback_no_result",
    },
)
workflow.add_edge("fallback_no_result", END)
workflow.add_edge("generate_answer", "save_recommend_cache")
workflow.add_edge("save_recommend_cache", "update_context")
workflow.add_edge("update_context", END)

workflow.add_edge("lookup_place", "generate_place_answer")
workflow.add_edge("generate_place_answer", "update_context")
workflow.add_edge("general_chat", END)

chat_app = workflow.compile()
