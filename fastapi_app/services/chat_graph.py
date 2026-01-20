from langgraph.graph import END, StateGraph

from services.chat_nodes.answer import answer_node
from services.chat_nodes.apply_location_filter import apply_location_filter_node
from services.chat_nodes.correct_place import correct_place_node
from services.chat_nodes.expand_radius import expand_radius_node
from services.chat_nodes.extract_place import extract_place_node
from services.chat_nodes.general_answer import general_answer_node
from services.chat_nodes.normalize_query import normalize_query_node
from services.chat_nodes.rerank import rerank_node
from services.chat_nodes.resolve_anchor import resolve_anchor_node
from services.chat_nodes.retrieve import retrieve_node
from services.chat_nodes.route import route_node
from services.chat_nodes.state import GraphState


# 그래프 구성
workflow = StateGraph(GraphState)
workflow.add_node("route", route_node)
workflow.add_node("normalize_query", normalize_query_node)
workflow.add_node("extract_place", extract_place_node)
workflow.add_node("correct_place", correct_place_node)
workflow.add_node("expand_radius", expand_radius_node)
workflow.add_node("resolve_anchor", resolve_anchor_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("apply_location_filter", apply_location_filter_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("answer", answer_node)
workflow.add_node("general_answer", general_answer_node)
workflow.set_entry_point("route")

# 조건부 엣지
workflow.add_conditional_edges(
    "route",
    lambda state: "normalize_query" if state["intent"] == "recommend" else "general_answer",
    {"normalize_query": "normalize_query", "general_answer": "general_answer"},
)

workflow.add_edge("normalize_query", "extract_place")
workflow.add_edge("extract_place", "correct_place")
workflow.add_conditional_edges(
    "correct_place",
    lambda state: "expand_radius" if state.get("expand_request") else "resolve_anchor",
    {"expand_radius": "expand_radius", "resolve_anchor": "resolve_anchor"},
)
workflow.add_conditional_edges(
    "expand_radius",
    lambda state: "answer" if state.get("expand_failed") else "retrieve",
    {"answer": "answer", "retrieve": "retrieve"},
)
workflow.add_edge("resolve_anchor", "retrieve")
workflow.add_edge("retrieve", "apply_location_filter")
workflow.add_edge("apply_location_filter", "rerank")
workflow.add_edge("rerank", "answer")
workflow.add_edge("answer", END)
workflow.add_edge("general_answer", END)

chat_app = workflow.compile()
