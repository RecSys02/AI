import os
from typing import Annotated, Dict, List, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from services.retriever import retrieve

# 상태 정의를 명확히 하여 데이터 유실 방지
class GraphState(TypedDict):
    query: str
    mode: str
    top_k: int
    history_place_ids: List[int]
    intent: str
    retrievals: List[dict]
    final: str
    messages: List[dict]

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=CHAT_MODEL, streaming=True, temperature=0.0)

def _detect_intent(query: str) -> str:
    q = query.lower()
    recommend_terms = ["추천", "어디", "가볼", "뭐가 있어", "top", "best", "3개", "5곳"]
    return "recommend" if any(t in q for t in recommend_terms) else "general"

async def route_node(state: GraphState) -> Dict:
    intent = _detect_intent(state.get("query", ""))
    return {"intent": intent} # 변경된 부분만 반환

async def retrieve_node(state: GraphState) -> Dict:
    query = state.get("query", "")
    mode = state.get("mode", "tourspot")
    requested_k = state.get("top_k", 1)
    history_ids = state.get("history_place_ids") or []
    # 후보를 넉넉히 가져옴
    hits = retrieve(query=query, mode=mode, top_k=max(requested_k, 20), history_place_ids=history_ids)
    return {"retrievals": hits}

async def answer_node(state: GraphState):
    retrievals = state.get("retrievals", [])
    if not retrievals:
        yield {"final": "검색된 결과가 없습니다."}
        return

    # 후보 컨텍스트: 이름 + 요약/설명 + 키워드까지 포함
    def _build_ctx(r: dict) -> str:
        meta = r.get("meta") or {}
        name = meta.get("name") or meta.get("title") or "장소"
        summary = meta.get("summary_one_sentence") or meta.get("description") or meta.get("content") or ""
        kw = meta.get("keywords") or []
        kw_str = ", ".join([str(k) for k in kw]) if kw else ""
        parts = [f"{name}", summary]
        if kw_str:
            parts.append(f"키워드: {kw_str}")
        return " ".join([p for p in parts if p])

    context = "\n".join([f"- {_build_ctx(r)}" for r in retrievals])
    messages = [
        (
            "system",
            "너는 서울 여행 가이드다. 아래 후보 목록에서만 선택해 한 줄 요약과 함께 추천해라. "
            "후보에 없는 장소는 절대 언급하지 말고, 후보 정보(이름/설명/키워드)를 활용해 답하라.",
        ),
        ("system", f"추천 개수: {state.get('top_k', 1)}\n후보:\n{context}"),
        ("user", state.get("query", "")),
    ]

    parts: List[str] = []
    async for chunk in llm.astream(messages):
        content = chunk.content
        if not content:
            continue
        parts.append(content)
        yield {"token": content}

    final_text = "".join(parts)
    yield {"final": final_text}


async def general_answer_node(state: GraphState):
    messages = [
        ("system", "서울 여행 관련 질문에 간결하게 답해라."),
        ("user", state.get("query", "")),
    ]
    parts: List[str] = []
    async for chunk in llm.astream(messages):
        content = chunk.content
        if not content:
            continue
        parts.append(content)
        yield {"token": content}

    final_text = "".join(parts)
    yield {"final": final_text}

# 그래프 구성
workflow = StateGraph(GraphState)

workflow.add_node("route", route_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("answer", answer_node)
workflow.add_node("general_answer", general_answer_node)

workflow.set_entry_point("route")

# 조건부 엣지
workflow.add_conditional_edges(
    "route",
    lambda state: "retrieve" if state["intent"] == "recommend" else "general_answer",
    {"retrieve": "retrieve", "general_answer": "general_answer"}
)

workflow.add_edge("retrieve", "answer")
workflow.add_edge("answer", END)
workflow.add_edge("general_answer", END)

chat_app = workflow.compile()
