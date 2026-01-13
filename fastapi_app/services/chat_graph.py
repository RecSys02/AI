import os
from typing import Dict, List

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from services.retriever import retrieve

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# LLM은 스트리밍 모드로 사용
llm = ChatOpenAI(model=CHAT_MODEL, streaming=True, temperature=0.0)


def retrieve_node(state: Dict) -> Dict:
    query = state.get("query", "")
    mode = state.get("mode", "tourspot")
    top_k = state.get("top_k", 5)
    history_place_ids: List[int] = state.get("history_place_ids") or []
    hits = retrieve(query=query, mode=mode, top_k=top_k, history_place_ids=history_place_ids)
    return {**state, "retrievals": hits}


def answer_node(state: Dict):
    retrievals = state.get("retrievals", [])
    if not retrievals:
        msg = "관련 후보를 찾지 못했습니다. 다른 요청이나 조건을 알려주세요."
        yield {**state, "answer": msg, "messages": state.get("messages", []) + [{"role": "assistant", "content": msg}]}
        return

    top_k = state.get("top_k", 1)
    context_lines = []
    for r in retrievals:
        meta = r.get("meta") or {}
        name = meta.get("name") or meta.get("title") or "장소"
        summary = meta.get("summary_one_sentence") or meta.get("description") or ""
        context_lines.append(f"- {name}: {summary}")
    context = "\n".join(context_lines)

    messages = [
        (
            "system",
            "너는 서울 여행지/맛집/카페를 추천하는 챗봇이다. "
            "반드시 아래 후보 목록에서만 선택해 최대 N개 추천해라. "
            "후보에 없는 장소는 절대 제시하지 말고, 부족하면 없는대로 알려라. "
            "불필요한 서론 없이 바로 추천을 제공해라.",
        ),
    ]
    if context:
        messages.append(("system", f"N={top_k}\n후보 목록:\n{context}"))
    messages.append(("user", state.get("query", "")))

    full = []
    for chunk in llm.stream(messages):
        content = chunk.content
        if not content:
            continue
        full.append(content)
        yield {"answer": content}

    final_answer = "".join(full)
    yield {**state, "answer": final_answer, "messages": state.get("messages", []) + [{"role": "assistant", "content": final_answer}]}


# LangGraph 구성
graph = StateGraph(dict)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)
graph.set_entry_point("retrieve")

chat_app = graph.compile()
