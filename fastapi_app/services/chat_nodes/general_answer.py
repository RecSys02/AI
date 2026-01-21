from typing import List

from services.chat_nodes.config import GENERAL_K
from services.chat_nodes.llm_clients import llm
from services.chat_nodes.mode import detect_mode, llm_detect_mode
from services.chat_nodes.state import GraphState, build_context
from services.retriever import retrieve
from utils.geo import append_node_trace_result


async def general_answer_node(state: GraphState):
    """Answer non-recommendation questions using lightweight retrieval + LLM."""
    query = state.get("query", "")
    # Detect category (tourspot/cafe/restaurant) to pick the right index.
    mode_raw = detect_mode(state.get("mode"), query)
    if mode_raw == "unknown":
        mode_raw = await llm_detect_mode(query)
    mode_unknown = mode_raw == "unknown"
    mode_used = "tourspot" if mode_unknown else mode_raw
    history_place_ids: List[int] = state.get("history_place_ids") or []
    if state.get("debug"):
        # general에서도 같은 형태로 디버그 반환
        pass

    # 일반 질의도 데이터 기반으로 답하도록 간단히 검색 사용
    hits = retrieve(
        query=query,
        mode=mode_used,
        top_k=GENERAL_K,
        history_place_ids=history_place_ids,
    )
    if not hits:
        # No search results for general queries.
        yield {"final": "관련 정보를 찾지 못했습니다."}
        yield {"context": build_context(state)}
        return
    if state.get("debug"):
        yield {"debug": hits}

    def _build_ctx(r: dict) -> str:
        meta = r.get("meta") or {}
        name = meta.get("name") or meta.get("title") or "장소"
        desc = meta.get("summary_one_sentence") or meta.get("description") or meta.get("content") or ""
        addr = meta.get("address") or meta.get("location", {}).get("addr1") or ""
        kw = meta.get("keywords") or []
        kw_str = ", ".join([str(k) for k in kw]) if kw else ""
        popularity = []
        if meta.get("views") is not None:
            popularity.append(f"조회수 {meta['views']}")
        if meta.get("likes") is not None:
            popularity.append(f"좋아요 {meta['likes']}")
        if meta.get("bookmarks") is not None:
            popularity.append(f"북마크 {meta['bookmarks']}")
        rating_parts = []
        if meta.get("starts"):
            rating_parts.append(f"평점 {meta['starts']}")
        if meta.get("counts"):
            rating_parts.append(f"리뷰 {meta['counts']}")
        rating_str = ", ".join(rating_parts)
        pop_str = ", ".join(popularity)
        parts = [name, desc]
        if addr:
            parts.append(f"주소: {addr}")
        if kw_str:
            parts.append(f"키워드: {kw_str}")
        if pop_str:
            parts.append(f"인기: {pop_str}")
        if rating_str:
            parts.append(rating_str)
        return " ".join([p for p in parts if p])

    context = "\n".join([f"- {_build_ctx(r)}" for r in hits])

    messages = [
        (
            "system",
            "너는 서울 여행/맛집/카페 정보를 안내하는 챗봇이다. "
            "반드시 아래 후보 정보만 활용해서 질문에 답해라. 후보 밖 내용은 말하지 마라.",
        ),
        ("system", f"후보 정보:\n{context}"),
    ]
    if mode_unknown:
        # Ask the user to clarify the category when detection is uncertain.
        messages.append(
            (
                "system",
                "모드를 정확히 인식하지 못했다면 관광지 기준으로 임시로 답하고,"
                "사용자에게 식당/카페/관광지 중 원하는 카테고리를 물어본다.",
            )
        )
    messages.append(("user", query))

    parts: List[str] = []
    async for chunk in llm.astream(messages):
        content = chunk.content
        if not content:
            continue
        parts.append(content)
        yield {"token": content}

    final_text = "".join(parts)
    append_node_trace_result(state.get("query", ""), "general_answer", {"final": final_text})
    yield {"final": final_text}
    yield {"context": build_context(state)}
