import os
from typing import Annotated, Dict, List, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from services.retriever import retrieve

class GraphState(TypedDict):
    query: str
    mode: str | None
    mode_detected: str | None
    mode_unknown: bool | None
    top_k: int
    history_place_ids: List[int]
    intent: str
    retrievals: List[dict]
    final: str
    messages: List[dict]
    debug: bool | None
    
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
DETECT_MODEL = os.getenv("DETECT_MODEL", CHAT_MODEL)
llm = ChatOpenAI(model=CHAT_MODEL, streaming=True, temperature=0.0)
# 모드 감지용은 스트리밍 없이 호출
detect_llm = ChatOpenAI(model=DETECT_MODEL, streaming=False, temperature=0.0)

def _detect_intent(query: str) -> str:
    q = query.lower()
    # 추천 의도가 포함된 경우 "recommend" 반환
    recommend_terms = ["추천", "어디", "가볼", "뭐가 있어", "top", "best", "3개", "5곳"]

    return "recommend" if any(t in q for t in recommend_terms) else "general"

def _detect_mode(mode_hint: str | None, query: str) -> str:
    # 직접적으로 모드가 주어지면 우선 사용
    if mode_hint in {"tourspot", "cafe", "restaurant"}:
        return mode_hint
    # 힌트가 없으면 쿼리 기반으로 추론
    q = query.lower()
    cafe_terms = ["카페", "커피", "디저트", "브런치", "빵", "라떼", "tea", "티룸"]
    restaurant_terms = [
        "맛집",
        "식당",
        "레스토랑",
        "밥",
        "점심",
        "저녁",
        "고기",
        "파스타",
        "스테이크",
        "스시",
        "초밥",
        "회",
        "중식",
        "한식",
        "양식",
        "분식",
        "라멘",
        "라면",
        "피자",
        "버거",
        "삼겹살",
        "술집",
        "포차",
        "안주",
        "뷔페",
    ]
    # 기본값은 unknown, 카페/맛집 키워드가 있으면 해당 모드로 변경
    if any(t in q for t in cafe_terms):
        return "cafe"
    if any(t in q for t in restaurant_terms):
        return "restaurant"
    return "unknown"


async def _llm_detect_mode(query: str) -> str:
    """LLM으로 모드 분류 (tourspot/cafe/restaurant/unknown 중 하나만 반환)."""
    messages = [
        (
            "system",
            "다음 사용자 질문이 관광지(tourspot), 카페(cafe), 식당/맛집(restaurant) 중 어느 카테고리에 해당하는지 "
            "정확히 하나의 단어만 소문자로 답하라. 해당이 없으면 unknown만 답하라.",
        ),
        ("user", query),
    ]
    try:
        resp = await detect_llm.ainvoke(messages, max_tokens=5)
        mode = (resp.content or "").strip().lower()
        return mode if mode in {"tourspot", "cafe", "restaurant"} else "unknown"
    except Exception:
        return "unknown"

async def route_node(state: GraphState) -> Dict:
    intent = _detect_intent(state.get("query", ""))
    return {"intent": intent} # 변경된 부분만 반환

async def retrieve_node(state: GraphState) -> Dict:
    query = state.get("query", "")
    mode_raw = _detect_mode(state.get("mode"), query)
    if mode_raw == "unknown":
        mode_raw = await _llm_detect_mode(query)
    mode_unknown = mode_raw == "unknown"
    mode_used = "tourspot" if mode_unknown else mode_raw
    requested_k = state.get("top_k", 1)
    # 이전 방문 기록 아직 없음. 어떻게 뽑아올지 정해야함(기존 추천시스템에서 방문한 이력들 DB에서 뽑아오도록 해야할듯?)
    history_ids = state.get("history_place_ids") or []

    debug_flag = bool(state.get("debug"))
    # 후보를 retrieve 함수를 사용해 뽑아옴
    hits = retrieve(
        query=query,
        mode=mode_used,
        top_k=max(requested_k, 20),
        history_place_ids=history_ids,
        debug=debug_flag,
    )
    return {
        "retrievals": hits,
        "mode": mode_used,
        "mode_detected": mode_raw,
        "mode_unknown": mode_unknown,
    }

async def answer_node(state: GraphState):
    retrievals = state.get("retrievals", [])
    if not retrievals:
        yield {"final": "검색된 결과가 없습니다."}
        return
    # retrieval 디버그 정보 포함해서 뭔지 확인하고 싶을때!
    if state.get("debug"):
        yield {"debug": retrievals}

    # 후보 컨텍스트: 이름 + 요약/설명 + 주소 + 키워드까지 포함
    def _build_ctx(r: dict) -> str:
        meta = r.get("meta") or {}
        name = meta.get("name") or meta.get("title") or "장소"
        summary = meta.get("summary_one_sentence") or meta.get("description") or meta.get("content") or ""
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
        parts = [f"{name}", summary]
        if addr:
            parts.append(f"주소: {addr}")
        if kw_str:
            parts.append(f"키워드: {kw_str}")
        if pop_str:
            parts.append(f"인기: {pop_str}")
        if rating_str:
            parts.append(rating_str)
        print(parts)
        return " ".join([p for p in parts if p])

    context = "\n".join([f"- {_build_ctx(r)}" for r in retrievals])
    print(context)
    messages = [
        (
            "system",
            "너는 서울 여행 가이드다. 아래 후보 목록에서만 선택해 한 줄 요약과 함께 추천해라. "
            "후보에 없는 장소는 절대 언급하지 말고, 후보 정보(이름/설명/키워드)를 활용해 답하라.",
        ),
        ("system", f"추천 개수: {state.get('top_k', 1)}\n후보:\n{context}"),
    ]
    if state.get("mode_unknown"):
        messages.append(
            (
                "system",
                "모드를 정확히 인식하지 못했다면 관광지 기준으로 임시 추천했으니, "
                "사용자가 카페/식당/관광지 중 원하는 카테고리를 답하도록 유도하라.",
            )
        )
    messages.append(("user", state.get("query", "")))

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
    query = state.get("query", "")
    mode_raw = _detect_mode(state.get("mode"), query)
    if mode_raw == "unknown":
        mode_raw = await _llm_detect_mode(query)
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
        top_k=max(state.get("top_k", 1), 5),
        history_place_ids=history_place_ids,
    )
    if not hits:
        yield {"final": "관련 정보를 찾지 못했습니다."}
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
        messages.append(
            (
                "system",
                "모드를 정확히 인식하지 못했다면 관광지 기준으로 임시로 답하고, "
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
# lambda 안쓰는 조건부 엣지
'''
def decide_next(state):
    intent = state.get("intent")
    return "retrieve" if intent == "recommend" else "general_answer"

workflow.add_conditional_edges(
    "route",
    decide_next,  # lambda 대신 함수
    {"retrieve": "retrieve", "general_answer": "general_answer"},
)
'''


workflow.add_edge("retrieve", "answer")
workflow.add_edge("answer", END)
workflow.add_edge("general_answer", END)
chat_app = workflow.compile()
