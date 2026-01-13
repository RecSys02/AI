import re
from typing import List, Optional

from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse

from services.chat_graph import chat_app

router = APIRouter()


def _parse_history(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    ids = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except Exception:
            continue
    return ids


def _infer_top_k(query_text: str, top_k_param: Optional[int]) -> int:
    # 우선 사용자가 명시한 top_k가 있으면 그대로 사용 (1~10 제한)
    if top_k_param is not None:
        return max(1, min(10, top_k_param))

    # 쿼리에서 숫자 추출 (예: "3개", "3가지")
    m = re.search(r"(\d+)\s*(개|가지)?", query_text)
    if m:
        try:
            val = int(m.group(1))
            return max(1, min(10, val))
        except Exception:
            pass

    # 한글 숫자 간단 매핑
    mapping = {
        "한": 1,
        "두": 2,
        "세": 3,
        "네": 4,
        "다섯": 5,
        "여섯": 6,
        "일곱": 7,
        "여덟": 8,
        "아홉": 9,
        "열": 10,
    }
    for k, v in mapping.items():
        if k in query_text:
            return v

    return 1


@router.get("/chat/stream")
async def chat_stream(
    q: str = Query(..., description="사용자 질문"),
    mode: str = Query("tourspot", pattern="^(tourspot|cafe|restaurant)$"),
    top_k: Optional[int] = Query(None, ge=1, le=10),
    history_place_ids: Optional[str] = Query(None, description="CSV 형태의 place_id 목록"),
    session_id: Optional[str] = Query(None, description="대화 스레드/세션 식별자"),
):
    history_ids = _parse_history(history_place_ids)
    resolved_top_k = _infer_top_k(q, top_k)
    initial_state = {
        "query": q,
        "mode": mode,
        "top_k": resolved_top_k,
        "history_place_ids": history_ids,
        "messages": [],
    }
    config = {"configurable": {"thread_id": session_id}} if session_id else {}

    async def event_gen():
        async for update in chat_app.astream(initial_state, config=config):
            delta = update.get("answer")
            if delta:
                yield {"event": "token", "data": delta}
        yield {"event": "done", "data": "ok"}

    return EventSourceResponse(event_gen(), media_type="text/event-stream")
