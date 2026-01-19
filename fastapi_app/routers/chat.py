from typing import List, Optional

import json
from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse

from services.chat_graph import chat_app
from models.chat_request import ChatRequest

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


@router.get("/chat/stream")
async def chat_stream(
    q: str = Query(..., description="사용자 질문"),
    mode: Optional[str] = Query(None, description="카테고리(tourspot/cafe/restaurant). 미지정 시 기본값 사용"),
    top_k: Optional[int] = Query(None, ge=1, le=10),
    history_place_ids: Optional[str] = Query(None, description="CSV 형태의 place_id 목록"),
    session_id: Optional[str] = Query(None, description="대화 스레드/세션 식별자"),
    debug: Optional[bool] = Query(False, description="디버그(점수) 포함 여부"),
):
    history_ids = _parse_history(history_place_ids)
    initial_state = {
        "query": q,
        "mode": mode,
        "top_k": max(1, min(10, top_k)) if top_k is not None else None,
        "history_place_ids": history_ids,
        "messages": [],
        "debug": bool(debug),
    }
    config = {"configurable": {"thread_id": session_id}} if session_id else {}

    async def event_gen():
        any_event = False
        final_sent = False
        async for event in chat_app.astream_events(initial_state, config=config, version="v2"):
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                content = getattr(event["data"].get("chunk"), "content", None)
                if content:
                    any_event = True
                    yield {"event": "token", "data": str(content)}
            elif kind == "on_chain_stream":
                data = event["data"].get("chunk") or {}
                if "debug" in data:
                    yield {"event": "debug", "data": json.dumps(data["debug"], ensure_ascii=False)}
                if "token" in data:
                    any_event = True
                    yield {"event": "token", "data": str(data["token"])}
                if "final" in data and not final_sent:
                    any_event = True
                    final_sent = True
                    yield {"event": "final", "data": str(data["final"])}
        if not any_event:
            final_state = chat_app.invoke(initial_state, config=config)
            final_text = None
            if isinstance(final_state, dict):
                final_text = final_state.get("final") or final_state.get("answer")
            if final_text and not final_sent:
                final_sent = True
                yield {"event": "token", "data": str(final_text)}
                yield {"event": "final", "data": str(final_text)}
        yield {"event": "done", "data": "ok"}


    return EventSourceResponse(event_gen(), media_type="text/event-stream")


@router.post("/chat/stream")
async def chat_stream_post(req: ChatRequest):
    top_k = req.top_k
    normalized_top_k = max(1, min(10, top_k)) if top_k is not None else None
    history_ids = [p.place_id for p in req.history_places]
    initial_state = {
        "query": req.query,
        "top_k": normalized_top_k,
        "history_place_ids": history_ids,
        "messages": [m.model_dump() for m in req.messages],
        "debug": bool(req.debug),
        "preferred_themes": req.preferred_themes,
        "preferred_moods": req.preferred_moods,
        "preferred_restaurant_types": req.preferred_restaurant_types,
        "preferred_cafe_types": req.preferred_cafe_types,
        "avoid": req.avoid,
        "activity_level": req.activity_level,
    }

    async def event_gen():
        any_event = False
        final_sent = False
        async for event in chat_app.astream_events(initial_state, version="v2"):
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                content = getattr(event["data"].get("chunk"), "content", None)
                if content:
                    any_event = True
                    yield {"event": "token", "data": str(content)}
            elif kind == "on_chain_stream":
                data = event["data"].get("chunk") or {}
                if "debug" in data:
                    yield {"event": "debug", "data": json.dumps(data["debug"], ensure_ascii=False)}
                if "token" in data:
                    any_event = True
                    yield {"event": "token", "data": str(data["token"])}
                if "final" in data and not final_sent:
                    any_event = True
                    final_sent = True
                    yield {"event": "final", "data": str(data["final"])}
        if not any_event:
            final_state = chat_app.invoke(initial_state, version="v2")
            final_text = None
            if isinstance(final_state, dict):
                final_text = final_state.get("final") or final_state.get("answer")
            if final_text and not final_sent:
                final_sent = True
                yield {"event": "token", "data": str(final_text)}
                yield {"event": "final", "data": str(final_text)}
        yield {"event": "done", "data": "ok"}

    return EventSourceResponse(event_gen(), media_type="text/event-stream")
