import os
from typing import List, Optional

import json
from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse

from services.chat_graph import chat_app
from utils.geo import append_node_trace
from models.chat_request import ChatRequest

router = APIRouter()

try:
    from langfuse.callback import CallbackHandler
except Exception:
    try:
        from langfuse.langchain import CallbackHandler
    except Exception:
        CallbackHandler = None


def _build_langfuse_callbacks(session_id: Optional[str] = None) -> List[object] | None:
    if CallbackHandler is None:
        return None
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None
    handler = None
    module_name = getattr(CallbackHandler, "__module__", "")
    if module_name.startswith("langfuse.langchain"):
        # Langfuse v3 LangChain handler reads keys/host from environment.
        try:
            handler = CallbackHandler(update_trace=True)
        except TypeError:
            handler = CallbackHandler(public_key=public_key, update_trace=True)
    else:
        kwargs = {"public_key": public_key, "secret_key": secret_key}
        host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
        if host:
            kwargs["host"] = host
        if session_id:
            kwargs["session_id"] = session_id
        try:
            handler = CallbackHandler(**kwargs)
        except TypeError:
            handler = CallbackHandler(public_key=public_key, secret_key=secret_key)
    if handler is None:
        return None
    return [handler]

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
    callbacks = _build_langfuse_callbacks(session_id)
    initial_state = {
        "query": q,
        "mode": mode,
        "top_k": 10,
        "history_place_ids": history_ids,
        "messages": [],
        "debug": bool(debug),
        "callbacks": callbacks,
    }
    config = {"configurable": {"thread_id": session_id}} if session_id else {}
    if callbacks:
        config["callbacks"] = callbacks

    async def event_gen():
        any_event = False
        final_sent = False
        context_sent = False
        async for event in chat_app.astream_events(initial_state, config=config, version="v2"):
            kind = event.get("event")
            if kind == "on_chain_start":
                node = event.get("name")
                if node:
                    append_node_trace(q, str(node))
                    yield {"event": "node", "data": str(node)}
            if kind == "on_chat_model_stream":
                content = getattr(event["data"].get("chunk"), "content", None)
                if content:
                    any_event = True
                    yield {"event": "token", "data": str(content)}
            elif kind == "on_chain_stream":
                data = event["data"].get("chunk") or {}
                if "debug" in data:
                    yield {"event": "debug", "data": json.dumps(data["debug"], ensure_ascii=False)}
                if "context" in data and not context_sent:
                    context_sent = True
                    yield {"event": "context", "data": json.dumps(data["context"], ensure_ascii=False)}
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
    history_ids = [p.place_id for p in req.history_places]
    callbacks = _build_langfuse_callbacks()
    initial_state = {
        "query": req.query,
        "top_k": 10,
        "history_place_ids": history_ids,
        "messages": [m.model_dump() for m in req.messages],
        "debug": bool(req.debug),
        "context": req.context.model_dump() if req.context else None,
        "preferred_themes": req.preferred_themes,
        "preferred_moods": req.preferred_moods,
        "preferred_restaurant_types": req.preferred_restaurant_types,
        "preferred_cafe_types": req.preferred_cafe_types,
        "avoid": req.avoid,
        "activity_level": req.activity_level,
        "callbacks": callbacks,
    }
    config = {}
    if callbacks:
        config["callbacks"] = callbacks

    async def event_gen():
        any_event = False
        final_sent = False
        context_sent = False
        async for event in chat_app.astream_events(initial_state, config=config, version="v2"):
            kind = event.get("event")
            if kind == "on_chain_start":
                node = event.get("name")
                if node:
                    append_node_trace(req.query, str(node))
                    yield {"event": "node", "data": str(node)}
            if kind == "on_chat_model_stream":
                content = getattr(event["data"].get("chunk"), "content", None)
                if content:
                    any_event = True
                    yield {"event": "token", "data": str(content)}
            elif kind == "on_chain_stream":
                data = event["data"].get("chunk") or {}
                if "debug" in data:
                    yield {"event": "debug", "data": json.dumps(data["debug"], ensure_ascii=False)}
                if "context" in data and not context_sent:
                    context_sent = True
                    yield {"event": "context", "data": json.dumps(data["context"], ensure_ascii=False)}
                if "token" in data:
                    any_event = True
                    yield {"event": "token", "data": str(data["token"])}
                if "final" in data and not final_sent:
                    any_event = True
                    final_sent = True
                    yield {"event": "final", "data": str(data["final"])}
        if not any_event:
            final_state = chat_app.invoke(initial_state, config=config, version="v2")
            final_text = None
            if isinstance(final_state, dict):
                final_text = final_state.get("final") or final_state.get("answer")
            if final_text and not final_sent:
                final_sent = True
                yield {"event": "token", "data": str(final_text)}
                yield {"event": "final", "data": str(final_text)}
        yield {"event": "done", "data": "ok"}

    return EventSourceResponse(event_gen(), media_type="text/event-stream")
