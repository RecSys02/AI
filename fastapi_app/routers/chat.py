import asyncio
import json
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, Query, Request
from sse_starlette.sse import EventSourceResponse

from services.chat_graph import chat_app
from services.chat_nodes.callbacks import update_langfuse_trace, wrap_langfuse_callback
from utils.geo import append_node_trace
from models.chat_request import ChatRequest

router = APIRouter()
logger = logging.getLogger("uvicorn.error")

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
        kwargs = {"public_key": public_key, "secret_key": secret_key, "update_trace": True}
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
    return [wrap_langfuse_callback(handler)]

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


def _preview_text(value: Optional[str], limit: int = 200) -> str:
    if not value:
        return ""
    cleaned = value.replace("\n", " ").replace("\r", " ")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "...(truncated)"


def _is_langgraph_node_event(event: dict) -> bool:
    metadata = event.get("metadata") or {}
    node_name = metadata.get("langgraph_node")
    return bool(node_name and node_name == event.get("name"))


@router.get("/chat/stream")
async def chat_stream(
    request: Request,
    q: str = Query(..., description="사용자 질문"),
    mode: Optional[str] = Query(None, description="카테고리(tourspot/cafe/restaurant). 미지정 시 기본값 사용"),
    top_k: Optional[int] = Query(None, ge=1, le=10),
    history_place_ids: Optional[str] = Query(None, description="CSV 형태의 place_id 목록"),
    session_id: Optional[str] = Query(None, description="대화 스레드/세션 식별자"),
    debug: Optional[bool] = Query(False, description="디버그(점수) 포함 여부"),
):
    req_id = os.urandom(4).hex()
    client_host = request.client.host if request.client else None
    forwarded_for = request.headers.get("x-forwarded-for")
    logger.info(
        "chat_stream start req_id=%s client=%s xff=%s session_id=%s mode=%s top_k=%s debug=%s q=%s",
        req_id,
        client_host,
        forwarded_for,
        session_id,
        mode,
        top_k,
        bool(debug),
        _preview_text(q),
    )
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
        cancelled = False
        final_text = None
        final_context = None
        token_parts: List[str] = []
        try:
            async for event in chat_app.astream_events(initial_state, config=config, version="v2"):
                kind = event.get("event")
                if kind == "on_chain_start" and _is_langgraph_node_event(event):
                    node = event.get("name")
                    if node:
                        append_node_trace(q, str(node))
                        yield {"event": "node", "data": str(node)}
                if kind == "on_chain_stream" and _is_langgraph_node_event(event):
                    data = event["data"].get("chunk") or {}
                    if "debug" in data:
                        yield {"event": "debug", "data": json.dumps(data["debug"], ensure_ascii=False)}
                    if "context" in data and not context_sent:
                        context_sent = True
                        final_context = data["context"]
                        yield {"event": "context", "data": json.dumps(data["context"], ensure_ascii=False)}
                    if "token" in data:
                        any_event = True
                        token = str(data["token"])
                        if not final_text:
                            token_parts.append(token)
                        yield {"event": "token", "data": token}
                    if "final" in data and not final_sent:
                        any_event = True
                        final_sent = True
                        final_text = str(data["final"])
                        yield {"event": "final", "data": str(data["final"])}
            if not any_event:
                final_state = chat_app.invoke(initial_state, config=config)
                final_text = None
                if isinstance(final_state, dict):
                    final_text = final_state.get("final") or final_state.get("answer")
                    final_context = final_state.get("context") or final_context
                if final_text and not final_sent:
                    final_sent = True
                    yield {"event": "token", "data": str(final_text)}
                    yield {"event": "final", "data": str(final_text)}
            if not final_text and token_parts:
                final_text = "".join(token_parts)
            yield {"event": "done", "data": "ok"}
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            output_payload = {}
            if final_text:
                output_payload["final"] = final_text
            if final_context:
                output_payload["context"] = final_context
            update_langfuse_trace(callbacks, input_state=initial_state, output=output_payload or None)
            logger.info(
                "chat_stream done req_id=%s any_event=%s final_sent=%s cancelled=%s",
                req_id,
                any_event,
                final_sent,
                cancelled,
            )


    return EventSourceResponse(event_gen(), media_type="text/event-stream")


@router.post("/chat/stream")
async def chat_stream_post(req: ChatRequest, request: Request):
    req_id = os.urandom(4).hex()
    client_host = request.client.host if request.client else None
    forwarded_for = request.headers.get("x-forwarded-for")
    logger.info(
        "chat_stream_post start req_id=%s client=%s xff=%s debug=%s q=%s",
        req_id,
        client_host,
        forwarded_for,
        bool(req.debug),
        _preview_text(req.query),
    )
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
        cancelled = False
        final_text = None
        final_context = None
        token_parts: List[str] = []
        try:
            async for event in chat_app.astream_events(initial_state, config=config, version="v2"):
                kind = event.get("event")
                if kind == "on_chain_start" and _is_langgraph_node_event(event):
                    node = event.get("name")
                    if node:
                        append_node_trace(req.query, str(node))
                        yield {"event": "node", "data": str(node)}
                if kind == "on_chain_stream" and _is_langgraph_node_event(event):
                    data = event["data"].get("chunk") or {}
                    if "debug" in data:
                        yield {"event": "debug", "data": json.dumps(data["debug"], ensure_ascii=False)}
                    if "context" in data and not context_sent:
                        context_sent = True
                        final_context = data["context"]
                        yield {"event": "context", "data": json.dumps(data["context"], ensure_ascii=False)}
                    if "token" in data:
                        any_event = True
                        token = str(data["token"])
                        if not final_text:
                            token_parts.append(token)
                        yield {"event": "token", "data": token}
                    if "final" in data and not final_sent:
                        any_event = True
                        final_sent = True
                        final_text = str(data["final"])
                        yield {"event": "final", "data": str(data["final"])}
            if not any_event:
                final_state = chat_app.invoke(initial_state, config=config, version="v2")
                final_text = None
                if isinstance(final_state, dict):
                    final_text = final_state.get("final") or final_state.get("answer")
                    final_context = final_state.get("context") or final_context
                if final_text and not final_sent:
                    final_sent = True
                    yield {"event": "token", "data": str(final_text)}
                    yield {"event": "final", "data": str(final_text)}
            if not final_text and token_parts:
                final_text = "".join(token_parts)
            yield {"event": "done", "data": "ok"}
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            output_payload = {}
            if final_text:
                output_payload["final"] = final_text
            if final_context:
                output_payload["context"] = final_context
            update_langfuse_trace(callbacks, input_state=initial_state, output=output_payload or None)
            logger.info(
                "chat_stream_post done req_id=%s any_event=%s final_sent=%s cancelled=%s",
                req_id,
                any_event,
                final_sent,
                cancelled,
            )

    return EventSourceResponse(event_gen(), media_type="text/event-stream")
