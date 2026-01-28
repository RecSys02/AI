from functools import lru_cache
import logging
import os
from typing import Dict, List, Optional


def build_callbacks_config(callbacks: List[object] | None) -> dict | None:
    """Return a LangChain config dict when callbacks are available."""
    if callbacks:
        return {"callbacks": callbacks}
    return None


LANGFUSE_STATE_KEYS = (
    "query",
    "normalized_query",
    "intent",
    "mode",
    "mode_detected",
    "mode_unknown",
    "place",
    "place_original",
    "anchor",
    "resolved_name",
    "input_place",
    "last_radius_km",
    "history_place_ids",
    "top_k",
    "messages",
    "context",
    "debug",
    "preferred_themes",
    "preferred_moods",
    "preferred_restaurant_types",
    "preferred_cafe_types",
    "avoid",
    "activity_level",
    "expand_request",
    "expand_failed",
    "anchor_failed",
    "retrievals",
    "final",
    "answer",
)


def build_langfuse_state(state: Dict) -> Dict:
    """Return a slimmed state dict for Langfuse input/output."""
    slim: Dict[str, object] = {}
    for key in LANGFUSE_STATE_KEYS:
        if key not in state:
            continue
        value = state.get(key)
        if value is None:
            continue
        if key == "messages":
            messages = value or []
            trimmed = []
            for msg in messages[-3:]:
                role = str(msg.get("role") or "").strip()
                content = str(msg.get("content") or "").strip()
                if not content:
                    continue
                trimmed.append({"role": role or "user", "content": content})
            if trimmed:
                slim[key] = trimmed
            continue
        if key == "retrievals":
            items = value or []
            slim[key] = _slim_retrievals(items)
            continue
        slim[key] = value
    return slim


def _slim_retrievals(items: List[dict]) -> List[dict]:
    slim: List[dict] = []
    for r in items:
        meta = r.get("meta") or {}
        slim.append(
            {
                "place_id": r.get("place_id"),
                "category": r.get("category"),
                "name": meta.get("name") or meta.get("title"),
                "score": r.get("score"),
            }
        )
    return slim


class FilteredLangfuseCallback:
    """Wrap Langfuse callback to trim large state inputs/outputs."""

    def __init__(self, inner):
        self._inner = inner

    def _slim_inputs(self, payload):
        if isinstance(payload, str):
            return {"query": payload}
        if not isinstance(payload, dict):
            return {}
        if "input" in payload:
            return self._slim_inputs(payload["input"])
        if "inputs" in payload:
            return self._slim_inputs(payload["inputs"])
        if "messages" in payload and "query" not in payload:
            messages = payload.get("messages") or []
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = str(msg.get("content") or "").strip()
                    if content:
                        return {"query": content}
        slim = build_langfuse_state(payload)
        if isinstance(slim, dict) and slim:
            return slim
        if "query" in payload:
            return {"query": payload.get("query") or ""}
        return {}

    def _slim_outputs(self, payload):
        if isinstance(payload, str):
            return {"final": payload}
        if not isinstance(payload, dict):
            return {}
        if "output" in payload:
            return self._slim_outputs(payload["output"])
        if "outputs" in payload:
            return self._slim_outputs(payload["outputs"])
        if "final" in payload:
            return {"final": payload.get("final") or ""}
        if "answer" in payload:
            return {"final": payload.get("answer") or ""}
        if "text" in payload:
            return {"final": payload.get("text") or ""}
        if "content" in payload:
            return {"final": payload.get("content") or ""}
        slim = build_langfuse_state(payload)
        if isinstance(slim, dict) and slim:
            return slim
        return {}

    def on_chain_start(self, serialized, inputs, **kwargs):
        return self._inner.on_chain_start(serialized, self._slim_inputs(inputs), **kwargs)

    def on_chain_end(self, outputs, **kwargs):
        if "inputs" in kwargs:
            kwargs = dict(kwargs)
            kwargs["inputs"] = self._slim_inputs(kwargs.get("inputs"))
        return self._inner.on_chain_end(self._slim_outputs(outputs), **kwargs)

    def on_chain_error(self, error, **kwargs):
        handler = getattr(self._inner, "on_chain_error", None)
        if callable(handler):
            return handler(error, **kwargs)
        return None

    def __getattr__(self, name):
        return getattr(self._inner, name)


def wrap_langfuse_callback(handler):
    return FilteredLangfuseCallback(handler)


def _trace_id_from_obj(obj) -> Optional[str]:
    for name in (
        "trace_id",
        "_trace_id",
        "traceId",
        "traceID",
        "langfuse_trace_id",
        "last_trace_id",
        "lastTraceId",
    ):
        trace_id = getattr(obj, name, None)
        if trace_id:
            return str(trace_id)
    getter = getattr(obj, "get_trace_id", None)
    if callable(getter):
        try:
            trace_id = getter()
        except Exception:
            trace_id = None
        if trace_id:
            return str(trace_id)
    for trace_attr in ("trace", "_trace", "root_trace"):
        trace = getattr(obj, trace_attr, None)
        if trace is not None:
            trace_id = getattr(trace, "id", None)
            if trace_id:
                return str(trace_id)
    return None


def _extract_langfuse_trace_id(callbacks: List[object] | None) -> Optional[str]:
    if not callbacks:
        return None
    for cb in callbacks:
        trace_id = _trace_id_from_obj(cb)
        if trace_id:
            return trace_id
        inner = getattr(cb, "_inner", None)
        if inner is not None:
            trace_id = _trace_id_from_obj(inner)
            if trace_id:
                return trace_id
    return None


def _extract_langfuse_client(callbacks: List[object] | None):
    if not callbacks:
        return None
    for cb in callbacks:
        for name in ("langfuse", "client", "_client", "_langfuse"):
            client = getattr(cb, name, None)
            if client is not None:
                return client
        inner = getattr(cb, "_inner", None)
        if inner is not None:
            for name in ("langfuse", "client", "_client", "_langfuse"):
                client = getattr(inner, name, None)
                if client is not None:
                    return client
    return None


def _unwrap_langfuse_client(client):
    if client is None:
        return None
    if hasattr(client, "trace"):
        return client
    for name in ("client", "_client", "api", "_api"):
        inner = getattr(client, name, None)
        if inner is not None and hasattr(inner, "trace"):
            return inner
    return client


def _trace_client_methods(client, keyword: str = "trace") -> List[str]:
    methods = []
    if client is None:
        return methods
    for name in dir(client):
        if keyword in name.lower():
            methods.append(name)
    return methods


def _call_trace_handle_method(handle, method_name: str, trace_id: str, update_payload: Dict[str, object]) -> bool:
    method = getattr(handle, method_name, None)
    if not callable(method):
        return False
    try:
        method(id=trace_id, **update_payload)
        return True
    except TypeError:
        try:
            method(trace_id=trace_id, **update_payload)
            return True
        except Exception as exc:
            _log_langfuse_debug("trace_handle_method_error", method=method_name, error=repr(exc))
    except Exception as exc:
        _log_langfuse_debug("trace_handle_method_error", method=method_name, error=repr(exc))
    return False


@lru_cache(maxsize=1)
def _get_langfuse_client():
    try:
        from langfuse import Langfuse
    except Exception:
        return None
    try:
        return Langfuse()
    except Exception:
        return None


def record_langfuse_timings(callbacks: List[object] | None, timings: Dict[str, float]) -> None:
    if not timings:
        return
    trace_id = _extract_langfuse_trace_id(callbacks)
    if not trace_id:
        return
    client = _get_langfuse_client()
    if client is None:
        return
    payload = {"retrieve_timings_ms": timings}
    # Best-effort span creation to keep timings in the same trace.
    try:
        if hasattr(client, "span"):
            span = client.span(name="retrieve_timings", trace_id=trace_id, metadata=payload)
            if hasattr(span, "end"):
                try:
                    span.end(metadata=payload)
                except TypeError:
                    span.end()
            return
    except Exception:
        pass
    try:
        if hasattr(client, "trace"):
            try:
                trace = client.trace(id=trace_id)
            except TypeError:
                trace = client.trace(trace_id=trace_id)
            if trace is not None and hasattr(trace, "update"):
                trace.update(metadata=payload)
    except Exception:
        return


def build_langfuse_output(payload: Dict) -> Dict:
    """Return a compact output payload for Langfuse trace output."""
    if not isinstance(payload, dict):
        return {}
    final = payload.get("final") or payload.get("answer") or payload.get("text") or payload.get("content")
    result: Dict[str, object] = {}
    if final:
        result["final"] = final
    if payload.get("context") is not None:
        result["context"] = payload.get("context")
    if payload.get("retrievals") is not None:
        result["retrievals"] = _slim_retrievals(payload.get("retrievals") or [])
    return result


def update_langfuse_trace(callbacks: List[object] | None, input_state: Dict | None = None, output: Dict | None = None) -> None:
    update_payload: Dict[str, object] = {}
    if input_state:
        slim_input = build_langfuse_state(input_state)
        if slim_input:
            update_payload["input"] = {"query": slim_input.get("query")} if slim_input.get("query") else slim_input
    if output:
        slim_output = build_langfuse_output(output)
        if slim_output:
            update_payload["output"] = {"final": slim_output.get("final")} if slim_output.get("final") else slim_output
    if not update_payload:
        return

    _log_langfuse_debug(
        "update_start",
        update_keys=list(update_payload.keys()),
        input=_preview_value(update_payload.get("input")),
        output=_preview_value(update_payload.get("output")),
    )

    # Prefer updating trace so the main trace list shows the final input/output.
    if callbacks:
        for cb in callbacks:
            trace = getattr(cb, "trace", None)
            if trace is not None and hasattr(trace, "update"):
                try:
                    trace.update(**update_payload)
                    _log_langfuse_debug("update_trace_cb", trace_id=getattr(trace, "id", None))
                    return
                except TypeError:
                    try:
                        trace.update(update_payload)
                        _log_langfuse_debug("update_trace_cb", trace_id=getattr(trace, "id", None))
                        return
                    except Exception as exc:
                        _log_langfuse_debug("update_trace_cb_error", error=repr(exc))
                except Exception as exc:
                    _log_langfuse_debug("update_trace_cb_error", error=repr(exc))

    trace_id = _extract_langfuse_trace_id(callbacks)
    client = _unwrap_langfuse_client(_extract_langfuse_client(callbacks) or _get_langfuse_client())
    _log_langfuse_debug(
        "trace_lookup",
        trace_id=trace_id,
        client_type=type(client).__name__ if client is not None else None,
        callback_types=[type(cb).__name__ for cb in callbacks] if callbacks else None,
        inner_types=[type(getattr(cb, "_inner", None)).__name__ for cb in callbacks] if callbacks else None,
    )
    if client is not None and _langfuse_debug_enabled():
        _log_langfuse_debug("trace_client_methods", methods=_trace_client_methods(client))
    if trace_id is None and callbacks and _langfuse_debug_enabled():
        for cb in callbacks:
            _log_langfuse_debug(
                "trace_attrs_cb",
                cb_type=type(cb).__name__,
                attrs=_trace_attr_snapshot(cb),
            )
            inner = getattr(cb, "_inner", None)
            if inner is not None:
                _log_langfuse_debug(
                    "trace_attrs_inner",
                    cb_type=type(inner).__name__,
                    attrs=_trace_attr_snapshot(inner),
                )
    trace = None
    if client is not None and trace_id:
        trace_handle = getattr(client, "trace", None)
        if trace_handle is not None and not callable(trace_handle):
            _log_langfuse_debug(
                "trace_handle_noncallable",
                handle_type=type(trace_handle).__name__,
            )
            if _langfuse_debug_enabled():
                _log_langfuse_debug(
                    "trace_handle_methods",
                    methods=_trace_client_methods(trace_handle, keyword=""),
                )
            for method_name in ("update", "patch", "upsert", "create"):
                if _call_trace_handle_method(trace_handle, method_name, trace_id, update_payload):
                    _log_langfuse_debug("update_trace_handle", method=method_name, trace_id=trace_id)
                    return
            if hasattr(trace_handle, "update"):
                try:
                    trace_handle.update(id=trace_id, **update_payload)
                    _log_langfuse_debug("update_trace_handle", trace_id=trace_id)
                    return
                except TypeError:
                    try:
                        trace_handle.update(trace_id=trace_id, **update_payload)
                        _log_langfuse_debug("update_trace_handle", trace_id=trace_id)
                        return
                    except Exception as exc:
                        _log_langfuse_debug("update_trace_handle_error", error=repr(exc))
                except Exception as exc:
                    _log_langfuse_debug("update_trace_handle_error", error=repr(exc))
            if hasattr(trace_handle, "get"):
                try:
                    trace = trace_handle.get(id=trace_id)
                except TypeError:
                    try:
                        trace = trace_handle.get(trace_id=trace_id)
                    except Exception:
                        trace = None
                except Exception:
                    trace = None
        if not hasattr(client, "trace"):
            for method_name in ("update_trace", "trace_update", "create_trace", "trace_create"):
                method = getattr(client, method_name, None)
                if not callable(method):
                    continue
                try:
                    method(id=trace_id, **update_payload)
                    _log_langfuse_debug("update_trace_client_method", method=method_name)
                    return
                except TypeError:
                    try:
                        method(trace_id=trace_id, **update_payload)
                        _log_langfuse_debug("update_trace_client_method", method=method_name)
                        return
                    except Exception as exc:
                        _log_langfuse_debug("update_trace_client_error", error=repr(exc), method=method_name)
                except Exception as exc:
                    _log_langfuse_debug("update_trace_client_error", error=repr(exc), method=method_name)
        try:
            if callable(trace_handle):
                trace = trace_handle(id=trace_id)
        except TypeError:
            try:
                if callable(trace_handle):
                    trace = trace_handle(trace_id=trace_id)
            except Exception:
                trace = None
        except Exception:
            trace = None
    if trace is None and client is not None and trace_id and callable(getattr(client, "trace", None)):
        try:
            trace = client.trace(id=trace_id, **update_payload)
            _log_langfuse_debug("trace_obj_created", trace_id=trace_id)
        except TypeError:
            try:
                trace = client.trace(trace_id=trace_id, **update_payload)
                _log_langfuse_debug("trace_obj_created", trace_id=trace_id)
            except Exception as exc:
                _log_langfuse_debug("trace_obj_create_error", error=repr(exc))
        except Exception as exc:
            _log_langfuse_debug("trace_obj_create_error", error=repr(exc))
    if trace is None:
        _log_langfuse_debug("trace_obj_missing", trace_id=trace_id)
    if trace is not None and hasattr(trace, "update"):
        try:
            trace.update(**update_payload)
            _log_langfuse_debug("update_trace_client", trace_id=getattr(trace, "id", None) or trace_id)
            return
        except TypeError:
            try:
                trace.update(update_payload)
                _log_langfuse_debug("update_trace_client", trace_id=getattr(trace, "id", None) or trace_id)
                return
            except Exception as exc:
                _log_langfuse_debug("update_trace_client_error", error=repr(exc))
        except Exception as exc:
            _log_langfuse_debug("update_trace_client_error", error=repr(exc))

    # Fallback: update span/observation when trace updates are not available.
    if callbacks:
        for cb in callbacks:
            for attr in ("root_span", "current_span", "span", "observation", "_root_span", "_current_span", "_span"):
                span = getattr(cb, attr, None)
                if span is not None and hasattr(span, "update"):
                    try:
                        span.update(**update_payload)
                        _log_langfuse_debug("update_span_cb", span_attr=attr)
                        return
                    except TypeError:
                        try:
                            span.update(update_payload)
                            _log_langfuse_debug("update_span_cb", span_attr=attr)
                            return
                        except Exception as exc:
                            _log_langfuse_debug("update_span_cb_error", error=repr(exc), span_attr=attr)
                    except Exception as exc:
                        _log_langfuse_debug("update_span_cb_error", error=repr(exc), span_attr=attr)
    _log_langfuse_debug("update_failed", trace_id=trace_id)
logger = logging.getLogger("uvicorn.error")


def _langfuse_debug_enabled() -> bool:
    return os.getenv("LANGFUSE_DEBUG") in {"1", "true", "True"}


def _preview_value(value, limit: int = 200):
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _log_langfuse_debug(message: str, **kwargs) -> None:
    if not _langfuse_debug_enabled():
        return
    safe_payload = {}
    for key, value in kwargs.items():
        if isinstance(value, dict):
            safe_payload[key] = {k: _preview_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            safe_payload[key] = [_preview_value(v) for v in value]
        else:
            safe_payload[key] = _preview_value(value)
    logger.info("langfuse_debug %s %s", message, safe_payload)


def _trace_attr_snapshot(obj) -> Dict[str, object]:
    snapshot: Dict[str, object] = {}
    for name in dir(obj):
        if "trace" in name.lower() or "span" in name.lower() or "observ" in name.lower():
            try:
                value = getattr(obj, name)
            except Exception:
                value = "<error>"
            if callable(value):
                snapshot[name] = "<callable>"
            else:
                snapshot[name] = _preview_value(value)
    return snapshot
