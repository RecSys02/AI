from functools import lru_cache
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


def _extract_langfuse_trace_id(callbacks: List[object] | None) -> Optional[str]:
    if not callbacks:
        return None
    for cb in callbacks:
        trace_id = getattr(cb, "trace_id", None) or getattr(cb, "traceId", None)
        if trace_id:
            return str(trace_id)
        getter = getattr(cb, "get_trace_id", None)
        if callable(getter):
            try:
                trace_id = getter()
            except Exception:
                trace_id = None
            if trace_id:
                return str(trace_id)
        trace = getattr(cb, "trace", None)
        if trace is not None:
            trace_id = getattr(trace, "id", None)
            if trace_id:
                return str(trace_id)
    return None


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

    # Prefer updating trace so the main trace list shows the final input/output.
    if callbacks:
        for cb in callbacks:
            trace = getattr(cb, "trace", None)
            if trace is not None and hasattr(trace, "update"):
                try:
                    trace.update(**update_payload)
                    return
                except TypeError:
                    try:
                        trace.update(update_payload)
                        return
                    except Exception:
                        pass

    trace_id = _extract_langfuse_trace_id(callbacks)
    if not trace_id:
        trace_id = None
    client = _get_langfuse_client()
    if client is None:
        client = None
    trace = None
    if client is not None and trace_id:
        try:
            trace = client.trace(id=trace_id)
        except TypeError:
            try:
                trace = client.trace(trace_id=trace_id)
            except Exception:
                trace = None
        except Exception:
            trace = None
    if trace is not None and hasattr(trace, "update"):
        try:
            trace.update(**update_payload)
            return
        except TypeError:
            try:
                trace.update(update_payload)
                return
            except Exception:
                pass

    # Fallback: update span/observation when trace updates are not available.
    if callbacks:
        for cb in callbacks:
            for attr in ("root_span", "current_span", "span", "observation", "_root_span", "_current_span", "_span"):
                span = getattr(cb, attr, None)
                if span is not None and hasattr(span, "update"):
                    try:
                        span.update(**update_payload)
                        return
                    except TypeError:
                        try:
                            span.update(update_payload)
                            return
                        except Exception:
                            pass
