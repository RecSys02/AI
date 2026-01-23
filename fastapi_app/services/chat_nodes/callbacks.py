from functools import lru_cache
from typing import Dict, List, Optional


def build_callbacks_config(callbacks: List[object] | None) -> dict | None:
    """Return a LangChain config dict when callbacks are available."""
    if callbacks:
        return {"callbacks": callbacks}
    return None


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
