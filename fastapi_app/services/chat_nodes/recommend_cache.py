import copy
import hashlib
import json
import os
import time
from collections import OrderedDict
from typing import Any


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_CACHE_TTL_SEC = max(1, _env_int("RECOMMEND_CACHE_TTL_SEC", 300))
_CACHE_MAX_ITEMS = max(1, _env_int("RECOMMEND_CACHE_MAX_ITEMS", 256))


def _anchor_signature(anchor: dict | None) -> dict:
    if not isinstance(anchor, dict):
        return {}
    centers = []
    for pair in anchor.get("centers") or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            centers.append([round(float(pair[0]), 6), round(float(pair[1]), 6)])
        except (TypeError, ValueError):
            continue
    radius_by_intent = {}
    for key, value in (anchor.get("radius_by_intent") or {}).items():
        try:
            radius_by_intent[str(key)] = round(float(value), 3)
        except (TypeError, ValueError):
            continue
    return {
        "centers": centers,
        "radius_by_intent": radius_by_intent,
        "source": anchor.get("source"),
    }


def build_recommend_cache_key(state: dict) -> str:
    payload = {
        "route_intent": state.get("route_intent"),
        "query": str(state.get("normalized_query") or state.get("query") or "").strip(),
        "mode": state.get("mode") or state.get("explicit_mode"),
        "resolved_name": state.get("resolved_name"),
        "place": state.get("place") or {},
        "anchor": _anchor_signature(state.get("anchor")),
        "top_k": state.get("top_k"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RecommendCache:
    def __init__(self):
        self._items: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [
            key
            for key, (stored_at, _value) in self._items.items()
            if (now - stored_at) > _CACHE_TTL_SEC
        ]
        for key in expired:
            self._items.pop(key, None)

    def get(self, key: str) -> dict[str, Any] | None:
        self._purge_expired()
        hit = self._items.get(key)
        if hit is None:
            return None
        stored_at, value = hit
        self._items.move_to_end(key)
        return copy.deepcopy({"stored_at": stored_at, **value})

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._purge_expired()
        self._items[key] = (time.monotonic(), copy.deepcopy(value))
        self._items.move_to_end(key)
        while len(self._items) > _CACHE_MAX_ITEMS:
            self._items.popitem(last=False)


_CACHE = RecommendCache()


def get_recommend_cache() -> RecommendCache:
    return _CACHE
