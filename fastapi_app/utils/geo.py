import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCATIONS_DIR = PROJECT_ROOT / "data" / "locations"

KEYWORD_ALIASES_PATH = LOCATIONS_DIR / "keyword_aliases.json"
ADMIN_ALIASES_PATH = LOCATIONS_DIR / "admin_aliases.json"
ANCHOR_CACHE_PATH = LOCATIONS_DIR / "anchor_cache.json"
GEO_CENTERS_PATH = LOCATIONS_DIR / "geo_centers.json"
PLACE_DEBUG_PATH = LOCATIONS_DIR / "place_debug.jsonl"
NODE_TRACE_PATH = LOCATIONS_DIR / "node_trace.txt"
NODE_TRACE_JSONL_PATH = LOCATIONS_DIR / "node_trace.jsonl"


def normalize_text(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9가-힣]", "", lowered)
    return cleaned


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_alias_map(data: Dict[str, Iterable[str]]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for canonical, aliases in data.items():
        canon_norm = normalize_text(canonical)
        if canon_norm:
            alias_map[canon_norm] = canonical
        if not aliases:
            continue
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if alias_norm:
                alias_map[alias_norm] = canonical
    return alias_map


def add_alias(data: Dict[str, list], canonical: str, alias: str) -> bool:
    if not canonical or not alias:
        return False
    if canonical not in data:
        data[canonical] = []
    if alias in data[canonical]:
        return False
    data[canonical].append(alias)
    return True


def load_keyword_aliases() -> Dict[str, list]:
    return load_json(KEYWORD_ALIASES_PATH, {})


def load_admin_aliases() -> Dict[str, list]:
    return load_json(ADMIN_ALIASES_PATH, {})


def load_anchor_cache() -> Dict[str, dict]:
    return load_json(ANCHOR_CACHE_PATH, {})


def load_geo_centers() -> Dict[str, dict]:
    return load_json(GEO_CENTERS_PATH, {})


def save_admin_aliases(data: Dict[str, list]) -> None:
    save_json(ADMIN_ALIASES_PATH, data)


def save_keyword_aliases(data: Dict[str, list]) -> None:
    save_json(KEYWORD_ALIASES_PATH, data)


def save_anchor_cache(data: Dict[str, dict]) -> None:
    save_json(ANCHOR_CACHE_PATH, data)


def append_place_debug(entry: dict) -> None:
    PLACE_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PLACE_DEBUG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def append_node_trace(query: str, node: str) -> None:
    NODE_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NODE_TRACE_PATH.open("a", encoding="utf-8") as f:
        f.write(f"query={query}\tnode={node}\n")


def append_node_trace_result(query: str, node: str, data: dict) -> None:
    NODE_TRACE_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NODE_TRACE_JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"query": query, "node": node, "data": data}, ensure_ascii=False) + "\n")


def resolve_alias(value: str, alias_map: Dict[str, str]) -> str:
    if not value:
        return value
    key = normalize_text(value)
    return alias_map.get(key, value)


def get_lat_lng(meta: dict):
    lat = meta.get("lat") or meta.get("latitude")
    lng = meta.get("lng") or meta.get("lon") or meta.get("longitude")
    if (lat is None or lng is None) and isinstance(meta.get("location"), dict):
        loc = meta["location"]
        lat = loc.get("lat") or loc.get("latitude") or lat
        lng = loc.get("lng") or loc.get("lon") or loc.get("longitude") or lng
    try:
        return float(lat), float(lng)
    except Exception:
        return None, None


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def distance_to_centers_km(lat: float, lng: float, centers: Iterable[Iterable[float]]) -> Optional[float]:
    if not centers:
        return None
    lat1 = math.radians(lat)
    lon1 = math.radians(lng)
    min_km = None
    for center in centers:
        if not center or len(center) != 2:
            continue
        lat2 = math.radians(center[0])
        lon2 = math.radians(center[1])
        dist = haversine_km(lat1, lon1, lat2, lon2)
        if min_km is None or dist < min_km:
            min_km = dist
    return min_km
