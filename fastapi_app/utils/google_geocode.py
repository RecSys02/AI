import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEOCODE_ERROR_LOG = PROJECT_ROOT / "data" / "locations" / "geocode_errors.txt"

GOOGLE_GEOCODE_KEY = os.getenv("GOOGLE_GEOCODE_KEY")


def geocode_address(query: str) -> Optional[dict]:
    if not query or not GOOGLE_GEOCODE_KEY:
        return None
    params = urllib.parse.urlencode(
        {
            "address": query,
            "key": GOOGLE_GEOCODE_KEY,
            "language": "ko",
        }
    )
    url = f"https://maps.googleapis.com/maps/api/geocode/json?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        _log_error(query, "request_failed")
        return None
    status = payload.get("status")
    if status != "OK":
        _log_error(query, f"status_{status}")
        return None
    results = payload.get("results") or []
    if not results:
        _log_error(query, "no_results")
        return None
    first = results[0]
    loc = (first.get("geometry") or {}).get("location") or {}
    try:
        lat = float(loc.get("lat"))
        lng = float(loc.get("lng"))
    except Exception:
        _log_error(query, "invalid_lat_lng")
        return None
    return {
        "lat": lat,
        "lng": lng,
        "address": first.get("formatted_address") or "",
    }


def geocode_place_id(place_id: str) -> Optional[dict]:
    if not place_id or not GOOGLE_GEOCODE_KEY:
        return None
    params = urllib.parse.urlencode(
        {
            "place_id": place_id,
            "key": GOOGLE_GEOCODE_KEY,
            "language": "ko",
        }
    )
    url = f"https://maps.googleapis.com/maps/api/geocode/json?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        _log_error(place_id, "place_id_request_failed")
        return None
    status = payload.get("status")
    if status != "OK":
        _log_error(place_id, f"place_id_status_{status}")
        return None
    results = payload.get("results") or []
    if not results:
        _log_error(place_id, "place_id_no_results")
        return None
    first = results[0]
    loc = (first.get("geometry") or {}).get("location") or {}
    try:
        lat = float(loc.get("lat"))
        lng = float(loc.get("lng"))
    except Exception:
        _log_error(place_id, "place_id_invalid_lat_lng")
        return None
    return {
        "lat": lat,
        "lng": lng,
        "address": first.get("formatted_address") or "",
    }


def _log_error(query: str, reason: str) -> None:
    GEOCODE_ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with GEOCODE_ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {reason} query={query}\n")
