import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List

GOOGLE_PLACES_KEY = os.getenv("GOOGLE_PLACES_KEY")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTOCOMPLETE_ERROR_LOG = PROJECT_ROOT / "data" / "locations" / "places_errors.txt"


def autocomplete_places(
    input_text: str,
    limit: int = 5,
    location: str = "37.5665,126.9780",
    radius: int = 30000,
    types: str = "(regions)",
) -> List[dict]:
    if not input_text or not GOOGLE_PLACES_KEY:
        return []
    params = urllib.parse.urlencode(
        {
            "input": input_text,
            "key": GOOGLE_PLACES_KEY,
            "language": "ko",
            "components": "country:kr",
            "location": location,
            "radius": str(radius),
            "types": types,
        }
    )
    url = f"https://maps.googleapis.com/maps/api/place/autocomplete/json?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        _log_error(input_text, "autocomplete_request_failed")
        return []
    status = payload.get("status")
    if status and status != "OK":
        _log_error(input_text, f"autocomplete_status_{status}")
        return []
    preds = payload.get("predictions") or []
    trimmed = preds[: max(1, limit)]
    out = []
    for item in trimmed:
        out.append(
            {
                "description": item.get("description") or "",
                "place_id": item.get("place_id") or "",
                "types": item.get("types") or [],
            }
        )
    return out


def _log_error(query: str, reason: str) -> None:
    AUTOCOMPLETE_ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with AUTOCOMPLETE_ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {reason} query={query}\n")
