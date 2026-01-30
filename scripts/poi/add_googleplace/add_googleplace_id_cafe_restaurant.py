import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "embedding_json"
DEFAULT_FILES = [
    "embedding_cafe.json",
    "embedding_restaurant.json",
]


def _load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GOOGLE_PLACES_KEY")
    if not api_key:
        raise ValueError(".env is missing GOOGLE_PLACES_KEY.")
    return api_key


def _get_lat_lng(item: dict) -> Tuple[Optional[float], Optional[float]]:
    lat = item.get("latitude")
    lng = item.get("longitude")
    if lat is None or lng is None:
        loc = item.get("location")
        if isinstance(loc, dict):
            lat = loc.get("lat")
            lng = loc.get("lng")
    try:
        return float(lat), float(lng)
    except (TypeError, ValueError):
        return None, None


def _get_address(item: dict) -> str:
    address = item.get("address")
    if address:
        return address
    parts = [item.get("city"), item.get("district"), item.get("road")]
    return " ".join(part for part in parts if part)


def _fetch_place(name: str, address: str, lat: float, lng: float, api_key: str, radius: float):
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.rating,places.userRatingCount",
    }
    query = " ".join(part for part in [name, address] if part)
    payload = {
        "textQuery": query,
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": radius,
            }
        },
        "maxResultCount": 1,
    }
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    if response.status_code != 200:
        return None
    results = response.json().get("places", [])
    return results[0] if results else None


def _load_json(path: Path) -> list:
    if not path.exists():
        print(f"Skipping: {path} (not found)")
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data


def _output_path(path: Path, in_place: bool) -> Path:
    if in_place:
        return path
    return path.with_name(path.stem + "_with_ids.json")


def _process_file(
    path: Path,
    api_key: str,
    in_place: bool,
    radius: float,
    sleep_every: int,
    sleep_seconds: float,
) -> None:
    items = _load_json(path)
    if not items:
        return

    total = len(items)
    updated = 0
    today_str = datetime.now().strftime("%Y-%m-%d")

    for index, item in enumerate(items):
        google = item.get("google") if isinstance(item.get("google"), dict) else {}
        if google.get("place_id"):
            continue

        name = item.get("name")
        if not name:
            print(f"[{index + 1}/{total}] Skip: Missing name")
            continue

        address = _get_address(item)
        lat, lng = _get_lat_lng(item)
        if lat is None or lng is None:
            print(f"[{index + 1}/{total}] Skip: Invalid coordinates for {name}")
            continue

        print(f"[{index + 1}/{total}] Searching: {name} ({address})")
        try:
            res = _fetch_place(name, address, lat, lng, api_key, radius)
        except Exception as exc:
            print(f"  -> Error: {exc}")
            res = None

        if res:
            if "google" not in item or not isinstance(item["google"], dict):
                item["google"] = {}
            item["google"].update(
                {
                    "place_id": res.get("id"),
                    "rating": res.get("rating"),
                    "user_ratings_total": res.get("userRatingCount"),
                    "last_updated": today_str,
                }
            )
            updated += 1
            print(f"  -> Found: {res.get('id')}")
        else:
            print("  -> Not found")

        if sleep_every > 0 and (index + 1) % sleep_every == 0:
            time.sleep(sleep_seconds)

    out_path = _output_path(path, in_place)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"{path.name}: updated {updated} items -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--radius", type=float, default=500.0)
    parser.add_argument("--sleep-every", type=int, default=10)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    args = parser.parse_args()

    api_key = _load_api_key()
    data_dir = Path(args.data_dir)
    for filename in args.files:
        _process_file(
            data_dir / filename,
            api_key=api_key,
            in_place=args.in_place,
            radius=args.radius,
            sleep_every=args.sleep_every,
            sleep_seconds=args.sleep_seconds,
        )


if __name__ == "__main__":
    main()
