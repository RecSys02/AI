#!/usr/bin/env python3
# uv run python ../scripts/generate_candidates.py --radius-km 3.0 --max-per-mode 50

import argparse
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_JSON_DIR = PROJECT_ROOT / "data" / "embedding_json"


def _get_lat_lng(meta: dict):
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


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * math.sin(dlon / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def _normalize_keywords(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    if isinstance(raw, str) and raw:
        return [raw]
    return []


def _load_mode(mode: str) -> list[dict]:
    path = EMBEDDING_JSON_DIR / f"embedding_{mode}.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    candidates = []
    for item in data:
        pid = item.get("place_id")
        if pid is None:
            continue
        lat, lng = _get_lat_lng(item)
        if lat is None or lng is None:
            continue
        candidates.append(
            {
                "place_id": int(pid),
                "lat": lat,
                "lng": lng,
                "province": item.get("province") or item.get("city") or "unknown",
                "name": item.get("name") or item.get("title") or "",
                "category": item.get("category") or mode,
                "content": item.get("content"),
                "keywords": _normalize_keywords(
                    item.get("keywords") or item.get("sicksin_keywords")
                ),
                "themes": item.get("themes"),
                "type": item.get("type"),
                "poi_type": item.get("poi_type"),
            }
        )
    return candidates


def _load_users(path: Path) -> list[dict]:
    users = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            users.append(json.loads(line))
    return users


def _build_anchor_lookup(candidates_by_mode: dict[str, list[dict]]) -> dict[int, tuple[float, float]]:
    lookup: dict[int, tuple[float, float]] = {}
    for candidates in candidates_by_mode.values():
        for item in candidates:
            pid = item["place_id"]
            if pid not in lookup:
                lookup[pid] = (item["lat"], item["lng"])
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-user POI candidates within radius (JSONL)."
    )
    parser.add_argument(
        "--users",
        default=str(PROJECT_ROOT / "data" / "eval" / "users_5000.jsonl"),
        help="Input users JSONL path.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "eval" / "candidates_3km.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument("--radius-km", type=float, default=3.0, help="Radius in km.")
    parser.add_argument(
        "--max-per-mode",
        type=int,
        default=50,
        help="Max candidates per mode (0 for no limit).",
    )
    parser.add_argument(
        "--modes",
        default="restaurant,cafe,tourspot",
        help="Comma-separated modes.",
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    candidates_by_mode = {mode: _load_mode(mode) for mode in modes}
    anchor_lookup = _build_anchor_lookup(candidates_by_mode)

    users_path = Path(args.users)
    users = _load_users(users_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    limit = args.max_per_mode if args.max_per_mode > 0 else None

    with output_path.open("w", encoding="utf-8") as f:
        for user in users:
            selected = user.get("selectedPlaces") or []
            if not selected:
                continue
            anchor_id = selected[-1].get("place_id")
            if anchor_id is None:
                continue
            anchor_coords = anchor_lookup.get(int(anchor_id))
            if not anchor_coords:
                continue
            anchor_lat, anchor_lng = anchor_coords

            combined = []
            for mode, candidates in candidates_by_mode.items():
                items = []
                for cand in candidates:
                    if cand["place_id"] == anchor_id:
                        continue
                    dist = _haversine_km(anchor_lat, anchor_lng, cand["lat"], cand["lng"])
                    if dist <= args.radius_km:
                        item = {
                            "id": cand["place_id"],
                            "category": cand["category"],
                            "name": cand["name"],
                            "distance_km": round(dist, 4),
                            "keywords": cand.get("keywords") or [],
                        }
                        if mode == "restaurant":
                            item["food_type"] = cand.get("content")
                        elif mode == "cafe":
                            item["cafe_type"] = cand.get("content")
                        else:
                            themes = cand.get("themes")
                            if not themes:
                                themes = cand.get("poi_type") or cand.get("type")
                            item["themes"] = themes
                        items.append(item)
                items.sort(key=lambda x: x["distance_km"])
                if limit is not None:
                    items = items[:limit]
                combined.extend(items)

            user_payload = {
                "userId": user.get("userId"),
                "preferredThemes": user.get("preferredThemes") or [],
                "preferredMoods": user.get("preferredMoods") or [],
                "preferredRestaurantTypes": user.get("preferredRestaurantTypes") or [],
                "preferredCafeTypes": user.get("preferredCafeTypes") or [],
                "avoid": user.get("avoid") or [],
                "activityLevel": user.get("activityLevel"),
                "companion": user.get("companion") or [],
                "budget": user.get("budget"),
                "region": user.get("region"),
            }
            record = {
                "user": user_payload,
                "distance_limit_km": args.radius_km,
                "candidates": combined,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote candidates to {output_path}")


if __name__ == "__main__":
    main()
