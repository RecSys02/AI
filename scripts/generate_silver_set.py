#!/usr/bin/env python3
import argparse
import json
import math
import random
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


def _region_label(meta: dict) -> str | None:
    for key in ("district", "dong", "city", "province"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _pop_score(meta: dict) -> float:
    def _num(key):
        try:
            return float(meta.get(key) or 0)
        except Exception:
            return 0.0

    views = _num("views")
    likes = _num("likes")
    bookmarks = _num("bookmarks")
    starts = _num("starts")
    counts = _num("counts")

    score = 0.0
    if views:
        score += math.log1p(views)
    if likes:
        score += 2.0 * math.log1p(likes)
    if bookmarks:
        score += 3.0 * math.log1p(bookmarks)
    if counts:
        score += 0.5 * math.log1p(counts)
    if starts:
        score += 5.0 * starts
    return score


def _load_mode(mode: str) -> list[dict]:
    path = EMBEDDING_JSON_DIR / f"embedding_{mode}.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def _build_user_payload(
    user_id: int,
    region: str,
    anchor_id: int,
    mode: str,
    province: str,
) -> dict:
    return {
        "userId": user_id,
        "region": region,
        "preferredThemes": [],
        "preferredMoods": [],
        "preferredRestaurantTypes": [],
        "preferredCafeTypes": [],
        "avoid": [],
        "activityLevel": None,
        "companion": None,
        "budget": None,
        "historyPlaces": [],
        "selectedPlaces": [
            {"place_id": anchor_id, "category": mode, "province": province}
        ],
    }


def generate_silver_set(
    modes: list[str],
    per_mode: int,
    relevant_k: int,
    radius_by_mode: dict,
    seed: int,
) -> list[dict]:
    random.seed(seed)
    records = []
    user_id = 1

    for mode in modes:
        items = _load_mode(mode)
        candidates = []
        for item in items:
            pid = item.get("place_id")
            if pid is None:
                continue
            lat, lng = _get_lat_lng(item)
            if lat is None or lng is None:
                continue
            region = _region_label(item)
            if not region:
                continue
            candidates.append(
                {
                    "place_id": int(pid),
                    "lat": lat,
                    "lng": lng,
                    "region": region,
                    "province": item.get("province") or item.get("city") or "unknown",
                    "meta": item,
                }
            )

        if not candidates:
            continue

        radius_km = float(radius_by_mode.get(mode, 3.0))
        samples = 0
        attempts = 0
        max_attempts = max(per_mode * 20, 50)

        while samples < per_mode and attempts < max_attempts:
            attempts += 1
            anchor = random.choice(candidates)
            anchor_id = anchor["place_id"]
            anchor_lat = anchor["lat"]
            anchor_lng = anchor["lng"]

            relevant = []
            for cand in candidates:
                if cand["place_id"] == anchor_id:
                    continue
                dist = _haversine_km(anchor_lat, anchor_lng, cand["lat"], cand["lng"])
                if dist <= radius_km:
                    relevant.append((cand, dist))

            if len(relevant) < relevant_k:
                continue

            relevant.sort(key=lambda x: _pop_score(x[0]["meta"]), reverse=True)
            relevant_ids = [cand["place_id"] for cand, _ in relevant[:relevant_k]]

            record = {
                "id": f"{mode}_{samples + 1}",
                "category": mode,
                "distance_max_km": radius_km,
                "relevant_ids": relevant_ids,
                "anchor": {
                    "place_id": anchor_id,
                    "lat": anchor_lat,
                    "lng": anchor_lng,
                    "region": anchor["region"],
                },
                "user": _build_user_payload(
                    user_id=user_id,
                    region=anchor["region"],
                    anchor_id=anchor_id,
                    mode=mode,
                    province=anchor["province"],
                ),
            }
            records.append(record)
            samples += 1
            user_id += 1

    return records


def main():
    parser = argparse.ArgumentParser(description="Generate a silver evaluation set (JSONL).")
    parser.add_argument(
        "--modes",
        default="restaurant,cafe,tourspot",
        help="Comma-separated modes to include.",
    )
    parser.add_argument("--per-mode", type=int, default=100, help="Samples per mode.")
    parser.add_argument("--relevant-k", type=int, default=5, help="Relevant IDs per sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "eval" / "silver_set.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument("--radius-restaurant", type=float, default=2.0)
    parser.add_argument("--radius-cafe", type=float, default=2.0)
    parser.add_argument("--radius-tourspot", type=float, default=5.0)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    radius_by_mode = {
        "restaurant": args.radius_restaurant,
        "cafe": args.radius_cafe,
        "tourspot": args.radius_tourspot,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = generate_silver_set(
        modes=modes,
        per_mode=args.per_mode,
        relevant_k=args.relevant_k,
        radius_by_mode=radius_by_mode,
        seed=args.seed,
    )

    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    print(f"Wrote {len(records)} samples to {output_path}")


if __name__ == "__main__":
    main()
