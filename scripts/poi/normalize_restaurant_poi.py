import json
import uuid
from typing import Optional


# =========================
# Utils
# =========================

def gen_poi_id(source_id: Optional[str]) -> str:
    return source_id if source_id else f"poi_{uuid.uuid4().hex[:10]}"


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def parse_review_count(x: Optional[str]) -> Optional[int]:
    """
    "(16)" → 16
    """
    if not x:
        return None
    x = x.strip()
    if x.startswith("(") and x.endswith(")"):
        return to_int(x[1:-1])
    return None


# =========================
# Normalizer
# =========================

def normalize_restaurant_poi(raw: dict) -> dict:
    return {
        "poi_id": gen_poi_id(raw.get("id")),

        "type": "restaurant",
        "sub_type": None,

        "name": raw.get("title"),
        "address": raw.get("address"),

        # ✅ 이미지 추가
        "image": raw.get("imglinks"),

        "location": {
            "lat": to_float(raw.get("latitude")),
            "lng": to_float(raw.get("longitude")),
        },

        "description": raw.get("description"),


        "attributes": {},

        "stats": {
            "views": to_int(raw.get("views")) or 0,
            "likes": to_int(raw.get("likes")) or 0,
            "bookmarks": to_int(raw.get("bookmarks")) or 0,
            "rating": to_float(raw.get("starts")),
            "review_count": parse_review_count(raw.get("counts"))
        }
    }


# =========================
# Main
# =========================

def main():
    input_path = "data/raw/rest_data.json"
    output_path = "data/interim/restaurant_poi_normalized.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized_data = []

    # ✅ data가 list일 때
    if isinstance(data, list):
        for row in data:
            normalized_data.append(normalize_restaurant_poi(row))

    # ✅ data가 {"items": [...]} 처럼 dict일 때 대응(혹시 몰라서)
    elif isinstance(data, dict):
        # 흔한 케이스들
        rows = data.get("items") or data.get("data") or data.get("rows") or []
        for row in rows:
            normalized_data.append(normalize_restaurant_poi(row))

    else:
        raise ValueError(f"Unsupported JSON top-level type: {type(data)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(normalized_data)} restaurant POIs → {output_path}")



if __name__ == "__main__":
    main()
