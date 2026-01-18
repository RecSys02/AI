import json
from pathlib import Path
from typing import Dict, Any, Optional


# =========================
# Utils
# =========================

def to_float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# =========================
# Parsers
# =========================

def parse_type(raw: Dict[str, Any]) -> Optional[str]:
    # attraction | leisure | shopping
    return raw.get("poi_type") or raw.get("type")


def parse_sub_type(category: Dict[str, Any]) -> Optional[str]:
    if not category:
        return None
    return category.get("cat3_name") or category.get("cat2_name")


def parse_parking(intro: Dict[str, Any]) -> Optional[bool]:
    if not intro:
        return None

    parking = intro.get("parking_info")
    if parking == "가능":
        return True
    if parking == "불가능":
        return False
    return None


def parse_rating(naver: Dict[str, Any]) -> Optional[float]:
    if not naver:
        return None
    return naver.get("naver_rating")


def parse_review_count(naver: Dict[str, Any]) -> Optional[int]:
    if not naver:
        return None
    return naver.get("naver_visitor_reviews")


def parse_image(raw: Dict[str, Any]) -> Optional[str]:
    media = raw.get("media")
    if media and media.get("firstimage"):
        return media["firstimage"]
    return None


# =========================
# Normalizer
# =========================

def normalize_poi(raw: Dict[str, Any]) -> Dict[str, Any]:
    location = raw.get("location", {})
    category = raw.get("category", {})
    intro = raw.get("intro", {})
    naver = raw.get("naver", {})

    poi_type = parse_type(raw)

    attributes = {
        "duration": raw.get("duration"),
        "activity": raw.get("activity"),
        "photospot": raw.get("photospot"),
        "indoor_outdoor": raw.get("indoor_outdoor"),
        "keywords": raw.get("keywords", []),
        "summary": raw.get("summary_one_sentence"),
        "themes": raw.get("themes", []),
        "mood": raw.get("mood", []),
        "visitor_type": raw.get("visitor_type", []),
        "best_time": raw.get("best_time", []),
    }

    # ✅ parking은 True일 때만 추가
    parking = parse_parking(intro)
    if parking is True:
        attributes["parking"] = True

    return {
        "poi_id": raw.get("poi_id") or raw.get("id"),
        "type": poi_type,
        "sub_type": parse_sub_type(category),
        "name": raw.get("name"),
        "address": location.get("addr1"),
        "gu_name": raw.get("gu_name"),
        "image": parse_image(raw),
        "location": {
            "lat": to_float(location.get("lat")),
            "lng": to_float(location.get("lng")),
        },
        "description": raw.get("overview"),
        "attributes": attributes,
        "stats": {
            "views": None,
            "likes": None,
            "bookmarks": None,
            "rating": parse_rating(naver),
            "review_count": parse_review_count(naver),
        },
    }


# =========================
# Main
# =========================

def main():
    BASE_DIR = Path(__file__).resolve().parents[2]

    input_path = BASE_DIR / "data/processed/poi_merged.json"
    output_path = BASE_DIR / "data/interim/attraction_poi_normalized.json"

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    normalized_data = []

    for raw in raw_data:
        poi_type = parse_type(raw)
        if poi_type not in {"attraction", "leisure", "shopping"}:
            continue

        normalized_data.append(normalize_poi(raw))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(normalized_data)} POIs → {output_path}")


if __name__ == "__main__":
    main()
