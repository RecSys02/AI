import json
import os
from pathlib import Path

# 이 스크립트는 여러 POI(관광지, 음식점, 카페 등) JSON 파일을 병합하여 백엔드 데이터베이스에 적합한 형식으로 변환하는 작업을 수행합니다.

def load_json(path: Path) -> list:
    if not path.exists():
        print(f"⚠️  Skip (not found): {path}")
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data


def _normalize_images(item: dict) -> list:
    # 1. 관광지(tourspot)인 경우 기존 로직 유지
    if item.get("category") == "tourspot":
        media = item.get("media") or {}
        first = media.get("firstimage")
        return [first] if first else []
    
    # 2. 그 외(cafe, restaurant 등)는 무조건 imglinks만 가져옴
    # 값이 없으면 빈 리스트([])를 반환하여 데이터 타입을 일관되게 유지합니다.
    return item.get("imglinks") or []


def _normalize_address(item: dict) -> str:
    if item.get("category") == "tourspot":
        location = item.get("location") if isinstance(item.get("location"), dict) else {}
        addr1 = location.get("addr1")
        if addr1:
            return addr1
    return item.get("address") or ""


def _normalize_lat_lng(item: dict) -> tuple:
    if item.get("category") == "tourspot":
        location = item.get("location") if isinstance(item.get("location"), dict) else {}
        return location.get("lat"), location.get("lng")
    return item.get("latitude"), item.get("longitude")


def _normalize_description(item: dict) -> str:
    if item.get("category") == "tourspot":
        return item.get("overview") or item.get("description") or ""
    return item.get("description") or ""


def _normalize_duration(item: dict) -> object:
    if item.get("category") == "tourspot":
        return item.get("duration")
    return None


def _normalize_keyword(item: dict) -> object:
    return item.get("keywords") or item.get("keyword")

def _get_google_rating_count(item: dict) -> tuple[float | None, float | None]:
    google = item.get("google") if isinstance(item.get("google"), dict) else {}
    rating = google.get("rating")
    count = google.get("user_ratings_total")
    if rating is None or count is None:
        return None, None
    try:
        rating_val = float(rating)
        count_val = float(count)
    except (TypeError, ValueError):
        return None, None
    if rating_val <= 0 or count_val <= 0:
        return None, None
    return rating_val, count_val


def _compute_popularity_score(
    item: dict, global_mean: float | None, min_votes: float
) -> object:
    rating_val, count_val = _get_google_rating_count(item)
    if rating_val is None or count_val is None or global_mean is None:
        return None
    min_votes = max(float(min_votes), 1.0)
    weighted = (count_val / (count_val + min_votes)) * rating_val + (
        min_votes / (count_val + min_votes)
    ) * global_mean
    return max(min(weighted / 5.0, 1.0), 0.0)


def to_backend_schema(item: dict, global_mean: float | None, min_votes: float) -> dict:
    lat, lng = _normalize_lat_lng(item)
    return {
        "place_id": item.get("place_id"),
        "category": item.get("category"),
        "province": item.get("province"),
        "name": item.get("name"),
        "address": _normalize_address(item),
        "duration": _normalize_duration(item),
        "description": _normalize_description(item),
        "images": _normalize_images(item),
        "latitude": lat,
        "longitude": lng,
        "keyword": _normalize_keyword(item),
        "popularity_score": _compute_popularity_score(item, global_mean, min_votes),
    }


INPUT_DIR = Path("/Users/park9379/Documents/GitHub/AI/data/embedding_json")
OUTPUT_PATH = Path("/Users/park9379/Documents/GitHub/AI/data/backend_db/merged_poi.json")


def main() -> None:
    src_dir = INPUT_DIR

    input_files = [
        "embedding_cafe.json",
        "embedding_restaurant.json",
        "embedding_tourspot.json",
    ]

    merged = []
    counts = {}
    all_items = []

    for filename in input_files:
        path = src_dir / filename
        items = load_json(path)
        all_items.extend(items)
        counts[filename] = len(items)

    ratings = []
    for item in all_items:
        rating_val, count_val = _get_google_rating_count(item)
        if rating_val is not None and count_val is not None:
            ratings.append(rating_val)
    global_mean = sum(ratings) / len(ratings) if ratings else None
    try:
        min_votes = float(os.getenv("POPULARITY_IMDB_MIN_VOTES", "50"))
    except ValueError:
        min_votes = 50.0

    merged.extend([to_backend_schema(item, global_mean, min_votes) for item in all_items])

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("✅ Merge completed")
    for name, count in counts.items():
        print(f" - {name}: {count}")
    print(f"➡️  Total POIs: {len(merged)}")
    print(f"📦 Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
