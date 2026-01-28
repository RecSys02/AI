import json
from pathlib import Path

# 이 코드 주석 달아줘
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


def _normalize_images(item: dict) -> object:
    if item.get("category") == "tourspot":
        media = item.get("media") or {}
        first = media.get("firstimage")
        if first:
            return [first]
        return []
    return item.get("images") if "images" in item else item.get("imglinks")


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


def to_backend_schema(item: dict) -> dict:
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

    for filename in input_files:
        path = src_dir / filename
        items = load_json(path)
        merged.extend([to_backend_schema(item) for item in items])
        counts[filename] = len(items)

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
