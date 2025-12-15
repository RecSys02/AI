import json
from pathlib import Path


def load_json(path: Path) -> list:
    if not path.exists():
        print(f"⚠️  Skip (not found): {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")

    return data


def main():
    BASE_DIR = Path(__file__).resolve().parents[2]
    INTERIM_DIR = BASE_DIR / "data" / "interim"

    input_files = [
        "attraction_poi_normalized.json",
        "cafe_poi_normalized.json",
        "restaurant_poi_normalized.json",
    ]

    merged = []
    counts = {}

    for filename in input_files:
        path = INTERIM_DIR / filename
        items = load_json(path)

        merged.extend(items)
        counts[filename] = len(items)

    output_path = INTERIM_DIR / "poi_all_interim.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print("✅ Merge completed")
    for k, v in counts.items():
        print(f" - {k}: {v}")
    print(f"➡️  Total POIs: {len(merged)}")
    print(f"📦 Output: {output_path}")


if __name__ == "__main__":
    main()
