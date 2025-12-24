# -*- coding: utf-8 -*-
"""
cafe/restaurant embedding_json 파일을 읽어 keywords 필드를 덮어쓴 새 JSON을 생성합니다.
기존 필드는 그대로 유지하고, 쉼표/공백 분리된 keywords 배열을 추가합니다.

사용 예:
    python scripts/poi/enrich_food_keywords.py --category cafe
    python scripts/poi/enrich_food_keywords.py --category restaurant --output data/embedding_json/embedding_restaurant_with_kw.json
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[2]  # /AI
EMBED_JSON_DIR = ROOT / "data" / "embedding_json"


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_keywords(raw_kw: Any) -> List[str]:
    raw = raw_kw or ""
    if not isinstance(raw, str):
        raw = str(raw)
    return [kw.strip() for kw in raw.replace("\n", " ").split(",") if kw.strip()]


def main():
    parser = argparse.ArgumentParser(description="Add parsed keywords array to cafe/restaurant embedding_json.")
    parser.add_argument("--category", choices=["cafe", "restaurant"], required=True)
    parser.add_argument(
        "--input",
        type=Path,
        help="입력 JSON 경로 (기본: data/embedding_json/embedding_<category>.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="출력 JSON 경로 (기본: data/embedding_json/embedding_<category>_with_kw.json)",
    )
    args = parser.parse_args()

    input_path = args.input or (EMBED_JSON_DIR / f"embedding_{args.category}.json")
    output_path = args.output or (EMBED_JSON_DIR / f"embedding_{args.category}_with_kw.json")

    data = load_json(input_path)
    for item in data:
        item["keywords_parsed"] = split_keywords(item.get("keywords", ""))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ saved {len(data)} rows -> {output_path}")


if __name__ == "__main__":
    main()
