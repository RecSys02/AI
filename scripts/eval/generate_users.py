#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_JSON_DIR = PROJECT_ROOT / "data" / "embedding_json"


THEMES = [
 "관광",
 "쇼핑",
 "음식",
 "자연",
 "체험",
 "야경/감성",
 "엔터테인먼트"
]

MOODS = [
    "고급스러운",
    "세련된",
    "아기자기한",
    "로컬/서민적인",
    "조용한",
    "활기찬",
    "자연친화적인",
    "감성적인",
    "젊은/힙한",
    "가족 친화적",
    "인기 많은곳",
]

AVOIDS = [
    "너무 시끄러운 곳",
    "오래 기다리는 곳",
    "고령자에게 불편한 곳",
]

ACTIVITY_LEVELS = [
    "거의 걷고 싶지 않음",
    "적당히 걷는",
    "오래 걸어도 상관 없음",
    "오래 걷는것 선호",
]

RESTAURANT_TAGS = ["양식", "한식", "일식", "중식", "세계음식"]
CAFE_TAGS = ["디저트 위주", "커피가 맛있다", "분위기 좋은", "뷰가 좋은", "프렌차이즈"]

COMPANIONS = ["친구", "연인", "가족", "부모님", "아이동반", "반려동물"]
BUDGETS = ["저렴", "중간", "중간~높음", "높음"]

FOOD_THEMES = {"미식", "한식", "현지맛집"}
CAFE_THEMES = {"카페", "베이커리", "디저트"}


def _load_poi_refs(modes: list[str]) -> list[dict]:
    refs: list[dict] = []
    for mode in modes:
        path = EMBEDDING_JSON_DIR / f"embedding_{mode}.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            pid = item.get("place_id")
            if pid is None:
                continue
            refs.append(
                {
                    "place_id": int(pid),
                    "category": mode,
                    "province": item.get("province") or item.get("city") or "unknown",
                }
            )
    return refs


def _sample(items: list[str], min_n: int, max_n: int) -> list[str]:
    if not items:
        return []
    n = random.randint(min_n, max_n)
    n = min(n, len(items))
    return random.sample(items, n)


def _build_user(user_id: int, poi_refs: list[dict]) -> dict:
    themes = _sample(THEMES, 1, 3)
    moods = _sample(MOODS, 1, 2)
    avoid = _sample(AVOIDS, 0, 2)

    companion = None
    r = random.random()
    if r < 0.6:
        companion = [random.choice(COMPANIONS)]
    elif r < 0.8:
        companion = random.sample(COMPANIONS, 2)

    activity_level = random.choice(ACTIVITY_LEVELS)
    if companion and any(c in companion for c in ("부모님", "아이동반")):
        activity_level = random.choice(ACTIVITY_LEVELS[:2])
        if "너무 시끄러운 곳" not in avoid:
            avoid.append("너무 시끄러운 곳")

    preferred_restaurant_types = _sample(RESTAURANT_TAGS, 1, 2)
    preferred_cafe_types = _sample(CAFE_TAGS, 1, 2)

    selected_places = []
    history_places = []
    if poi_refs:
        selected_places.append(random.choice(poi_refs))
        if random.random() < 0.5:
            history_places = random.sample(poi_refs, k=random.randint(0, 2))

    return {
        "userId": user_id,
        "preferredThemes": themes,
        "preferredMoods": moods,
        "preferredRestaurantTypes": preferred_restaurant_types,
        "preferredCafeTypes": preferred_cafe_types,
        "avoid": avoid,
        "activityLevel": activity_level,
        "region": "서울",
        "companion": companion,
        "budget": random.choice(BUDGETS),
        "historyPlaces": history_places,
        "selectedPlaces": selected_places,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random user profiles (JSONL).")
    parser.add_argument("--count", type=int, default=5000, help="Number of users to generate.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "eval" / "users_5000.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--modes",
        default="restaurant,cafe,tourspot",
        help="Comma-separated POI modes to sample selected/history places from.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    poi_refs = _load_poi_refs(modes)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for i in range(1, args.count + 1):
            user = _build_user(i, poi_refs)
            f.write(json.dumps(user, ensure_ascii=False) + "\n")

    print(f"Wrote {args.count} users to {output_path}")


if __name__ == "__main__":
    main()
