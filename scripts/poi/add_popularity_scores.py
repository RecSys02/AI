import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

# Compute IMDb-style weighted popularity score and write back to embedding JSON.
#
# Example:
#   python scripts/poi/add_popularity_scores.py --in-place
#   python scripts/poi/add_popularity_scores.py --files embedding_cafe.json embedding_restaurant.json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "embedding_json"
DEFAULT_FILES = [
    "embedding_cafe.json",
    "embedding_restaurant.json",
    "embedding_tourspot.json",
]


def _get_google_rating_count(item: dict) -> Tuple[Optional[float], Optional[float]]:
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
    rating_val: float, count_val: float, global_mean: float, min_votes: float
) -> float:
    min_votes = max(float(min_votes), 1.0)
    weighted = (count_val / (count_val + min_votes)) * rating_val + (
        min_votes / (count_val + min_votes)
    ) * global_mean
    return max(min(weighted / 5.0, 1.0), 0.0)


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
    return path.with_name(path.stem + "_with_popularity.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--min-votes",
        type=float,
        default=float(os.getenv("POPULARITY_IMDB_MIN_VOTES", "50")),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    for filename in args.files:
        path = data_dir / filename
        items = _load_json(path)
        if not items:
            continue

        ratings = []
        for item in items:
            rating_val, count_val = _get_google_rating_count(item)
            if rating_val is not None and count_val is not None:
                ratings.append(rating_val)
        global_mean = sum(ratings) / len(ratings) if ratings else None
        if global_mean is None:
            print(f"{filename}: no ratings found, skipping popularity_score")
            continue

        updated = 0
        for item in items:
            rating_val, count_val = _get_google_rating_count(item)
            if rating_val is None or count_val is None:
                item["popularity_score"] = None
                continue
            item["popularity_score"] = _compute_popularity_score(
                rating_val, count_val, global_mean, args.min_votes
            )
            updated += 1

        out_path = _output_path(path, args.in_place)
        if args.dry_run:
            print(f"{filename}: would update {updated} items -> {out_path}")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"{filename}: updated {updated} items -> {out_path}")


if __name__ == "__main__":
    main()
