#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _load_labels(path: Path) -> set[tuple[int, str]]:
    keys = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_id = obj.get("userId")
            category = obj.get("category")
            if user_id is None or category is None:
                continue
            keys.add((int(user_id), str(category)))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract candidates missing labels by user/category."
    )
    parser.add_argument(
        "--candidates",
        default="data/eval/filtered/candidates.jsonl",
        help="Candidates JSONL path.",
    )
    parser.add_argument(
        "--labels",
        default="data/eval/labels_top5_all.jsonl",
        help="Merged labels JSONL path.",
    )
    parser.add_argument(
        "--output",
        default="data/eval/missing_candidates.jsonl",
        help="Output JSONL path for missing candidates.",
    )
    args = parser.parse_args()

    label_keys = _load_labels(Path(args.labels))
    candidates_path = Path(args.candidates)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing_count = 0
    with candidates_path.open("r", encoding="utf-8") as f_in, output_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            user = obj.get("user") or {}
            user_id = user.get("userId")
            if user_id is None:
                continue
            need = [
                cat
                for cat in ("restaurant", "cafe", "tourspot")
                if (int(user_id), cat) not in label_keys
            ]
            if not need:
                continue
            filtered = [
                cand for cand in obj.get("candidates", []) if cand.get("category") in need
            ]
            if not filtered:
                continue
            out_obj = {
                "user": user,
                "distance_limit_km": obj.get("distance_limit_km"),
                "candidates": filtered,
            }
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            missing_count += 1

    print(f"Missing users: {missing_count}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
