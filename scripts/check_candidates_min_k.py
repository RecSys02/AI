#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check users with fewer than k candidates per category."
    )
    parser.add_argument(
        "--input",
        default="data/eval/candidates_3km.jsonl",
        help="Input candidates JSONL path.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Minimum candidates per category.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    total = 0
    missing_any = 0
    missing_by_category = {"restaurant": 0, "cafe": 0, "tourspot": 0}

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            counts = {"restaurant": 0, "cafe": 0, "tourspot": 0}
            for cand in record.get("candidates", []):
                cat = cand.get("category")
                if cat in counts:
                    counts[cat] += 1
            any_missing = False
            for cat, cnt in counts.items():
                if cnt < args.k:
                    missing_by_category[cat] += 1
                    any_missing = True
            if any_missing:
                missing_any += 1

    print(f"Total users: {total}")
    print(f"Users with any category < {args.k}: {missing_any}")
    for cat, cnt in missing_by_category.items():
        print(f"{cat}: {cnt}")


if __name__ == "__main__":
    main()
