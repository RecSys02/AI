#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> tuple[list[dict], int]:
    records = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                skipped += 1
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    return records, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge label JSONL files with de-dup.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument(
        "--prefer",
        choices=["first", "last"],
        default="last",
        help="Which duplicate to keep (by userId+category).",
    )
    args = parser.parse_args()

    merged: dict[tuple[int, str], dict] = {}
    total_skipped = 0
    for input_path in args.inputs:
        path = Path(input_path)
        records, skipped = _read_jsonl(path)
        total_skipped += skipped
        for rec in records:
            user_id = rec.get("userId")
            category = rec.get("category")
            if user_id is None or category is None:
                continue
            key = (int(user_id), str(category))
            if key in merged and args.prefer == "first":
                continue
            merged[key] = rec

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in merged.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(merged)} records to {output_path}")
    if total_skipped:
        print(f"Skipped invalid lines: {total_skipped}")


if __name__ == "__main__":
    main()
