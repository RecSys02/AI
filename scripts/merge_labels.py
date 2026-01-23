#!/usr/bin/env python3
'''
python scripts/merge_labels.py \
  --inputs data/eval/labels_top5_*.jsonl \
  --output data/eval/labels_top5_all.jsonl


'''
import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


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
    for input_path in args.inputs:
        path = Path(input_path)
        for rec in _read_jsonl(path):
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


if __name__ == "__main__":
    main()
