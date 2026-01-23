#!/usr/bin/env python3
# uv run ../scripts/eval/filter_users_by_candidates.py --k 20

import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _user_id_from_record(record: dict) -> int | None:
    if "userId" in record:
        return record.get("userId")
    user = record.get("user") or {}
    if isinstance(user, dict):
        return user.get("userId")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter users by minimum candidates per category and sync datasets."
    )
    parser.add_argument(
        "--candidates",
        default="../data/eval/candidates_3km.jsonl",
        help="Candidates JSONL path.",
    )
    parser.add_argument(
        "--users",
        default="../data/eval/users_5000.jsonl",
        help="Users JSONL path.",
    )
    parser.add_argument(
        "--recommendations",
        default="../data/eval/recommend_top10_rerank_gemini.jsonl",
        help="Recommendations JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        default="../data/eval/filtered_rerank",
        help="Output directory for filtered JSONL files.",
    )
    parser.add_argument("--k", type=int, default=10, help="Minimum per category (0 to disable).")
    parser.add_argument("--min-total", type=int, default=0, help="Minimum total candidates (0 to disable).")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    candidates = _load_jsonl(candidates_path)

    allowed_user_ids: set[int] = set()
    for rec in candidates:
        counts = {"restaurant": 0, "cafe": 0, "tourspot": 0}
        for cand in rec.get("candidates", []):
            cat = cand.get("category")
            if cat in counts:
                counts[cat] += 1
        total = sum(counts.values())
        if args.k > 0 and not all(counts[cat] >= args.k for cat in counts):
            continue
        if args.min_total > 0 and total < args.min_total:
            continue
        user_id = _user_id_from_record(rec)
        if user_id is not None:
            allowed_user_ids.add(int(user_id))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter candidates
    filtered_candidates = [
        rec for rec in candidates if _user_id_from_record(rec) in allowed_user_ids
    ]
    _write_jsonl(out_dir / "candidates.jsonl", filtered_candidates)

    # Filter users
    users = _load_jsonl(Path(args.users))
    filtered_users = [rec for rec in users if _user_id_from_record(rec) in allowed_user_ids]
    _write_jsonl(out_dir / "users.jsonl", filtered_users)

    # Filter recommendations
    recs = _load_jsonl(Path(args.recommendations))
    filtered_recs = [rec for rec in recs if _user_id_from_record(rec) in allowed_user_ids]
    _write_jsonl(out_dir / "recommendations.jsonl", filtered_recs)

    print(f"Allowed users: {len(allowed_user_ids)}")
    print(f"Wrote: {out_dir / 'users.jsonl'}")
    print(f"Wrote: {out_dir / 'candidates.jsonl'}")
    print(f"Wrote: {out_dir / 'recommendations.jsonl'}")


if __name__ == "__main__":
    main()
