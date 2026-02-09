#!/usr/bin/env python3
"""
Compute recall@K using label JSONL and recommendation JSONL.

Example:
  python scripts/eval/compute_recall_at10.py \
    --labels data/eval/labels_top5_gemini.jsonl \
    --preds data/eval/recommend_top10_rerank_gemini.jsonl \
    --k 10 --output data/eval/recall_at10_gemini.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def _get_user_id(row: dict) -> int:
    return row.get("userId") or row.get("user_id")


def _get_place_id(item: dict):
    return item.get("place_id") or item.get("placeId")


def load_labels(path: Path) -> List[dict]:
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(json.loads(line))
    return labels


def load_predictions(path: Path) -> Dict[int, Dict[str, List[int]]]:
    preds: Dict[int, Dict[str, List[int]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            user_id = _get_user_id(row)
            if user_id is None:
                continue
            user_map = preds.setdefault(int(user_id), {})
            for rec in row.get("recommendations", []):
                category = rec.get("category")
                items = rec.get("items") or []
                ids = []
                for item in items:
                    pid = _get_place_id(item)
                    if pid is None:
                        continue
                    ids.append(int(pid))
                if category:
                    user_map[category] = ids
    return preds


def load_candidates(path: Path) -> Dict[int, Dict[str, Set[int]]]:
    candidates: Dict[int, Dict[str, Set[int]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            user = row.get("user") or {}
            user_id = _get_user_id(user) or _get_user_id(row)
            if user_id is None:
                continue
            user_map = candidates.setdefault(int(user_id), {})
            for item in row.get("candidates") or []:
                category = item.get("category")
                pid = item.get("id") or item.get("place_id") or item.get("placeId")
                if category and pid is not None:
                    user_map.setdefault(category, set()).add(int(pid))
    return candidates


def compute_recall(
    labels: List[dict],
    preds: Dict[int, Dict[str, List[int]]],
    k: int,
    candidates: Dict[int, Dict[str, Set[int]]] | None = None,
) -> dict:
    summary = {
        "k": k,
        "total": 0,
        "missing": 0,
        "recall_mean": 0.0,
        "candidate_coverage_mean": 0.0,
        "candidate_recall_mean": 0.0,
        "candidate_recall_count": 0,
        "by_category": {},
    }

    for row in labels:
        user_id = _get_user_id(row)
        category = row.get("category")
        relevant = row.get("relevant_ids") or row.get("relevantIds") or []

        if user_id is None or not category:
            continue

        relevant_set = {int(x) for x in relevant if x is not None}
        pred_list = preds.get(int(user_id), {}).get(category, [])
        topk = pred_list[:k]
        topk_set = {int(x) for x in topk if x is not None}

        hits = len(relevant_set & topk_set) if relevant_set else 0
        recall = (hits / len(relevant_set)) if relevant_set else 0.0

        candidate_set = None
        if candidates is not None:
            candidate_set = candidates.get(int(user_id), {}).get(category)
        if candidate_set is None:
            candidate_set = relevant_set
        relevant_in_candidate = relevant_set & candidate_set if relevant_set else set()
        hits_in_candidate = len(relevant_in_candidate & topk_set) if relevant_in_candidate else 0
        candidate_recall = (
            hits_in_candidate / len(relevant_in_candidate)
            if relevant_in_candidate
            else 0.0
        )
        candidate_coverage = (
            len(relevant_in_candidate) / len(relevant_set) if relevant_set else 0.0
        )

        summary["total"] += 1
        summary["recall_mean"] += recall
        summary["candidate_coverage_mean"] += candidate_coverage
        if relevant_in_candidate:
            summary["candidate_recall_mean"] += candidate_recall
            summary["candidate_recall_count"] += 1
        if not pred_list:
            summary["missing"] += 1

        cat_stats = summary["by_category"].setdefault(
            category,
            {
                "total": 0,
                "missing": 0,
                "recall_mean": 0.0,
                "candidate_coverage_mean": 0.0,
                "candidate_recall_mean": 0.0,
                "candidate_recall_count": 0,
            },
        )
        cat_stats["total"] += 1
        cat_stats["recall_mean"] += recall
        cat_stats["candidate_coverage_mean"] += candidate_coverage
        if relevant_in_candidate:
            cat_stats["candidate_recall_mean"] += candidate_recall
            cat_stats["candidate_recall_count"] += 1
        if not pred_list:
            cat_stats["missing"] += 1

    if summary["total"]:
        summary["recall_mean"] = summary["recall_mean"] / summary["total"]
        summary["candidate_coverage_mean"] = summary["candidate_coverage_mean"] / summary["total"]
    if summary["candidate_recall_count"]:
        summary["candidate_recall_mean"] = (
            summary["candidate_recall_mean"] / summary["candidate_recall_count"]
        )
    for cat_stats in summary["by_category"].values():
        if cat_stats["total"]:
            cat_stats["recall_mean"] = cat_stats["recall_mean"] / cat_stats["total"]
            cat_stats["candidate_coverage_mean"] = (
                cat_stats["candidate_coverage_mean"] / cat_stats["total"]
            )
        if cat_stats["candidate_recall_count"]:
            cat_stats["candidate_recall_mean"] = (
                cat_stats["candidate_recall_mean"] / cat_stats["candidate_recall_count"]
            )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute recall@K for recommendations.")
    parser.add_argument(
        "--labels",
        default="data/eval/labels_top5_gemini.jsonl",
        help="Label JSONL path.",
    )
    parser.add_argument(
        "--preds",
        default="data/eval/recommend_top10_rerank_gemini.jsonl",
        help="Recommendation JSONL path.",
    )
    parser.add_argument(
        "--candidates",
        default="data/eval/candidates_3km.jsonl",
        help="Candidate JSONL path (per-user candidates).",
    )
    parser.add_argument("--k", type=int, default=10, help="Recall@K (default 10).")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    labels = load_labels(Path(args.labels))
    preds = load_predictions(Path(args.preds))
    candidates = load_candidates(Path(args.candidates)) if args.candidates else None

    summary = compute_recall(labels, preds, args.k, candidates=candidates)
    print(
        f"recall@{args.k}: {summary['recall_mean']:.6f} | total={summary['total']} "
        f"missing={summary['missing']}"
    )
    if candidates is not None:
        print(
            f"candidate_coverage_mean: {summary['candidate_coverage_mean']:.6f} | "
            f"candidate_recall_mean: {summary['candidate_recall_mean']:.6f} "
            f"(count={summary['candidate_recall_count']})"
        )
    for category, stats in summary["by_category"].items():
        print(
            f"  {category}: recall@{args.k}={stats['recall_mean']:.6f} "
            f"total={stats['total']} missing={stats['missing']}"
        )
        if candidates is not None:
            print(
                f"    candidate_coverage_mean={stats['candidate_coverage_mean']:.6f} "
                f"candidate_recall_mean={stats['candidate_recall_mean']:.6f} "
                f"(count={stats['candidate_recall_count']})"
            )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
