#!/usr/bin/env python3
'''
python scripts/eval/compute_metrics.py \
  --labels data/eval/labels_top5_all.jsonl \
  --pred data/eval/filtered/recommendations.jsonl \
  --k 10
'''
import argparse
import json
import math
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


def _load_labels(paths: list[Path]) -> dict[tuple[int, str], list[int]]:
    labels: dict[tuple[int, str], list[int]] = {}
    for path in paths:
        for rec in _load_jsonl(path):
            user_id = rec.get("userId")
            category = rec.get("category")
            relevant = rec.get("relevant_ids") or []
            if user_id is None or category is None:
                continue
            labels[(int(user_id), str(category))] = [int(x) for x in relevant]
    return labels


def _load_predictions(path: Path) -> dict[tuple[int, str], list[int]]:
    pred: dict[tuple[int, str], list[int]] = {}
    for rec in _load_jsonl(path):
        user_id = rec.get("userId")
        if user_id is None:
            continue
        recs = rec.get("recommendations") or []
        for block in recs:
            category = block.get("category")
            if not category:
                continue
            items = block.get("items") or []
            ids = []
            for item in items:
                pid = item.get("place_id")
                if pid is None:
                    pid = item.get("id")
                if pid is None:
                    continue
                ids.append(int(pid))
            pred[(int(user_id), str(category))] = ids
    return pred


def _recall_at_k(pred: list[int], relevant: list[int], k: int) -> float:
    if not relevant:
        return 0.0
    hit = len(set(pred[:k]) & set(relevant))
    return hit / float(len(relevant))


def _ndcg_at_k(pred: list[int], relevant: list[int], k: int) -> float:
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    dcg = 0.0
    for idx, pid in enumerate(pred[:k], start=1):
        if pid in rel_set:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_k = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_k + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _summarize(pred: dict[tuple[int, str], list[int]], labels: dict[tuple[int, str], list[int]], k: int, categories: list[str]) -> dict:
    stats = {
        "total_labels": 0,
        "eval_labels": 0,
        "missing_preds": 0,
        "empty_relevant": 0,
        "by_category": {cat: {"count": 0, "recall_sum": 0.0, "ndcg_sum": 0.0} for cat in categories},
    }
    for key, relevant in labels.items():
        user_id, category = key
        if category not in categories:
            continue
        stats["total_labels"] += 1
        if not relevant:
            stats["empty_relevant"] += 1
            continue
        pred_list = pred.get(key)
        if not pred_list:
            stats["missing_preds"] += 1
            continue
        stats["eval_labels"] += 1
        recall = _recall_at_k(pred_list, relevant, k)
        ndcg = _ndcg_at_k(pred_list, relevant, k)
        cat_stats = stats["by_category"][category]
        cat_stats["count"] += 1
        cat_stats["recall_sum"] += recall
        cat_stats["ndcg_sum"] += ndcg
    return stats


def _print_summary(name: str, stats: dict, categories: list[str]) -> None:
    print(f"\n== {name} ==")
    print(f"total_labels: {stats['total_labels']}")
    print(f"eval_labels: {stats['eval_labels']}")
    print(f"missing_preds: {stats['missing_preds']}")
    print(f"empty_relevant: {stats['empty_relevant']}")
    for cat in categories:
        cat_stats = stats["by_category"][cat]
        count = cat_stats["count"]
        recall = cat_stats["recall_sum"] / count if count else 0.0
        ndcg = cat_stats["ndcg_sum"] / count if count else 0.0
        print(f"{cat}: count={count} recall@k={recall:.4f} ndcg@k={ndcg:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Recall@K and NDCG@K.")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels JSONL paths.")
    parser.add_argument("--pred", action="append", required=True, help="Prediction JSONL path (repeatable).")
    parser.add_argument("--k", type=int, default=10, help="K for Recall/NDCG.")
    parser.add_argument(
        "--categories",
        default="restaurant,cafe,tourspot",
        help="Comma-separated categories.",
    )
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    label_paths = [Path(p) for p in args.labels]
    labels = _load_labels(label_paths)

    for pred_path in args.pred:
        path = Path(pred_path)
        pred = _load_predictions(path)
        stats = _summarize(pred, labels, args.k, categories)
        _print_summary(path.name, stats, categories)


if __name__ == "__main__":
    main()
