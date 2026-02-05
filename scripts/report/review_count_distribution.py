#!/usr/bin/env python3
"""Print review count (v) distribution for popularity scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

DEFAULT_DATA_DIR = Path("data/embedding_json")
DEFAULT_FILES = {
    "tourspot": DEFAULT_DATA_DIR / "embedding_tourspot.json",
    "cafe": DEFAULT_DATA_DIR / "embedding_cafe.json",
    "restaurant": DEFAULT_DATA_DIR / "embedding_restaurant.json",
}


def _extract_v(obj: dict) -> Optional[float]:
    # Prefer top-level google field (current embedding_json format)
    google = obj.get("google") if isinstance(obj.get("google"), dict) else None
    if google is None:
        meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
        google = meta.get("google") if isinstance(meta.get("google"), dict) else None
    if not google:
        return None
    v = google.get("user_ratings_total")
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v) or v <= 0:
        return None
    return v


def _load_counts(path: Path) -> list[Optional[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    counts: list[Optional[float]] = []
    for obj in data:
        if not isinstance(obj, dict):
            counts.append(None)
            continue
        counts.append(_extract_v(obj))
    return counts


def _summary(values: Iterable[Optional[float]]) -> dict:
    values = list(values)
    total = len(values)
    arr = np.array([v for v in values if v is not None], dtype=float)
    valid = int(arr.size)
    missing = total - valid
    if valid == 0:
        return {
            "total": total,
            "valid": 0,
            "missing": missing,
            "percentiles": {},
            "mean": None,
            "min": None,
            "max": None,
            "pct_v_le_2": None,
            "pct_v_le_5": None,
            "pct_v_le_10": None,
        }
    percentiles = {
        "p10": np.percentile(arr, 10),
        "p25": np.percentile(arr, 25),
        "p50": np.percentile(arr, 50),
        "p60": np.percentile(arr, 60),
        "p70": np.percentile(arr, 70),
        "p80": np.percentile(arr, 80),
        "p90": np.percentile(arr, 90),
        "p95": np.percentile(arr, 95),
    }
    return {
        "total": total,
        "valid": valid,
        "missing": missing,
        "percentiles": percentiles,
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "pct_v_le_2": float((arr <= 2).mean() * 100),
        "pct_v_le_5": float((arr <= 5).mean() * 100),
        "pct_v_le_10": float((arr <= 10).mean() * 100),
    }


def _bucket_counts(values: Iterable[Optional[float]]):
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return []
    # Bucket edges (inclusive lower, exclusive upper)
    edges = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000]
    buckets = []
    prev = 0
    for edge in edges:
        buckets.append((prev + 1, edge))
        prev = edge
    buckets.append((edges[-1] + 1, float("inf")))

    counts = []
    for lo, hi in buckets:
        if hi == float("inf"):
            mask = arr >= lo
        else:
            mask = (arr >= lo) & (arr <= hi)
        counts.append((lo, hi, int(mask.sum())))
    return counts


def _print_summary(label: str, values: list[Optional[float]]) -> None:
    s = _summary(values)
    print(f"[{label}] total={s['total']}, valid={s['valid']}, missing={s['missing']}")
    if s["valid"] == 0:
        print("  no valid review counts")
        return
    p = s["percentiles"]
    print(
        "  "
        + " ".join(
            [
                f"min={s['min']:.0f}",
                f"mean={s['mean']:.1f}",
                f"p50={p['p50']:.0f}",
                f"p60={p['p60']:.0f}",
                f"p70={p['p70']:.0f}",
                f"p80={p['p80']:.0f}",
                f"p90={p['p90']:.0f}",
                f"max={s['max']:.0f}",
            ]
        )
    )
    print(
        f"  v<=2: {s['pct_v_le_2']:.1f}%  v<=5: {s['pct_v_le_5']:.1f}%  v<=10: {s['pct_v_le_10']:.1f}%"
    )

    buckets = _bucket_counts(values)
    if buckets:
        print("  bucket counts:")
        for lo, hi, cnt in buckets:
            label = f"{int(lo)}+" if hi == float("inf") else f"{int(lo)}-{int(hi)}"
            pct = cnt / s["valid"] * 100 if s["valid"] else 0
            print(f"    {label:>9}: {cnt:>5} ({pct:>5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print review count distribution from embedding JSON files")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to embedding_json directory (default: data/embedding_json)",
    )
    parser.add_argument(
        "--category",
        choices=["tourspot", "cafe", "restaurant", "all"],
        default="all",
        help="Category to analyze (default: all)",
    )
    args = parser.parse_args()

    files = {
        "tourspot": args.data_dir / "embedding_tourspot.json",
        "cafe": args.data_dir / "embedding_cafe.json",
        "restaurant": args.data_dir / "embedding_restaurant.json",
    }

    if args.category != "all":
        path = files[args.category]
        counts = _load_counts(path)
        _print_summary(args.category, counts)
        return

    all_counts: list[Optional[float]] = []
    for cat, path in files.items():
        counts = _load_counts(path)
        _print_summary(cat, counts)
        all_counts.extend(counts)
    _print_summary("overall", all_counts)


if __name__ == "__main__":
    main()
