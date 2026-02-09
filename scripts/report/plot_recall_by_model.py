#!/usr/bin/env python3
"""
Plot recall@10 by model from a JSON summary.

Usage:
  python scripts/report/plot_recall_by_model.py \
    --input data/eval/recall_at10_by_model.json \
    --output data/eval/recall_at10_by_model.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

LABEL_MAP: Dict[str, str] = {
    "multilingual-e5-small-ko": "multilingual-e5-small-ko",
    "klue-bert-base": "klue-bert-base",
    "klue-roberta-base": "klue-roberta-base",
    "bge-m3-ko": "bge-m3-ko",
    "bge-m3": "bge-m3",
}


def load_results(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot recall@10 by model.")
    parser.add_argument(
        "--input",
        default="data/eval/recall_at10_by_model.json",
        help="Input JSON path.",
    )
    parser.add_argument(
        "--output",
        default="data/eval/recall_at10_by_model.png",
        help="Output image path (png).",
    )
    parser.add_argument("--show", action="store_true", help="Show plot window.")
    args = parser.parse_args()

    data = load_results(Path(args.input))
    if not data:
        raise SystemExit("No data found in input JSON.")

    # Sort by overall recall desc
    data = sorted(data, key=lambda x: x.get("recall_mean", 0.0), reverse=True)

    labels = [LABEL_MAP.get(d.get("label", ""), d.get("label", "")) for d in data]
    overall = [d.get("recall_mean", 0.0) for d in data]

    fig, ax = plt.subplots(figsize=(11, 7.2), constrained_layout=True)

    # Overall recall
    x = np.arange(len(labels))
    ax.bar(x, overall, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Recall@10")
    ax.set_title("Model Performance: Recall@10")
    for i, v in enumerate(overall):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
