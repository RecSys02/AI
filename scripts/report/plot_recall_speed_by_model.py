#!/usr/bin/env python3
"""
Plot recall@10 vs speed (encode/search time) by model.

Usage:
  python scripts/report/plot_recall_speed_by_model.py \
    --input data/eval/user_search_benchmark.json \
    --output data/eval/recall_speed_by_model.png
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
    parser = argparse.ArgumentParser(description="Plot recall@10 vs speed by model.")
    parser.add_argument(
        "--input",
        default="data/eval/user_search_benchmark.json",
        help="Input JSON path.",
    )
    parser.add_argument(
        "--output",
        default="data/eval/recall_speed_by_model.png",
        help="Output image path (png).",
    )
    parser.add_argument("--show", action="store_true", help="Show plot window.")
    args = parser.parse_args()

    data = load_results(Path(args.input))
    if not data:
        raise SystemExit("No data found in input JSON.")

    labels = [LABEL_MAP.get(d.get("label", ""), d.get("label", "")) for d in data]
    encode_time = np.array([d.get("encode_time_sec_mean", d.get("encode_time_sec", 0.0)) for d in data])
    search_time = np.array([d.get("search_time_sec_mean", d.get("search_time_sec", 0.0)) for d in data])
    if not any("encode_time_sec_mean" in d for d in data):
        print(
            "Warning: encode_time_sec_mean/search_time_sec_mean not found. "
            "Falling back to encode_time_sec/search_time_sec."
        )
    total_time = encode_time + search_time

    fig, ax = plt.subplots(figsize=(11, 7.2), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, encode_time, label="encode", color="#54A24B")
    ax.bar(x, search_time, bottom=encode_time, label="search", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Model Performance: Encode + Search Time")
    ax.legend()

    # Labels inside bars and total on top
    totals = encode_time + search_time
    for i, (enc, sea, total) in enumerate(zip(encode_time, search_time, totals)):
        if enc > 0:
            ax.text(i, enc / 2, f"{enc:.2f}s", ha="center", va="center", color="white", fontsize=9)
        if sea > 0:
            ax.text(i, enc + sea / 2, f"{sea:.2f}s", ha="center", va="center", color="white", fontsize=9)
        ax.text(i, total + max(totals) * 0.01, f"{total:.2f}s", ha="center", va="bottom", fontsize=9)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
