#!/usr/bin/env python3
"""
Compare recall@K across embedding models using candidate-restricted Milvus search.

Flow:
- Load labels (userId, category, relevant_ids)
- Load candidates_3km (per-user candidate ids)
- Build user query texts (same as POST /recommend)
- For each model, embed queries and search only within candidate ids
- Compute recall@K (and candidate coverage)

Usage:
  python scripts/model_comparison/compare_models_recall_at10_candidates.py \
    --labels data/eval/labels_top5_gemini.jsonl \
    --candidates data/eval/candidates_3km.jsonl \
    --k 10 --batch-size 32 \
    --output data/eval/recall_at10_by_model.json
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
from dotenv import load_dotenv
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "fastapi_app"))

from models.user_input import UserInput  # noqa: E402
from utils.user_text_builder import (  # noqa: E402
    build_cafe_text,
    build_restaurant_text,
    build_tourspot_text,
)


@dataclass
class ModelSpec:
    label: str
    model_name: str
    query_prefix: str = ""
    normalize: bool = True
    trust_remote_code: bool = False


DEFAULT_MODELS: List[ModelSpec] = [
    ModelSpec(
        label="multilingual-e5-small-ko",
        model_name="dragonkue/multilingual-e5-small-ko",
        query_prefix="query: ",
        normalize=True,
    ),
    ModelSpec(
        label="klue-bert-base",
        model_name="klue/bert-base",
        normalize=True,
    ),
    ModelSpec(
        label="klue-roberta-base",
        model_name="klue/roberta-base",
        normalize=True,
    ),
    ModelSpec(
        label="bge-m3-ko",
        model_name="dragonkue/bge-m3-ko",
        normalize=True,
        trust_remote_code=True,
    ),
    ModelSpec(
        label="bge-m3",
        model_name="BAAI/bge-m3",
        normalize=True,
        trust_remote_code=True,
    ),
]

COLLECTION_PREFIX_BY_MODEL: Dict[str, str] = {
    "multilingual-e5-small-ko": "poi_",
    "klue-bert-base": "poi_klue_bert_base_",
    "klue-roberta-base": "poi_klue_roberta_base_",
    "bge-m3-ko": "poi_bge_m3_ko_",
    "bge-m3": "poi_bge_m3_",
}


@dataclass
class QueryItem:
    user_id: int
    category: str
    text: str


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def connect_milvus() -> None:
    host = os.getenv("MILVUS_HOST", "")
    if not host:
        raise SystemExit("MILVUS_HOST is not set")
    port = int(os.getenv("MILVUS_PORT", "19530"))
    user = os.getenv("MILVUS_USER")
    password = os.getenv("MILVUS_PASSWORD")
    secure = _env_bool("MILVUS_SECURE", default=False)
    connections.connect(
        alias="default",
        host=host,
        port=port,
        user=user,
        password=password,
        secure=secure,
    )


def get_collection(category: str, prefix: str) -> Collection:
    name = f"{prefix}{category}"
    return Collection(name)


def get_collection_dim(collection: Collection) -> int:
    for field in collection.schema.fields:
        if field.name == "embedding":
            dim = None
            if hasattr(field, "params") and field.params:
                dim = field.params.get("dim")
            if dim is None and hasattr(field, "dim"):
                dim = field.dim
            if dim is None:
                raise ValueError(f"Embedding dim not found for collection {collection.name}")
            return int(dim)
    raise ValueError(f"Embedding field not found for collection {collection.name}")


def _get_user_id(row: dict) -> int | None:
    return row.get("userId") or row.get("user_id")


def load_labels(path: Path) -> List[dict]:
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(json.loads(line))
    return labels


def load_candidates(path: Path) -> tuple[Dict[int, UserInput], Dict[int, Dict[str, Set[int]]]]:
    users: Dict[int, UserInput] = {}
    candidates: Dict[int, Dict[str, Set[int]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            user_obj = row.get("user") or {}
            user_id = _get_user_id(user_obj) or _get_user_id(row)
            if user_id is None:
                continue
            user_id = int(user_id)
            if user_id not in users:
                try:
                    users[user_id] = UserInput.model_validate(user_obj)
                except AttributeError:
                    users[user_id] = UserInput.parse_obj(user_obj)
            cand_map = candidates.setdefault(user_id, {})
            for item in row.get("candidates") or []:
                category = item.get("category")
                pid = item.get("id") or item.get("place_id") or item.get("placeId")
                if category and pid is not None:
                    cand_map.setdefault(category, set()).add(int(pid))
    return users, candidates


def build_query_texts(users: Dict[int, UserInput]) -> Dict[int, Dict[str, str]]:
    builders = {
        "tourspot": build_tourspot_text,
        "cafe": build_cafe_text,
        "restaurant": build_restaurant_text,
    }
    texts: Dict[int, Dict[str, str]] = {}
    for user_id, user in users.items():
        user_texts = {}
        for category, builder in builders.items():
            text = (builder(user) or "").strip()
            if text:
                user_texts[category] = text
        texts[user_id] = user_texts
    return texts


def iter_batches(items: List[QueryItem], batch_size: int) -> Iterable[List[QueryItem]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _search_with_expr(
    collection: Collection,
    vector: np.ndarray,
    candidate_ids: List[int],
    top_k: int,
) -> List[int]:
    if not candidate_ids:
        return []
    limit = min(top_k, len(candidate_ids))
    expr = f"place_id in [{', '.join(str(pid) for pid in candidate_ids)}]"
    results = collection.search(
        [vector],
        "embedding",
        {"metric_type": "IP", "params": {"ef": max(64, top_k)}},
        limit=limit,
        expr=expr,
        output_fields=["place_id"],
    )
    hits = results[0] if results else []
    return [int(hit.id) for hit in hits]


def _search_fallback(
    collection: Collection,
    vector: np.ndarray,
    candidate_ids: List[int],
    top_k: int,
) -> List[int]:
    if not candidate_ids:
        return []
    expr = f"place_id in [{', '.join(str(pid) for pid in candidate_ids)}]"
    rows = collection.query(expr, output_fields=["place_id", "embedding"])
    if not rows:
        return []
    vec = np.asarray(vector, dtype=np.float32)
    scored = []
    for row in rows:
        emb = row.get("embedding")
        pid = row.get("place_id")
        if emb is None or pid is None:
            continue
        emb_vec = np.asarray(emb, dtype=np.float32)
        score = float(np.dot(vec, emb_vec))
        scored.append((int(pid), score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in scored[: min(top_k, len(scored))]]


def search_candidates(
    collection: Collection,
    vector: np.ndarray,
    candidate_ids: List[int],
    top_k: int,
) -> List[int]:
    try:
        return _search_with_expr(collection, vector, candidate_ids, top_k)
    except Exception:
        return _search_fallback(collection, vector, candidate_ids, top_k)


def compute_recall(
    labels: List[dict],
    preds: Dict[Tuple[int, str], List[int]],
    candidates: Dict[int, Dict[str, Set[int]]],
    k: int,
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
        user_id = int(user_id)

        relevant_set = {int(x) for x in relevant if x is not None}
        pred_list = preds.get((user_id, category), [])
        topk = pred_list[:k]
        topk_set = {int(x) for x in topk if x is not None}

        hits = len(relevant_set & topk_set) if relevant_set else 0
        recall = (hits / len(relevant_set)) if relevant_set else 0.0

        candidate_set = candidates.get(user_id, {}).get(category, set())
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
        summary["candidate_coverage_mean"] = (
            summary["candidate_coverage_mean"] / summary["total"]
        )
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
    parser = argparse.ArgumentParser(description="Compare recall@K across models.")
    parser.add_argument(
        "--labels",
        default="data/eval/labels_top5_gemini.jsonl",
        help="Label JSONL path.",
    )
    parser.add_argument(
        "--candidates",
        default="data/eval/candidates_3km.jsonl",
        help="Candidate JSONL path.",
    )
    parser.add_argument("--k", type=int, default=10, help="Recall@K (default 10).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    connect_milvus()

    labels = load_labels(Path(args.labels))
    users, candidates = load_candidates(Path(args.candidates))
    user_texts = build_query_texts(users)

    label_pairs = []
    for row in labels:
        user_id = _get_user_id(row)
        category = row.get("category")
        if user_id is None or not category:
            continue
        label_pairs.append((int(user_id), category))

    results = []

    for spec in DEFAULT_MODELS:
        print(f"\n[MODEL] {spec.label} ({spec.model_name})")
        model = SentenceTransformer(spec.model_name, trust_remote_code=spec.trust_remote_code)
        prefix = COLLECTION_PREFIX_BY_MODEL.get(spec.label, "poi_")

        categories = ["tourspot", "cafe", "restaurant"]
        collections: Dict[str, Collection] = {}
        collection_dims: Dict[str, int] = {}
        for category in categories:
            try:
                collection = get_collection(category, prefix)
            except Exception as exc:
                print(f"  - skip category={category}: {exc}")
                continue
            collections[category] = collection
            try:
                dim = get_collection_dim(collection)
            except Exception as exc:
                print(f"  - skip category={category}: {exc}")
                continue
            collection_dims[category] = dim
            try:
                collection.load()
            except Exception as exc:
                print(f"  - skip category={category}: failed to load collection ({exc})")
                collection_dims.pop(category, None)
                collections.pop(category, None)

        query_items: List[QueryItem] = []
        for user_id, category in label_pairs:
            if category not in collection_dims:
                continue
            if user_id not in user_texts:
                continue
            text = user_texts[user_id].get(category, "")
            if not text:
                continue
            if not candidates.get(user_id, {}).get(category):
                continue
            query_items.append(QueryItem(user_id=user_id, category=category, text=text))

        print(f"  queries: {len(query_items)}")

        preds: Dict[Tuple[int, str], List[int]] = {}
        encode_time = 0.0
        search_time = 0.0

        for batch in iter_batches(query_items, args.batch_size):
            texts = [spec.query_prefix + q.text if spec.query_prefix else q.text for q in batch]
            start = time.perf_counter()
            vectors = model.encode(
                texts,
                batch_size=args.batch_size,
                normalize_embeddings=spec.normalize,
                show_progress_bar=False,
            )
            encode_time += time.perf_counter() - start

            start = time.perf_counter()
            for vec, q in zip(vectors, batch):
                dim = collection_dims.get(q.category)
                if dim is None or dim != len(vec):
                    preds[(q.user_id, q.category)] = []
                    continue
                collection = collections[q.category]
                candidate_ids = list(candidates[q.user_id][q.category])
                pred_ids = search_candidates(collection, vec, candidate_ids, args.k)
                preds[(q.user_id, q.category)] = pred_ids
            search_time += time.perf_counter() - start

        summary = compute_recall(labels, preds, candidates, args.k)
        summary["label"] = spec.label
        summary["model"] = spec.model_name
        summary["collection_prefix"] = prefix
        summary["queries"] = len(query_items)
        summary["encode_time_sec"] = round(encode_time, 6)
        summary["search_time_sec"] = round(search_time, 6)
        results.append(summary)

        print(
            f"  recall@{args.k}: {summary['recall_mean']:.6f} | "
            f"candidate_recall: {summary['candidate_recall_mean']:.6f}"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
