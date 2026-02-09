#!/usr/bin/env python3
"""
Benchmark Milvus vector search latency using real user inputs.

This script loads users from JSONL, builds category-specific query texts
(similar to the POST /recommend flow), embeds them with multiple models,
then queries Milvus (top-k limited).

Usage:
  python scripts/model_comparison/run_user_search_benchmark.py \
    --users data/eval/users_5000.jsonl --repeat-users 500 --repeats 5 --warmup 3 \
    --top-k 400 --batch-size 32 \
    --output data/eval/user_search_benchmark.json
"""

import argparse
import json
import os
import sys
import time
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
        normalize=False,
    ),
    ModelSpec(
        label="klue-roberta-base",
        model_name="klue/roberta-base",
        normalize=False,
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
    text: str
    category: str


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


def load_users(path: Path, limit: int) -> List[UserInput]:
    users: List[UserInput] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            try:
                user = UserInput.model_validate(data)
            except AttributeError:
                user = UserInput.parse_obj(data)
            users.append(user)
            if limit and len(users) >= limit:
                break
    return users


def build_queries(users: Iterable[UserInput], categories: Iterable[str]) -> List[QueryItem]:
    builders = {
        "tourspot": build_tourspot_text,
        "cafe": build_cafe_text,
        "restaurant": build_restaurant_text,
    }
    queries: List[QueryItem] = []
    for user in users:
        for category in categories:
            builder = builders.get(category)
            if not builder:
                continue
            text = (builder(user) or "").strip()
            if not text:
                continue
            queries.append(QueryItem(text=text, category=category))
    return queries


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


def iter_batches(items: List[QueryItem], batch_size: int) -> Iterable[List[QueryItem]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    n = len(values_sorted)
    mid = n // 2
    if n % 2 == 1:
        return values_sorted[mid]
    return (values_sorted[mid - 1] + values_sorted[mid]) / 2


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = int((len(values_sorted) * 0.95) - 1)
    idx = max(0, min(idx, len(values_sorted) - 1))
    return values_sorted[idx]


def _run_queries(
    model: SentenceTransformer,
    queries: List[QueryItem],
    batch_size: int,
    top_k: int,
    collections: Dict[str, Collection],
    collection_dims: Dict[str, int],
    normalize: bool,
    query_prefix: str,
) -> Tuple[float, float, int, int]:
    encode_time = 0.0
    search_time = 0.0
    search_count = 0
    skipped_search = 0

    for batch in iter_batches(queries, batch_size):
        texts = [query_prefix + q.text if query_prefix else q.text for q in batch]
        start = time.perf_counter()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        encode_time += time.perf_counter() - start

        start = time.perf_counter()
        for vec, q in zip(vectors, batch):
            dim = collection_dims.get(q.category)
            if dim is None or dim != len(vec):
                skipped_search += 1
                continue
            collection = collections[q.category]
            collection.search(
                [vec],
                "embedding",
                {"metric_type": "IP", "params": {"ef": max(64, top_k)}},
                limit=top_k,
                output_fields=["place_id"],
            )
            search_count += 1
        search_time += time.perf_counter() - start

    return encode_time, search_time, search_count, skipped_search


def _apply_temp_hf_cache(enabled: bool) -> tuple[Path | None, Dict[str, str | None]]:
    if not enabled:
        return None, {}
    cache_dir = Path(tempfile.mkdtemp(prefix="hf_cache_"))
    keys = [
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "SENTENCE_TRANSFORMERS_HOME",
    ]
    old_env: Dict[str, str | None] = {}
    for key in keys:
        old_env[key] = os.environ.get(key)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir / "sentence_transformers")
    return cache_dir, old_env


def _restore_env(old_env: Dict[str, str | None]) -> None:
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Milvus search using user inputs.")
    parser.add_argument(
        "--users",
        default=str(PROJECT_ROOT / "data" / "eval" / "users_5000.jsonl"),
        help="Input users JSONL path.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max users to load (0 = auto).")
    parser.add_argument("--repeat-users", type=int, default=500, help="Users per repeat.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeats.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs to discard.")
    parser.add_argument("--top-k", type=int, default=400, help="Milvus search top-k limit.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for model encode.")
    parser.add_argument(
        "--cold-cache",
        action="store_true",
        help="Use a fresh HF cache per model (forces downloads).",
    )
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    connect_milvus()

    desired_users = args.repeat_users * args.repeats
    limit = args.limit if args.limit and args.limit > 0 else desired_users
    users = load_users(Path(args.users), limit)
    if not users:
        raise SystemExit("No users loaded")

    if len(users) < desired_users:
        raise SystemExit(
            f"Not enough users loaded ({len(users)}). Need at least {desired_users} "
            f"for repeat-users={args.repeat_users} * repeats={args.repeats}."
        )

    repeat_slices = [
        users[i * args.repeat_users : (i + 1) * args.repeat_users]
        for i in range(args.repeats)
    ]

    results = []

    for spec in DEFAULT_MODELS:
        print(f"\n[MODEL] {spec.label} ({spec.model_name})")
        cache_dir, old_env = _apply_temp_hf_cache(args.cold_cache)
        load_start = time.perf_counter()
        model = SentenceTransformer(spec.model_name, trust_remote_code=spec.trust_remote_code)
        load_time = time.perf_counter() - load_start
        embed_dim = model.get_sentence_embedding_dimension()
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

        valid_categories = [c for c in categories if c in collection_dims]
        if not valid_categories:
            print("  - no valid categories, skipping model")
            if cache_dir is not None:
                _restore_env(old_env)
                shutil.rmtree(cache_dir, ignore_errors=True)
            continue

        warmup_queries = build_queries(repeat_slices[0], valid_categories)
        if not warmup_queries:
            raise SystemExit("No queries built from warmup users")

        for _ in range(args.warmup):
            _run_queries(
                model=model,
                queries=warmup_queries,
                batch_size=args.batch_size,
                top_k=args.top_k,
                collections=collections,
                collection_dims=collection_dims,
                normalize=spec.normalize,
                query_prefix=spec.query_prefix,
            )

        run_stats = []
        total_search_count = 0
        total_skipped = 0

        for idx, repeat_users in enumerate(repeat_slices, start=1):
            queries = build_queries(repeat_users, valid_categories)
            if not queries:
                continue
            encode_time, search_time, search_count, skipped = _run_queries(
                model=model,
                queries=queries,
                batch_size=args.batch_size,
                top_k=args.top_k,
                collections=collections,
                collection_dims=collection_dims,
                normalize=spec.normalize,
                query_prefix=spec.query_prefix,
            )
            total_time = encode_time + search_time
            per_query_encode = encode_time / len(queries)
            per_query_search = (search_time / search_count) if search_count else 0.0
            per_query_total = total_time / len(queries)
            run_stats.append(
                {
                    "repeat": idx,
                    "users": len(repeat_users),
                    "queries": len(queries),
                    "encode_time_sec": encode_time,
                    "search_time_sec": search_time,
                    "total_time_sec": total_time,
                    "per_query_encode_sec": per_query_encode,
                    "per_query_search_sec": per_query_search,
                    "per_query_total_sec": per_query_total,
                    "search_count": search_count,
                    "search_skipped": skipped,
                }
            )
            total_search_count += search_count
            total_skipped += skipped

        encode_times = [r["encode_time_sec"] for r in run_stats]
        search_times = [r["search_time_sec"] for r in run_stats]
        total_times = [r["total_time_sec"] for r in run_stats]
        per_query_encode_times = [r["per_query_encode_sec"] for r in run_stats]
        per_query_search_times = [r["per_query_search_sec"] for r in run_stats]
        per_query_total_times = [r["per_query_total_sec"] for r in run_stats]
        queries_per_repeat = [r["queries"] for r in run_stats]

        result = {
            "label": spec.label,
            "model": spec.model_name,
            "users_total": len(users),
            "users_per_repeat": args.repeat_users,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "queries_per_repeat_mean": _mean(queries_per_repeat),
            "encoded_dim": embed_dim,
            "collection_prefix": prefix,
            "cold_cache": bool(args.cold_cache),
            "load_time_sec": round(load_time, 6),
            "encode_time_sec_mean": round(_mean(encode_times), 6),
            "encode_time_sec_median": round(_median(encode_times), 6),
            "encode_time_sec_p95": round(_p95(encode_times), 6),
            "search_time_sec_mean": round(_mean(search_times), 6),
            "search_time_sec_median": round(_median(search_times), 6),
            "search_time_sec_p95": round(_p95(search_times), 6),
            "total_time_sec_mean": round(_mean(total_times), 6),
            "total_time_sec_median": round(_median(total_times), 6),
            "total_time_sec_p95": round(_p95(total_times), 6),
            "per_query_encode_sec_mean": round(_mean(per_query_encode_times), 6),
            "per_query_encode_sec_median": round(_median(per_query_encode_times), 6),
            "per_query_encode_sec_p95": round(_p95(per_query_encode_times), 6),
            "per_query_search_sec_mean": round(_mean(per_query_search_times), 6),
            "per_query_search_sec_median": round(_median(per_query_search_times), 6),
            "per_query_search_sec_p95": round(_p95(per_query_search_times), 6),
            "per_query_total_sec_mean": round(_mean(per_query_total_times), 6),
            "per_query_total_sec_median": round(_median(per_query_total_times), 6),
            "per_query_total_sec_p95": round(_p95(per_query_total_times), 6),
            "search_count_total": total_search_count,
            "search_skipped_total": total_skipped,
            "runs": run_stats,
        }
        results.append(result)

        print(f"  users_per_repeat: {args.repeat_users} | repeats: {args.repeats}")
        print(f"  load_time_sec: {result['load_time_sec']}")
        print(f"  encode_time_sec_mean: {result['encode_time_sec_mean']}")
        print(f"  search_time_sec_mean: {result['search_time_sec_mean']}")
        print(f"  total_time_sec_mean: {result['total_time_sec_mean']}")
        print(f"  search_count_total: {total_search_count} | skipped_total: {total_skipped}")

        if cache_dir is not None:
            _restore_env(old_env)
            shutil.rmtree(cache_dir, ignore_errors=True)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
