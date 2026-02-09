#!/usr/bin/env python3
"""
Compare embedding inference + Milvus search latency across models.

Usage:
  python scripts/model_comparison/run_model_comparison.py \
    --max-queries 300 --top-k 400 --batch-size 32 --output data/eval/model_comparison.json
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from dotenv import load_dotenv
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "embedding_json"
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_JSON_FILES = [
    DATA_DIR / "embedding_tourspot.json",
    DATA_DIR / "embedding_cafe.json",
    DATA_DIR / "embedding_restaurant.json",
]

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

# Optional per-model collection prefix override.
# If empty or missing, uses MILVUS_COLLECTION_PREFIX or "poi_".
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


def build_query_text(row: dict) -> str:
    summary = (row.get("summary_one_sentence") or "").strip()
    overview = (row.get("overview") or "").strip()
    name = (row.get("name") or row.get("title") or "").strip()
    keywords = row.get("keywords")

    parts: List[str] = []
    if summary:
        parts.append(summary)
    elif overview:
        parts.append(overview)
    elif name:
        parts.append(name)

    if keywords:
        if isinstance(keywords, list):
            kw = ", ".join(str(k) for k in keywords if k)
        else:
            kw = str(keywords)
        kw = kw.strip()
        if kw:
            parts.append(f"keywords: {kw}")

    if not parts and name:
        parts.append(name)

    return " ".join(parts).strip()


def load_queries(paths: Iterable[Path], max_queries: int) -> List[QueryItem]:
    queries: List[QueryItem] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Missing file: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            category = (row.get("category") or "").strip() or "unknown"
            text = build_query_text(row)
            if not text:
                continue
            queries.append(QueryItem(text=text, category=category))
            if max_queries and len(queries) >= max_queries:
                return queries
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare embedding inference speed across models.")
    parser.add_argument("--max-queries", type=int, default=300, help="Max queries to use.")
    parser.add_argument("--top-k", type=int, default=400, help="Milvus search top-k limit.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for model encode.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    connect_milvus()

    queries = load_queries(DEFAULT_JSON_FILES, args.max_queries)
    if not queries:
        raise SystemExit("No queries loaded from embedding_json")

    default_prefix = os.getenv("MILVUS_COLLECTION_PREFIX", "poi_")
    categories = sorted({q.category for q in queries})

    results = []

    for spec in DEFAULT_MODELS:
        print(f"\n[MODEL] {spec.label} ({spec.model_name})")
        load_start = time.perf_counter()
        model = SentenceTransformer(spec.model_name, trust_remote_code=spec.trust_remote_code)
        load_time = time.perf_counter() - load_start
        embed_dim = model.get_sentence_embedding_dimension()
        prefix = COLLECTION_PREFIX_BY_MODEL.get(spec.label, default_prefix)

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

        encode_time = 0.0
        search_time = 0.0
        search_count = 0
        skipped_search = 0

        for batch in iter_batches(queries, args.batch_size):
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
                    skipped_search += 1
                    continue
                collection = collections[q.category]
                collection.search(
                    [vec],
                    "embedding",
                    {"metric_type": "IP", "params": {"ef": max(64, args.top_k)}},
                    limit=args.top_k,
                    output_fields=["place_id"],
                )
                search_count += 1
            search_time += time.perf_counter() - start

        total_time = encode_time + search_time
        per_query_encode = encode_time / len(queries)
        per_query_search = (search_time / search_count) if search_count else 0.0

        result = {
            "label": spec.label,
            "model": spec.model_name,
            "queries": len(queries),
            "encoded_dim": embed_dim,
            "collection_prefix": prefix,
            "load_time_sec": round(load_time, 6),
            "encode_time_sec": round(encode_time, 6),
            "search_time_sec": round(search_time, 6),
            "total_time_sec": round(total_time, 6),
            "per_query_encode_sec": round(per_query_encode, 6),
            "per_query_search_sec": round(per_query_search, 6),
            "search_count": search_count,
            "search_skipped": skipped_search,
        }
        results.append(result)

        print(f"  queries: {len(queries)}")
        print(f"  load_time_sec: {result['load_time_sec']}")
        print(f"  encode_time_sec: {result['encode_time_sec']}")
        print(f"  search_time_sec: {result['search_time_sec']}")
        print(f"  total_time_sec: {result['total_time_sec']}")
        print(f"  search_count: {search_count} | skipped: {skipped_search}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
