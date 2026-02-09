#!/usr/bin/env python3
"""
Ingest POI embeddings into Milvus for multiple models.

This builds embeddings from data/embedding_json/*.json and writes them into
model-specific Milvus collections.

Usage:
  python scripts/model_comparison/ingest_model_embeddings.py
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "embedding_json"
load_dotenv(PROJECT_ROOT / ".env")

CATEGORIES = ("tourspot", "cafe", "restaurant")


@dataclass
class ModelSpec:
    label: str
    model_name: str
    collection_prefix: str
    normalize: bool = True
    trust_remote_code: bool = False


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        label="klue-bert-base",
        model_name="klue/bert-base",
        collection_prefix="poi_klue_bert_base_",
        normalize=True,
    ),
    ModelSpec(
        label="klue-roberta-base",
        model_name="klue/roberta-base",
        collection_prefix="poi_klue_roberta_base_",
        normalize=True,
    ),
    ModelSpec(
        label="bge-m3-ko",
        model_name="dragonkue/bge-m3-ko",
        collection_prefix="poi_bge_m3_ko_",
        normalize=True,
        trust_remote_code=True,
    ),
    ModelSpec(
        label="bge-m3",
        model_name="BAAI/bge-m3",
        collection_prefix="poi_bge_m3_",
        normalize=True,
        trust_remote_code=True,
    ),
]


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


def build_item_text(row: dict) -> str:
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


def load_items(category: str) -> List[Tuple[int, str]]:
    path = DATA_DIR / f"embedding_{category}.json"
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items: List[Tuple[int, str]] = []
    for row in data:
        place_id = row.get("place_id")
        if place_id is None:
            continue
        text = build_item_text(row)
        if not text:
            continue
        items.append((int(place_id), text))
    return items


def ensure_collection(name: str, dim: int) -> Collection:
    if utility.has_collection(name):
        utility.drop_collection(name)
    fields = [
        FieldSchema(name="place_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="poi embeddings")
    collection = Collection(name, schema)
    collection.create_index(
        "embedding",
        {"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}},
    )
    return collection


def iter_batches(items: List[Tuple[int, str]], batch_size: int) -> Iterable[List[Tuple[int, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def ingest_category(model: SentenceTransformer, spec: ModelSpec, category: str, batch_size: int) -> int:
    items = load_items(category)
    if not items:
        print(f"  - skip {category}: no items")
        return 0

    collection_name = f"{spec.collection_prefix}{category}"
    collection = None
    inserted = 0

    for batch in iter_batches(items, batch_size):
        place_ids = [pid for pid, _ in batch]
        texts = [text for _, text in batch]
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=spec.normalize,
            show_progress_bar=False,
        )
        if collection is None:
            dim = len(vectors[0]) if len(vectors) else 0
            if dim <= 0:
                raise RuntimeError(f"Invalid embedding dim for {spec.label} {category}")
            collection = ensure_collection(collection_name, dim)
        collection.insert([place_ids, vectors.tolist()])
        inserted += len(place_ids)

    if collection is not None:
        collection.flush()
        collection.load()
    return inserted


def main() -> None:
    connect_milvus()

    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    for spec in MODEL_SPECS:
        print(f"\n[MODEL] {spec.label} ({spec.model_name})")
        start = time.perf_counter()
        model = SentenceTransformer(spec.model_name, trust_remote_code=spec.trust_remote_code)
        total_inserted = 0
        for category in CATEGORIES:
            count = ingest_category(model, spec, category, batch_size)
            total_inserted += count
            print(f"  {category}: {count} rows -> {spec.collection_prefix}{category}")
        elapsed = time.perf_counter() - start
        print(f"  total_inserted: {total_inserted} | elapsed_sec: {elapsed:.2f}")


if __name__ == "__main__":
    main()
