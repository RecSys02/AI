# -*- coding: utf-8 -*-
"""
Load embeddings into Milvus collections.
"""
# 실행법 : python ingest_poi_embeddings.py
import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from dotenv import load_dotenv
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
load_dotenv(dotenv_path=ROOT / ".env")


def connect() -> None:
    host = os.getenv("MILVUS_HOST", "")
    if not host:
        raise ValueError("MILVUS_HOST is not set")
    port = int(os.getenv("MILVUS_PORT", "19530"))
    connections.connect(host=host, port=port)


def get_collection(name: str, dim: int, drop: bool = False) -> Collection:
    if utility.has_collection(name):
        if drop:
            utility.drop_collection(name)
        else:
            return Collection(name)
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


def load_inputs(mode: str) -> Tuple[np.ndarray, np.ndarray]:
    emb = np.load(EMBEDDINGS_DIR / f"embeddings_{mode}.npy")
    keys = np.load(EMBEDDINGS_DIR / f"keys_{mode}.npy", allow_pickle=True)
    if len(emb) != len(keys):
        raise ValueError(f"embeddings/keys length mismatch: {len(emb)} vs {len(keys)}")
    return emb, keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest POI embeddings into Milvus.")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    connect()
    collection_prefix = "poi_"
    drop_existing = True
    for mode in ("tourspot", "cafe", "restaurant"):
        embeddings, keys = load_inputs(mode)
        dim = embeddings.shape[1]
        collection_name = f"{collection_prefix}{mode}"
        collection = get_collection(collection_name, dim=dim, drop=drop_existing)

        place_ids = [int(k[2]) for k in keys]
        total = len(place_ids)
        for start in range(0, total, args.batch_size):
            end = min(start + args.batch_size, total)
            batch_ids = place_ids[start:end]
            batch_emb = embeddings[start:end].tolist()
            collection.insert([batch_ids, batch_emb])
        collection.flush()
        collection.load()
        print(f"✅ Ingested {total} vectors into {collection_name}")


if __name__ == "__main__":
    main()
