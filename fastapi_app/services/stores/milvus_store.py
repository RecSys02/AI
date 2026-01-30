import os
from functools import lru_cache
from typing import Iterable, Optional

import numpy as np
from pymilvus import Collection, connections


class MilvusStore:
    def __init__(
        self,
        host: str,
        port: int,
        user: Optional[str] = None,
        password: Optional[str] = None,
        secure: bool = False,
        collection_prefix: str = "poi_",
    ):
        self._alias = "default"
        self._collection_prefix = collection_prefix
        connections.connect(
            alias=self._alias,
            host=host,
            port=port,
            user=user,
            password=password,
            secure=secure,
        )
        self._collections: dict[str, Collection] = {}

    def _collection_name(self, category: str) -> str:
        return f"{self._collection_prefix}{category}"

    def _get_collection(self, category: str) -> Collection:
        name = self._collection_name(category)
        if name not in self._collections:
            self._collections[name] = Collection(name)
        return self._collections[name]

    def search(
        self,
        category: str,
        vector: np.ndarray,
        top_k: int,
        params: Optional[dict] = None,
    ) -> list[tuple[int, float]]:
        collection = self._get_collection(category)
        collection.load()
        search_params = params or {"metric_type": "IP", "params": {"ef": 64}}
        results = collection.search(
            [vector],
            "embedding",
            search_params,
            limit=top_k,
            output_fields=["place_id"],
        )
        hits = results[0] if results else []
        return [(int(hit.id), float(hit.score)) for hit in hits]

    def get_vectors(self, category: str, place_ids: Iterable[int]) -> dict[int, np.ndarray]:
        ids = [int(pid) for pid in place_ids]
        if not ids:
            return {}
        collection = self._get_collection(category)
        expr = f"place_id in [{', '.join(str(pid) for pid in ids)}]"
        rows = collection.query(expr, output_fields=["place_id", "embedding"])
        return {
            int(row["place_id"]): np.array(row["embedding"], dtype=np.float32)
            for row in rows
            if row.get("embedding") is not None
        }


@lru_cache(maxsize=1)
def get_milvus_store() -> MilvusStore:
    host = os.getenv("MILVUS_HOST", "")
    if not host:
        raise ValueError("MILVUS_HOST is not set")
    port = int(os.getenv("MILVUS_PORT", "19530"))
    user = os.getenv("MILVUS_USER")
    password = os.getenv("MILVUS_PASSWORD")
    secure = os.getenv("MILVUS_SECURE", "false").lower() == "true"
    prefix = os.getenv("MILVUS_COLLECTION_PREFIX", "poi_")
    return MilvusStore(host=host, port=port, user=user, password=password, secure=secure, collection_prefix=prefix)
