import asyncio
import os
from functools import lru_cache
from typing import Iterable, Optional

import numpy as np
from pymilvus import AsyncMilvusClient, Collection, connections


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
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._secure = secure
        self._collection_prefix = collection_prefix
        scheme = "https" if secure else "http"
        self._uri = f"{scheme}://{host}:{port}"
        self._sync_connected = False
        self._collections: dict[str, Collection] = {}
        self._async_client = AsyncMilvusClient(
            uri=self._uri,
            user=user or "",
            password=password or "",
        )
        self._async_loaded_collections: set[str] = set()
        self._async_collection_locks: dict[str, asyncio.Lock] = {}

    def _collection_name(self, category: str) -> str:
        return f"{self._collection_prefix}{category}"

    def _ensure_sync_connection(self) -> None:
        if self._sync_connected:
            return
        connections.connect(
            alias=self._alias,
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            secure=self._secure,
        )
        self._sync_connected = True

    def _get_collection(self, category: str) -> Collection:
        self._ensure_sync_connection()
        name = self._collection_name(category)
        if name not in self._collections:
            self._collections[name] = Collection(name)
        return self._collections[name]

    async def _ensure_async_collection_loaded(self, category: str) -> str:
        name = self._collection_name(category)
        if name in self._async_loaded_collections:
            return name
        lock = self._async_collection_locks.get(name)
        if lock is None:
            lock = asyncio.Lock()
            self._async_collection_locks[name] = lock
        async with lock:
            if name in self._async_loaded_collections:
                return name
            await self._async_client.load_collection(name)
            self._async_loaded_collections.add(name)
        return name

    def search(
        self,
        category: str,
        vector: np.ndarray,
        top_k: int,
        params: Optional[dict] = None,
    ) -> list[tuple[int, float]]:
        collection = self._get_collection(category)
        collection.load()
        search_params = params.copy() if params else {"metric_type": "IP", "params": {}}
        if "metric_type" not in search_params:
            search_params["metric_type"] = "IP"
        inner_params = dict(search_params.get("params") or {})
        ef = inner_params.get("ef")
        min_ef = max(64, top_k)
        if ef is None or ef < min_ef:
            inner_params["ef"] = min_ef
        search_params["params"] = inner_params
        results = collection.search(
            [vector],
            "embedding",
            search_params,
            limit=top_k,
            output_fields=["place_id"],
        )
        hits = results[0] if results else []
        return [(int(hit.id), float(hit.score)) for hit in hits]

    async def search_async(
        self,
        category: str,
        vector: np.ndarray,
        top_k: int,
        params: Optional[dict] = None,
    ) -> list[tuple[int, float]]:
        name = await self._ensure_async_collection_loaded(category)
        search_params = params.copy() if params else {"metric_type": "IP", "params": {}}
        if "metric_type" not in search_params:
            search_params["metric_type"] = "IP"
        inner_params = dict(search_params.get("params") or {})
        ef = inner_params.get("ef")
        min_ef = max(64, top_k)
        if ef is None or ef < min_ef:
            inner_params["ef"] = min_ef
        search_params["params"] = inner_params
        results = await self._async_client.search(
            collection_name=name,
            data=[vector.tolist()],
            anns_field="embedding",
            search_params=search_params,
            limit=top_k,
            output_fields=["place_id"],
        )
        hits = results[0] if results else []
        parsed: list[tuple[int, float]] = []
        for hit in hits:
            entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else {}
            raw_id = hit.get("id")
            if raw_id is None:
                raw_id = hit.get("place_id")
            if raw_id is None and entity:
                raw_id = entity.get("place_id")
            if raw_id is None:
                continue
            raw_score = hit.get("score")
            if raw_score is None:
                raw_score = hit.get("distance")
            if raw_score is None:
                raw_score = hit.get("similarity")
            if raw_score is None:
                continue
            parsed.append((int(raw_id), float(raw_score)))
        return parsed

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
