import asyncio
import importlib
import pathlib
import sys
import types
from contextlib import asynccontextmanager

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_retriever_module():
    dummy_st = types.ModuleType("sentence_transformers")

    class DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

    dummy_st.SentenceTransformer = DummySentenceTransformer
    sys.modules["sentence_transformers"] = dummy_st

    dummy_stores = types.ModuleType("services.stores")
    dummy_stores.get_milvus_store = lambda: None
    dummy_stores.get_postgres_store = lambda: None
    sys.modules["services.stores"] = dummy_stores

    if "services.retriever" in sys.modules:
        del sys.modules["services.retriever"]

    return importlib.import_module("services.retriever")


def test_retrieve_runs_async_store_and_milvus_pipeline(monkeypatch):
    retriever = _load_retriever_module()
    calls = []

    class FakePostgresStore:
        @asynccontextmanager
        async def async_connection(self):
            calls.append("open_conn")
            yield object()
            calls.append("close_conn")

        async def fetch_names_async(self, place_ids, category, conn=None):
            calls.append(("fetch_names_async", list(place_ids), category, conn is not None))
            return ["최근장소"]

        async def fts_scores_async(self, query_text, category, limit, place_ids=None, conn=None):
            calls.append(("fts_scores_async", query_text, category, limit, conn is not None))
            return {101: 4.0, 202: 1.0}

        async def fetch_meta_async(self, place_ids, category, conn=None):
            calls.append(("fetch_meta_async", list(place_ids), category, conn is not None))
            return {
                101: {"place_id": 101, "meta": {"name": "알파", "description": "파스타 맛집"}},
                202: {"place_id": 202, "meta": {"name": "베타", "description": "브런치 카페"}},
            }

    class FakeMilvusStore:
        async def search_async(self, category, vector, top_k, params=None):
            calls.append(("search_async", category, top_k, isinstance(vector, np.ndarray)))
            return [(101, 0.9), (202, 0.2)]

    async def fake_encode(query_text_embed):
        calls.append(("encode_async", query_text_embed))
        return np.array([0.1, 0.2, 0.3], dtype=float), {
            "load_model_ms": 1.0,
            "encode_ms": 2.0,
        }

    monkeypatch.setattr(retriever, "get_postgres_store", lambda: FakePostgresStore())
    monkeypatch.setattr(retriever, "get_milvus_store", lambda: FakeMilvusStore())
    monkeypatch.setattr(retriever, "_encode_query_async", fake_encode)

    timings = {}
    result = asyncio.run(
        retriever.retrieve(
            query="강남 파스타 추천",
            mode="restaurant",
            top_k=1,
            history_place_ids=[7],
            timings=timings,
        )
    )

    assert result
    assert result[0]["place_id"] == 101
    assert result[0]["meta"]["name"] == "알파"
    assert ("fetch_names_async", [7], "restaurant", True) in calls
    assert any(call[0] == "fts_scores_async" for call in calls if isinstance(call, tuple))
    assert any(call[0] == "search_async" for call in calls if isinstance(call, tuple))
    assert any(call[0] == "fetch_meta_async" for call in calls if isinstance(call, tuple))
    assert timings["load_model_ms"] == 1.0
    assert "dense_search_ms" in timings


def test_geohash_prefilter_keeps_nearby_candidates():
    retriever = _load_retriever_module()
    center = [[37.5665, 126.9780]]
    candidate_ids = [1, 2]
    meta_map = {
        1: {"lat": 37.5670, "lng": 126.9785},
        2: {"lat": 37.6200, "lng": 127.0500},
    }

    filtered_ids, precision, prefix_count = retriever._apply_geohash_prefilter(
        candidate_ids,
        meta_map,
        center,
        radius_km=2.0,
    )

    assert 1 in filtered_ids
    assert 2 not in filtered_ids
    assert precision == 5
    assert prefix_count > 0
