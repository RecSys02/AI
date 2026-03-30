import asyncio
import importlib
import pathlib
import sys
import types
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_general_answer_module():
    dummy_llm_clients = types.ModuleType("services.chat_nodes.llm_clients")

    class DummyRateLimitError(Exception):
        pass

    class DummyLLM:
        async def astream(self, messages, config=None):
            if False:
                yield None

    class DummyDetectLLM:
        async def ainvoke(self, messages, **kwargs):
            return SimpleNamespace(content="unknown")

    dummy_llm_clients.llm = DummyLLM()
    dummy_llm_clients.detect_llm = DummyDetectLLM()
    dummy_llm_clients.LLMRateLimitError = DummyRateLimitError
    dummy_llm_clients.max_tokens_kwargs = lambda max_tokens: {"max_tokens": max_tokens}
    dummy_llm_clients.rate_limit_fallback_text = lambda: "잠시 후 다시 시도해 주세요."
    sys.modules["services.chat_nodes.llm_clients"] = dummy_llm_clients

    dummy_retriever = types.ModuleType("services.retriever")
    async def dummy_retrieve(*args, **kwargs):
        return []
    dummy_retriever.retrieve = dummy_retrieve
    sys.modules["services.retriever"] = dummy_retriever

    for module_name in ["services.chat_nodes.general_answer", "services.chat_nodes.mode"]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    return importlib.import_module("services.chat_nodes.general_answer")


def test_general_answer_returns_capability_message_for_region_non_recommend():
    general_answer_module = _load_general_answer_module()
    general_answer_node = general_answer_module.general_answer_node
    state = {
        "query": "강남 계획 짜줘",
        "normalized_query": "강남 계획 짜줘",
        "context": {},
        "history_place_ids": [],
    }

    async def _collect():
        chunks = []
        async for chunk in general_answer_node(state):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())
    assert chunks
    assert "지역 기반 추천만 지원" in chunks[0]["final"]
    assert "카페, 식당, 여행지 중 하나만" in chunks[0]["final"]


def test_general_answer_offloads_blocking_retrieve(monkeypatch):
    general_answer_module = _load_general_answer_module()
    general_answer_node = general_answer_module.general_answer_node
    called = {}

    async def fake_retrieve(*args, **kwargs):
        called["kwargs"] = kwargs
        return [
            {
                "place_id": 1,
                "category": "restaurant",
                "score": 0.9,
                "meta": {"name": "테스트 식당", "description": "설명"},
            }
        ]

    class FakeLLM:
        async def astream(self, messages, config=None):
            yield SimpleNamespace(content="테스트 답변")

    monkeypatch.setattr(general_answer_module, "detect_mode", lambda mode, query: "restaurant")
    monkeypatch.setattr(general_answer_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(general_answer_module, "llm", FakeLLM())

    state = {
        "query": "맛집 특징 설명해줘",
        "normalized_query": "맛집 특징 설명해줘",
        "context": {},
        "history_place_ids": [],
    }

    async def _collect():
        chunks = []
        async for chunk in general_answer_node(state):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())

    assert called["kwargs"]["mode"] == "restaurant"
    assert chunks[-2]["final"] == "테스트 답변"


def test_general_answer_returns_rate_limit_fallback(monkeypatch):
    general_answer_module = _load_general_answer_module()
    general_answer_node = general_answer_module.general_answer_node

    async def fake_retrieve(*args, **kwargs):
        return [
            {
                "place_id": 1,
                "category": "restaurant",
                "score": 0.9,
                "meta": {"name": "테스트 식당", "description": "설명"},
            }
        ]

    class RateLimitedLLM:
        async def astream(self, messages, config=None):
            raise general_answer_module.LLMRateLimitError("rate limited")
            if False:
                yield None

    monkeypatch.setattr(general_answer_module, "detect_mode", lambda mode, query: "restaurant")
    monkeypatch.setattr(general_answer_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(general_answer_module, "llm", RateLimitedLLM())

    state = {
        "query": "맛집 특징 설명해줘",
        "normalized_query": "맛집 특징 설명해줘",
        "context": {},
        "history_place_ids": [],
    }

    async def _collect():
        chunks = []
        async for chunk in general_answer_node(state):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())

    assert chunks[-2]["final"] == "잠시 후 다시 시도해 주세요."
