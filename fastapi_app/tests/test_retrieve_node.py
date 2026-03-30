import asyncio
import importlib
import pathlib
import sys
import types
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_retrieve_node_module():
    dummy_llm_clients = types.ModuleType("services.chat_nodes.llm_clients")

    class DummyDetectLLM:
        async def ainvoke(self, messages, **kwargs):
            return SimpleNamespace(content="unknown")

    dummy_llm_clients.detect_llm = DummyDetectLLM()
    dummy_llm_clients.max_tokens_kwargs = lambda max_tokens: {"max_tokens": max_tokens}
    sys.modules["services.chat_nodes.llm_clients"] = dummy_llm_clients

    dummy_retriever = types.ModuleType("services.retriever")
    async def dummy_retrieve(*args, **kwargs):
        return []
    dummy_retriever.retrieve = dummy_retrieve
    sys.modules["services.retriever"] = dummy_retriever

    for module_name in ["services.chat_nodes.retrieve", "services.chat_nodes.mode"]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    return importlib.import_module("services.chat_nodes.retrieve")


def test_retrieve_node_offloads_blocking_retrieve(monkeypatch):
    retrieve_module = _load_retrieve_node_module()
    called = {}

    async def fake_retrieve(*args, **kwargs):
        called["kwargs"] = kwargs
        return [{"place_id": 1, "category": "restaurant", "score": 0.7, "meta": {}}]

    monkeypatch.setattr(retrieve_module, "detect_mode", lambda mode, query: "restaurant")
    monkeypatch.setattr(retrieve_module, "retrieve", fake_retrieve)

    state = {
        "query": "강남 맛집 추천해줘",
        "normalized_query": "강남 맛집 추천해줘",
        "mode": "restaurant",
        "history_place_ids": [],
    }

    result = asyncio.run(retrieve_module.retrieve_node(state))

    assert called["kwargs"]["mode"] == "restaurant"
    assert result["retrievals"][0]["place_id"] == 1
