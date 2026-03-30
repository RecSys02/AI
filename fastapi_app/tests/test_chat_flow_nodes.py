import asyncio
import importlib
import pathlib
import sys
import types
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_flow_nodes_module():
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
    dummy_llm_clients.parse_json_response = lambda raw: None
    dummy_llm_clients.rate_limit_fallback_text = lambda: "잠시 후 다시 시도해 주세요."
    sys.modules["services.chat_nodes.llm_clients"] = dummy_llm_clients

    dummy_retriever = types.ModuleType("services.retriever")

    async def dummy_retrieve(*args, **kwargs):
        return []

    dummy_retriever.retrieve = dummy_retrieve
    sys.modules["services.retriever"] = dummy_retriever

    for module_name in [
        "services.chat_nodes.answer",
        "services.chat_nodes.extract_place",
        "services.chat_nodes.general_answer",
        "services.chat_nodes.place_llm",
        "services.chat_nodes.flow_nodes",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    return importlib.import_module("services.chat_nodes.flow_nodes")


def test_route_intent_detects_followup_recommend():
    flow_nodes = _load_flow_nodes_module()
    state = {
        "query": "거기 말고 다른 카페 추천해줘",
        "context": {
            "last_anchor": {"centers": [[37.5, 127.0]], "radius_by_intent": {"cafe": 2.0}},
            "last_recommended_names": ["테스트 카페"],
        },
        "anaphora_detected": True,
        "place": None,
    }

    result = asyncio.run(flow_nodes.route_intent_node(state))

    assert result["route_intent"] == "followup_recommend"
    assert result["intent"] == "recommend"


def test_route_intent_detects_place_detail():
    flow_nodes = _load_flow_nodes_module()
    state = {
        "query": "몽탄 어때?",
        "context": {},
        "place": {"place": "몽탄"},
        "anaphora_detected": False,
    }

    result = asyncio.run(flow_nodes.route_intent_node(state))

    assert result["route_intent"] == "place_detail"


def test_recommend_cache_round_trip():
    flow_nodes = _load_flow_nodes_module()
    state = {
        "query": "성수 카페 추천해줘",
        "normalized_query": "성수 카페 추천해줘",
        "route_intent": "recommendation",
        "mode": "cafe",
        "place": {"place": "성수"},
        "resolved_name": "성수",
        "anchor": {
            "centers": [[37.5447, 127.0557]],
            "radius_by_intent": {"cafe": 2.0},
            "source": "test",
        },
        "retrievals": [
            {
                "place_id": 101,
                "category": "cafe",
                "score": 0.95,
                "meta": {"name": "테스트 카페", "description": "설명"},
            }
        ],
        "final": "성수에서 가볼 만한 카페로 테스트 카페를 추천드려요.",
    }

    saved = asyncio.run(flow_nodes.save_recommend_cache_node(state))
    checked = asyncio.run(flow_nodes.check_recommend_cache_node(state))

    assert saved["recommend_cache_saved"] is True
    assert checked["recommend_cache_hit"] is True
    assert checked["final"] == state["final"]
    assert checked["retrievals"][0]["place_id"] == 101
