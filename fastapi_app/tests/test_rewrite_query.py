import asyncio
import os
import pathlib
import sys

os.environ.setdefault("CHAT_PROVIDER", "openai")
os.environ.setdefault("CHAT_MODEL", "gpt-4o-mini")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.chat_nodes import rewrite_query


class DummyResp:
    def __init__(self, content: str):
        self.content = content


class DummyLLM:
    def __init__(self, content: str):
        self._content = content
        self.messages = None
        self.kwargs = None

    async def ainvoke(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return DummyResp(self._content)


def _run_rewrite(monkeypatch, state, llm_content):
    dummy_llm = DummyLLM(llm_content)
    monkeypatch.setattr(rewrite_query, "detect_llm", dummy_llm)
    result = asyncio.run(rewrite_query.rewrite_query_node(state))
    return result, dummy_llm


def test_rewrite_keeps_current_route_query_when_llm_drifts(monkeypatch):
    query = "천호동 -> 길동 -> 하남 순서로 3일간의 여행 계획을 만들어봐"
    state = {
        "query": query,
        "context": {
            "last_resolved_name": "서울특별시 강동구청역앞",
            "last_mode": "tourspot",
            "last_normalized_query": "서울 강동구에서 할만한 것 추천해줘",
        },
        "messages": [
            {
                "role": "user",
                "content": (
                    "서울 강동구에서 어떤 것을 하면 좋을까요? 우선순위: 현재 입력이 1순위이며 "
                    "핵심 규칙: ... 반환 형식: {\"normalized_query\": \"...\"}"
                ),
            }
        ],
    }

    result, dummy = _run_rewrite(
        monkeypatch,
        state,
        '{"normalized_query":"서울 강동구에서 어떤 것을 하면 좋을지 추천해줘"}',
    )

    assert result["normalized_query"] == query
    # Prompt-like leaked history should not be forwarded into rewrite context.
    assert "서울 강동구에서 어떤 것을 하면 좋을까요?" not in dummy.messages[0][1]


def test_rewrite_falls_back_to_query_when_llm_returns_empty(monkeypatch):
    query = "강남 맛집 추천해줘"
    state = {"query": query, "context": {}, "messages": []}

    result, _ = _run_rewrite(monkeypatch, state, '{"normalized_query":""}')

    assert result["normalized_query"] == query


def test_rewrite_falls_back_when_location_is_lost(monkeypatch):
    query = "강남 맛집 추천해줘"
    state = {"query": query, "context": {}, "messages": []}

    result, _ = _run_rewrite(
        monkeypatch,
        state,
        '{"normalized_query":"서울 강동구에서 어떤 것을 하면 좋을지 추천해줘"}',
    )

    assert result["normalized_query"] == query


def test_rewrite_allows_context_for_short_elliptical_query(monkeypatch):
    state = {
        "query": "카페는?",
        "explicit_mode": "cafe",
        "context_action": "keep",
        "place": {"place": "강남역"},
        "context": {
            "last_resolved_name": "강남역",
            "last_mode": "restaurant",
            "last_normalized_query": "강남역 근처 맛집 추천해줘",
        },
        "messages": [{"role": "user", "content": "강남역 근처 맛집 추천해줘"}],
    }

    result, dummy = _run_rewrite(
        monkeypatch,
        state,
        '{"normalized_query":"강남역 근처 카페 추천해줘"}',
    )

    assert result["normalized_query"] == "강남역 근처 카페 추천해줘"


def test_rewrite_preserves_short_category_followup_with_context_topic(monkeypatch):
    state = {
        "query": "식당",
        "explicit_mode": "restaurant",
        "context_action": "keep",
        "place": {"place": "강남역"},
        "context": {
            "last_query": "강남역에서 뭐할까?",
            "last_normalized_query": "강남역 놀거리 추천해줘",
            "last_resolved_name": "강남역",
        },
        "messages": [],
    }

    result, _ = _run_rewrite(
        monkeypatch,
        state,
        '{"normalized_query":"엄마랑 같이 할만한 것 추천해줘"}',
    )

    assert result["normalized_query"] == "강남역 근처 식당 추천해줘"


def test_rewrite_resolves_anaphora_with_last_place(monkeypatch):
    state = {
        "query": "거기 근처 카페는?",
        "explicit_mode": "cafe",
        "context_action": "keep",
        "place": {"place": "코엑스"},
        "anaphora_detected": True,
        "context": {
            "last_resolved_name": "코엑스",
            "last_mode": "tourspot",
            "last_normalized_query": "코엑스 놀거리 추천해줘",
        },
        "messages": [],
    }

    result, _ = _run_rewrite(
        monkeypatch,
        state,
        '{"normalized_query":"카페 추천해줘"}',
    )

    assert result["normalized_query"] == "코엑스 근처 카페 추천해줘"


def test_rewrite_uses_confirmed_place_when_llm_mixes_old_context(monkeypatch):
    state = {
        "query": "코액스 주변 놀거리",
        "explicit_mode": None,
        "context_action": "flush",
        "place": {"point": "코엑스", "area": None},
        "place_original": {"point": "코액스", "area": None},
        "has_explicit_place": True,
        "context": {
            "last_resolved_name": "잠실",
            "last_mode": "restaurant",
        },
        "messages": [],
    }

    result, _ = _run_rewrite(
        monkeypatch,
        state,
        '{"normalized_query":"잠실 맛집 주변 코엑스 놀거리 추천해줘"}',
    )

    assert result["normalized_query"] == "코엑스 주변 놀거리 추천해줘"


def test_rewrite_falls_back_when_western_cuisine_intent_is_lost(monkeypatch):
    query = "강남 양식집 추천해줘"
    state = {"query": query, "context": {}, "messages": []}

    result, _ = _run_rewrite(
        monkeypatch,
        state,
        '{"normalized_query":"강남 맛집 추천해줘"}',
    )

    assert result["normalized_query"] == query
