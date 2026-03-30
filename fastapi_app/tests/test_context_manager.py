import asyncio
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.chat_nodes.context_manager import context_manager_node


def _run(state):
    return asyncio.run(context_manager_node(state))


def test_context_manager_flushes_when_new_place_is_extracted():
    result = _run(
        {
            "query": "코엑스 주변 놀거리",
            "place": {"area": None, "point": "코엑스"},
            "place_original": {"area": None, "point": "코액스"},
            "anaphora_detected": False,
            "place_confidence": 0.85,
            "context": {
                "last_place": {"place": "잠실"},
                "last_resolved_name": "잠실",
            },
        }
    )

    assert result["context_action"] == "flush"
    assert result["place"] == {"area": None, "point": "코엑스"}
    assert result["needs_clarification"] is False


def test_context_manager_keeps_last_place_for_anaphora():
    result = _run(
        {
            "query": "거기 근처 카페는?",
            "place": None,
            "place_original": None,
            "anaphora_detected": True,
            "place_confidence": 0.0,
            "context": {
                "last_place": {"place": "코엑스"},
                "last_resolved_name": "코엑스",
            },
        }
    )

    assert result["context_action"] == "keep"
    assert result["place"] == {"place": "코엑스"}
    assert result["resolved_name"] == "코엑스"
    assert result["needs_clarification"] is False


def test_context_manager_requests_place_when_none_is_available():
    result = _run(
        {
            "query": "식당",
            "place": None,
            "place_original": None,
            "anaphora_detected": False,
            "place_confidence": 0.0,
            "context": {},
        }
    )

    assert result["context_action"] == "none"
    assert result["needs_clarification"] is True
    assert result["clarification_reason"] == "missing_place"
