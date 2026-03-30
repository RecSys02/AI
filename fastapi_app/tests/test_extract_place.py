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

from services.chat_nodes import extract_place


async def _run(state):
    return await extract_place.extract_place_node(state)


def test_extract_place_corrects_typo(monkeypatch):
    async def fake_extract(query, callbacks=None):
        return {"area": None, "point": "코액스"}

    async def fake_correct(query, area, point, callbacks=None):
        return {"area": None, "point": "코엑스", "changed": True}

    monkeypatch.setattr(extract_place, "llm_extract_place", fake_extract)
    monkeypatch.setattr(extract_place, "llm_correct_place", fake_correct)

    result = asyncio.run(_run({"query": "코액스 주변 놀거리"}))

    assert result["place"] == {"area": None, "point": "코엑스"}
    assert result["place_original"] == {"area": None, "point": "코액스"}
    assert result["anaphora_detected"] is False
    assert result["place_confidence"] > 0.6
