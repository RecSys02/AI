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

from services.chat_nodes.general_answer import general_answer_node


def test_general_answer_returns_capability_message_for_region_non_recommend():
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
