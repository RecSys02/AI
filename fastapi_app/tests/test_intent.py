import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.chat_nodes.intent import detect_intent, is_region_non_recommend_query


def test_region_non_recommend_query_detected():
    assert is_region_non_recommend_query("강남 계획 짜줘") is True
    assert is_region_non_recommend_query("천호동 -> 길동 -> 하남 순서로 3일간의 여행 계획을 만들어봐") is True
    assert is_region_non_recommend_query("강남 여행지 추천해줘") is False


def test_detect_intent_region_non_recommend_falls_back_general():
    assert detect_intent("강남 계획 짜줘") == "general"
    assert detect_intent("천호동 -> 길동 -> 하남 순서로 3일간의 여행 계획을 만들어봐") == "general"
    assert detect_intent("강남 여행지 추천해줘") == "recommend"
