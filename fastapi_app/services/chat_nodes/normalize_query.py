import json
from typing import Dict

from services.chat_nodes.callbacks import build_callbacks_config
from services.chat_nodes.llm_clients import detect_llm
from services.chat_nodes.state import GraphState
from utils.geo import append_node_trace_result


async def normalize_query_node(state: GraphState) -> Dict:
    """Normalize the user query with LLM guidance while preserving critical place details."""
    query = state.get("query", "")
    callbacks = state.get("callbacks")
    config = build_callbacks_config(callbacks)
    # LLM prompt is tuned to keep specific place names and improve search-friendly phrasing.
    messages = [
        (
            "system",
            "너는 검색 엔진을 위한 쿼리 최적화 전문가야. 사용자 질문을 아래 규칙에 따라 정규화하라.\n\n"
            "1. **구체적 지명 보존**: '반포 자이', '삼성의원', '강남역 1번 출구'와 같은 구체적인 건물, 아파트명, 지점 정보는 절대 생략하거나 광역 지명(예: 강남)으로 축소하지 마라.\n"
            "2. **의도 명확화**: '놀만한 거'는 '놀거리/명소'로, '맛있는 곳'은 '맛집/식당'으로 검색에 유리한 단어로 치환하라.\n"
            "3. **오타 수정**: 메뉴/행동/의도 표현의 오타만 수정하라. 지명/상호/역명 등 고유명사는 절대 수정하지 마라.\n"
            "4. **메뉴 강조**: '짜장면', '방어' 같은 구체적 메뉴가 있다면 이를 문장의 핵심으로 유지하라.\n"
            "JSON 형식만 반환: {\"normalized_query\": \"...\"}"
        ),
        ("user", query),
    ]
    normalized = query
    try:
        # Use a short, deterministic completion for consistent normalization.
        resp = await detect_llm.ainvoke(messages, max_tokens=80, config=config)
        raw = (resp.content or "").strip()
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("normalized_query"):
            normalized = str(data["normalized_query"]).strip()
    except Exception:
        # On any failure, fall back to the raw user query.
        pass
    result = {"normalized_query": normalized}
    append_node_trace_result(query, "normalize_query", result)
    return result
