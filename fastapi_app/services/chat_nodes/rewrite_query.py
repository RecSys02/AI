import json
from typing import Dict

from services.chat_nodes.callbacks import build_callbacks_config
from services.chat_nodes.llm_clients import detect_llm
from services.chat_nodes.state import GraphState
from utils.geo import append_node_trace_result


async def rewrite_query_node(state: GraphState) -> Dict:
    """Rewrite the user query with conversational context for better intent and retrieval."""
    query = state.get("query", "")
    context = state.get("context") or {}
    callbacks = state.get("callbacks")
    config = build_callbacks_config(callbacks)

    last_place = context.get("last_resolved_name")
    last_mode = context.get("last_mode")
    context_hint = (
        "문맥 정보: "
        f"이전 장소={last_place or '없음'}, "
        f"이전 카테고리={last_mode or '없음'}"
    )

    messages = [
        (
            "system",
            "너는 사용자의 모호한 질문을 대화 문맥을 바탕으로 구체적인 검색용 질문으로 재구성하는 전문가야.\n"
            "이전 대화의 장소나 주제가 현재 질문과 연결된다면, 생략된 정보를 채워 넣어라.\n"
            f"{context_hint}\n"
            "1. **구체적 지명 보존**: '반포 자이', '삼성의원', '강남역 1번 출구'와 같은 구체적인 건물, 아파트명, 지점 정보는 절대 생략하거나 광역 지명(예: 강남)으로 축소하지 마라.\n"
            "2. **의도 명확화**: '놀만한 거'는 '놀거리/명소'로, '맛있는 곳'은 '맛집/식당'으로 검색에 유리한 단어로 치환하라.\n"
            "3. **오타 수정**: 메뉴/행동/의도 표현의 오타만 수정하라. 지명/상호/역명 등 고유명사는 절대 수정하지 마라.\n"
            "4. **메뉴 강조**: '짜장면', '방어' 같은 구체적 메뉴가 있다면 이를 문장의 핵심으로 유지하라.\n"
            "JSON 형식만 반환: {\"normalized_query\": \"...\"}",
        ),
        ("user", query),
    ]

    normalized = query
    try:
        resp = await detect_llm.ainvoke(messages, max_tokens=80, config=config)
        raw = (resp.content or "").strip()
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("normalized_query"):
            normalized = str(data["normalized_query"]).strip()
    except Exception:
        pass

    result = {"normalized_query": normalized}
    append_node_trace_result(query, "rewrite_query", result)
    return result
