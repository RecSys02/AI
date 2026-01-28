from typing import Dict

from services.chat_nodes.callbacks import build_callbacks_config
from services.chat_nodes.llm_clients import detect_llm, max_tokens_kwargs, parse_json_response
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
    last_normalized_query = context.get("last_normalized_query")
    history = []
    for msg in state.get("messages") or []:
        role = str(msg.get("role") or "").strip()
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        if role not in {"user", "assistant", "system"}:
            role = "user"
        history.append({"role": role, "content": content})
    if history and history[-1]["role"] == "user" and history[-1]["content"] == query:
        history = history[:-1]
    history = history[-3:]
    history_hint = ""
    if history:
        history_lines = [f"- {item['role']}: {item['content']}" for item in history]
        history_hint = "최근 대화 기록:\n" + "\n".join(history_lines)
    context_hint = (
        "문맥 정보: "
        f"이전 장소={last_place or '없음'}, "
        f"이전 카테고리={last_mode or '없음'}, "
        f"이전 정규화 쿼리={last_normalized_query or '없음'}"
    )

    messages = [
        (
            "system",
            "너는 사용자의 질문을 검색 엔진과 의도 분류기가 이해하기 쉽게 '완결된 문장'으로 재구성하는 전문가야.\n"
            f"{context_hint}\n"
            f"{history_hint}\n"
            "핵심 규칙:\n"
            "1. **생략된 맥락 복원**: 사용자가 '카페는?', '맛집은?'처럼 장소 없이 묻는다면 문맥 정보의 '이전 장소'를 결합해 '강남역 근처 카페 추천'처럼 바꿔라.\n"
            "2. **의도 명확화**: 단순히 '장소+맛집' 형식(예: 도봉구 맛집)으로 질문하면, '도봉구 맛집 추천해줘'처럼 추천 의도가 명확히 드러나게 문장을 완성하라.\n"
            "3. **검색 최적화**: '놀거리/명소', '맛집/식당' 등 검색 시스템이 사용하는 단어를 활용하라.\n"
            "4. **고유명사 보존**: 지명, 상호명은 절대 수정하거나 축소하지 마라.\n"
            "결과는 반드시 JSON 형식으로만 반환하라: {\"normalized_query\": \"...\"}",
        ),
        ("user", query),
    ]

    normalized = query
    try:
        resp = await detect_llm.ainvoke(messages, **max_tokens_kwargs(80), config=config)
        raw = (resp.content or "").strip()
        data = parse_json_response(raw)
        if isinstance(data, dict) and data.get("normalized_query"):
            normalized = str(data["normalized_query"]).strip()
    except Exception:
        pass

    result = {"normalized_query": normalized}
    append_node_trace_result(query, "rewrite_query", result)
    return result
