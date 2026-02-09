import re
from typing import Dict

from services.chat_nodes.callbacks import build_callbacks_config
from services.chat_nodes.llm_clients import detect_llm, max_tokens_kwargs, parse_json_response
from services.chat_nodes.state import GraphState
from utils.geo import append_node_trace_result


async def rewrite_query_node(state: GraphState) -> Dict:
    """Rewrite the user query with conversational context for better intent and retrieval."""
    raw_query = str(state.get("query", "")).strip()

    def _strip_prompt_leak(text: str) -> str:
        if not text:
            return ""
        markers = (
            "핵심 규칙",
            "반환 형식",
            "결과는 반드시 JSON",
            "normalized_query",
            "의미 없는 입력 처리",
            "생략된 맥락 복원",
            "의도 명확화",
            "검색 최적화",
            "고유명사 보존",
            "너는 사용자의 질문을",
        )
        first_idx = None
        for marker in markers:
            idx = text.find(marker)
            if idx != -1 and (first_idx is None or idx < first_idx):
                first_idx = idx
        if first_idx is not None:
            prefix = text[:first_idx].strip()
            if prefix:
                return prefix
        hit_count = sum(1 for marker in markers if marker in text)
        if hit_count >= 2:
            return ""
        if re.search(r'\{\s*"normalized_query"\s*:', text):
            return ""
        return text

    query = _strip_prompt_leak(raw_query)
    trace_query = raw_query or query

    def _is_meaningless(text: str) -> bool:
        stripped = re.sub(r"\s+", "", text)
        if not stripped:
            return True
        if len(stripped) <= 1:
            return True
        if re.fullmatch(r"[ㅋㅎㅠㅜㅇ]+", stripped):
            return True
        if not re.search(r"[0-9A-Za-z가-힣]", stripped):
            return True
        return False

    if _is_meaningless(query):
        result = {"normalized_query": ""}
        append_node_trace_result(trace_query, "rewrite_query", result)
        return result
    context = state.get("context") or {}
    callbacks = state.get("callbacks")
    config = build_callbacks_config(callbacks)

    last_place = context.get("last_resolved_name")
    last_mode = context.get("last_mode")
    last_normalized_query = context.get("last_normalized_query")
    if context.get("last_recommended_names") is not None:
        state["last_recommended_names"] = context.get("last_recommended_names")
    history = []
    for msg in state.get("messages") or []:
        role = str(msg.get("role") or "").strip()
        content = _strip_prompt_leak(str(msg.get("content") or "").strip())
        if not content:
            continue
        if role != "user":
            continue
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
            "우선순위: 현재 입력이 1순위이며, 문맥/대화 기록은 누락된 정보만 최소로 보완하는 참고용이다.\n"
            "핵심 규칙:\n"
            "0. **의미 없는 입력 처리**: 입력이 장소/카테고리/의도를 전혀 포함하지 않으면 정규화하지 말고 "
            "normalized_query를 빈 문자열로 반환하라.\n"
            "1. **생략된 맥락 복원**: 사용자가 '카페는?', '맛집은?'처럼 장소 없이 묻는다면 문맥 정보의 '이전 장소'를 결합해 "
            "'강남역 근처 카페 추천'처럼 바꿔라.\n"
            "1-1. **현재 장소 우선**: 사용자의 입력에 장소/지명이 포함되어 있으면 이전 장소/이전 정규화 쿼리는 사용하지 말고 "
            "현재 입력만 기반으로 정규화하라.\n"
            "1-2. **이전 활동 표현**: '갔다가/이후/끝나고/먹고' 등이 있으면 앞선 활동은 추천 대상이 아니다. "
            "'카페 추천'처럼 축소하지 말고 '카페 이후 갈만한 장소/놀거리'처럼 다음 장소 요청을 유지하라. "
            "단, '또/다시 카페'처럼 동일 카테고리를 명시하면 그대로 반영하라.\n"
            "2. **의도 명확화**: 단순히 '장소+맛집' 형식(예: 도봉구 맛집)으로 질문하면, "
            "'도봉구 맛집 추천해줘'처럼 추천 의도가 명확히 드러나게 문장을 완성하라. "
            "단, 1-2 규칙이 있는 경우 이를 우선한다.\n"
            "3. **범주 과잉추론 금지**: 입력에 '장소/곳/스팟/코스' 등 일반 표현이 있으면 특정 카테고리로 바꾸지 마라.\n"
            "4. **검색 최적화**: '놀거리/명소', '맛집/식당' 등 검색 시스템이 사용하는 단어를 활용하라.\n"
            "5. **고유명사 보존**: 지명, 상호명은 절대 수정하거나 축소하지 마라.\n"
            "예시: 입력이 \"a\", \"음\", \"ㅋㅋ\", \"?\"라면 {\"normalized_query\": \"\"}를 반환한다.\n"
            "예시: 입력이 \"카페 갔다가 갈만한 장소 추천해줘\"라면 "
            "{\"normalized_query\": \"카페 이후 갈만한 장소 추천해줘\"}처럼 다음 장소 요청을 유지한다.\n"
            "결과는 반드시 JSON 형식으로만 반환하라. 다른 텍스트/설명/코드블록 금지.\n"
            "반환 형식: {\"normalized_query\": \"...\"}",
        ),
        ("user", query),
    ]

    normalized = query
    try:
        resp = await detect_llm.ainvoke(messages, **max_tokens_kwargs(80), config=config)
        raw = (resp.content or "").strip()
        data = parse_json_response(raw)
        if isinstance(data, dict) and "normalized_query" in data:
            value = data.get("normalized_query")
            if value is None:
                normalized = ""
            else:
                normalized = str(value).strip()
    except Exception:
        pass

    if _is_meaningless(normalized):
        normalized = ""

    result = {"normalized_query": normalized}
    append_node_trace_result(trace_query, "rewrite_query", result)
    return result
