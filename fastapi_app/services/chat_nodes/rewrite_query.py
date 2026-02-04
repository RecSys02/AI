import re
from typing import Dict

from services.chat_nodes.callbacks import build_callbacks_config
from services.chat_nodes.llm_clients import detect_llm, max_tokens_kwargs, parse_json_response
from services.chat_nodes.place_llm import llm_extract_place
from services.chat_nodes.state import GraphState
from utils.geo import append_node_trace_result, normalize_text


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

    def _has_explicit_location(text: str) -> bool:
        if re.search(r"[가-힣]{2,}(역|구|동|시|군|읍|면)", text):
            return True
        norm = normalize_text(text)
        region_hints = (
            "서울",
            "경기",
            "인천",
            "부산",
            "대구",
            "대전",
            "광주",
            "울산",
            "세종",
            "제주",
            "강원",
            "충북",
            "충남",
            "전북",
            "전남",
            "경북",
            "경남",
        )
        return any(hint in norm for hint in region_hints)

    def _has_category_hint(text: str) -> bool:
        norm = normalize_text(text)
        category_terms = (
            "카페",
            "커피",
            "디저트",
            "브런치",
            "맛집",
            "식당",
            "레스토랑",
            "밥",
            "점심",
            "저녁",
            "놀거리",
            "명소",
            "관광지",
            "볼거리",
            "핫플",
        )
        return any(term in norm for term in category_terms)

    def _has_intent_hint(text: str) -> bool:
        norm = normalize_text(text)
        intent_terms = ("추천", "어디", "가볼", "뭐가", "top", "best", "몇개", "몇곳")
        return any(term in norm for term in intent_terms)

    current_has_category = _has_category_hint(query)
    current_has_intent = _has_intent_hint(query)

    async def _has_llm_location(text: str) -> bool:
        try:
            place = await llm_extract_place(text, callbacks=callbacks)
        except Exception:
            return False
        if not place:
            return False
        return bool(place.get("area") or place.get("point"))

    has_location = await _has_llm_location(query)
    if not has_location:
        has_location = _has_explicit_location(query)
    if has_location:
        # Current query already includes a location; ignore previous context to avoid mixing.
        last_place = None
        last_mode = None
        last_normalized_query = None
        history_hint = ""
    else:
        if current_has_category:
            last_mode = None
        if current_has_category or current_has_intent:
            last_normalized_query = None
        need_history = False
        if not last_place:
            need_history = True
        if not current_has_category and not last_mode:
            need_history = True
        if not need_history:
            history_hint = ""

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
            "2. **의도 명확화**: 단순히 '장소+맛집' 형식(예: 도봉구 맛집)으로 질문하면, "
            "'도봉구 맛집 추천해줘'처럼 추천 의도가 명확히 드러나게 문장을 완성하라.\n"
            "3. **검색 최적화**: '놀거리/명소', '맛집/식당' 등 검색 시스템이 사용하는 단어를 활용하라.\n"
            "4. **고유명사 보존**: 지명, 상호명은 절대 수정하거나 축소하지 마라.\n"
            "예시: 입력이 \"a\", \"음\", \"ㅋㅋ\", \"?\"라면 {\"normalized_query\": \"\"}를 반환한다.\n"
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
