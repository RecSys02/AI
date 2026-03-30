import re
from typing import Dict, Iterable, List

from services.chat_nodes.callbacks import build_callbacks_config
from services.chat_nodes.intent import is_expand_query
from services.chat_nodes.llm_clients import detect_llm, max_tokens_kwargs, parse_json_response
from services.chat_nodes.mode import detect_mode
from services.chat_nodes.state import GraphState
from utils.geo import append_node_trace_result

PROMPT_LEAK_MARKERS = (
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
    "우선순위:",
    "문맥 정보:",
    "최근 대화 기록:",
)

PROMPT_LEAK_PREFIXES = (
    "너는 사용자의 질문을",
    "문맥 정보:",
    "최근 대화 기록:",
    "우선순위:",
    "핵심 규칙:",
)

LOCATION_SUFFIXES = (
    "도",
    "시",
    "군",
    "구",
    "읍",
    "면",
    "동",
    "리",
    "가",
    "역",
    "로",
    "길",
)

INTENT_KEYWORDS = (
    "추천",
    "알려",
    "어디",
    "코스",
    "계획",
    "일정",
    "여행",
    "관광",
    "놀거리",
    "명소",
    "맛집",
    "식당",
    "카페",
)

CATEGORY_HINT_WORDS = (
    "카페",
    "맛집",
    "식당",
    "관광지",
    "명소",
    "놀거리",
    "여행지",
    "코스",
    "장소",
)

WESTERN_CUISINE_HINTS = (
    "양식",
    "양식집",
    "서양식",
    "서양음식",
    "이탈리안",
    "파스타",
    "피자",
    "스테이크",
    "프렌치",
    "비스트로",
    "레스토랑",
)

MODE_KEYWORDS = {
    "restaurant": (
        "맛집",
        "식당",
        "레스토랑",
        "밥",
        "점심",
        "저녁",
        "한식",
        "중식",
        "일식",
        "양식",
        "파스타",
        "피자",
        "스테이크",
    ),
    "cafe": ("카페", "커피", "디저트", "브런치", "베이커리"),
    "tourspot": ("관광지", "명소", "놀거리", "여행지", "코스", "장소"),
}

MODE_LABELS = {
    "restaurant": "식당",
    "cafe": "카페",
    "tourspot": "관광지",
}

RECOMMEND_LIKE_HINTS = (
    "갈만",
    "뭐할",
    "뭐 하지",
    "뭐하지",
    "어디 갈",
    "데이트",
    "아이랑",
    "가볼",
)

RECOMMEND_PHRASES = (
    "추천",
    "어디",
    "가볼",
    "뭐가 있어",
    "top",
    "best",
)


def _has_prompt_leak(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith(PROMPT_LEAK_PREFIXES):
        return True
    hit_count = sum(1 for marker in PROMPT_LEAK_MARKERS if marker in stripped)
    return hit_count >= 2


def _strip_prompt_leak(text: str) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if stripped.startswith(PROMPT_LEAK_PREFIXES):
        return ""

    first_idx = None
    for marker in PROMPT_LEAK_MARKERS:
        idx = stripped.find(marker)
        if idx != -1 and (first_idx is None or idx < first_idx):
            first_idx = idx

    if first_idx is not None:
        prefix = stripped[:first_idx].strip()
        if prefix.lower().startswith("user:"):
            prefix = prefix.split(":", 1)[-1].strip()
        if prefix:
            return prefix

    if _has_prompt_leak(stripped):
        return ""
    if re.search(r'\{\s*"normalized_query"\s*:', stripped):
        return ""
    return stripped


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


def _extract_route_segments(text: str) -> List[str]:
    if "->" not in text and "→" not in text and "⇒" not in text and "➡" not in text:
        return []
    segments = re.split(r"\s*(?:->|→|⇒|➡)\s*", text)
    cleaned = []
    for segment in segments:
        part = re.sub(r"[\"'`.,!?]", "", segment).strip()
        if len(part) >= 2:
            cleaned.append(part)
    return cleaned


def _extract_location_tokens(text: str) -> List[str]:
    tokens = set()
    words = re.findall(r"[0-9A-Za-z가-힣]+", text)
    for word in words:
        if len(word) < 2:
            continue
        if any(word.endswith(suffix) for suffix in LOCATION_SUFFIXES):
            tokens.add(word)

    for match in re.finditer(
        r"([0-9A-Za-z가-힣]{2,})\s*(?:에서|으로|쪽|근처|부근|인근)",
        text,
    ):
        tokens.add(match.group(1))

    for match in re.finditer(
        r"([0-9A-Za-z가-힣]{2,})\s*(?:근처\s*)?(?:맛집|식당|카페|관광지|명소|놀거리|여행|코스)",
        text,
    ):
        tokens.add(match.group(1))

    return sorted(tokens)


def _has_intent_signal(text: str) -> bool:
    return any(keyword in text for keyword in INTENT_KEYWORDS)


def _has_recommend_phrase(text: str) -> bool:
    lowered = str(text or "").lower()
    if any(token in lowered for token in RECOMMEND_PHRASES):
        return True
    return bool(re.search(r"(\d{1,2})(개|곳|군데)", lowered))


def _should_use_history(query: str, anaphora_detected: bool = False) -> bool:
    if anaphora_detected:
        return True
    if _extract_route_segments(query):
        return False
    if _extract_location_tokens(query):
        return False
    if len(query) > 20:
        return False
    return any(word in query for word in CATEGORY_HINT_WORDS)


def _should_fallback_to_query(query: str, normalized: str) -> bool:
    if _is_meaningless(query):
        return False
    if not normalized:
        return True

    route_segments = _extract_route_segments(query)
    if route_segments and not all(segment in normalized for segment in route_segments):
        return True

    location_tokens = _extract_location_tokens(query)
    if location_tokens and not any(token in normalized for token in location_tokens):
        return True

    if _has_intent_signal(query) and not _has_intent_signal(normalized):
        return True
    if any(token in query for token in WESTERN_CUISINE_HINTS) and not any(
        token in normalized for token in WESTERN_CUISINE_HINTS
    ):
        return True
    return False


def _contains_mode_keyword(text: str, mode: str | None) -> bool:
    if not text or not mode:
        return False
    return any(token in text for token in MODE_KEYWORDS.get(mode, ()))


def _place_to_label(place: dict | None) -> str:
    if not isinstance(place, dict):
        return ""
    point = str(place.get("point") or "").strip()
    area = str(place.get("area") or "").strip()
    single = str(place.get("place") or "").strip()
    if area and point and area not in point:
        return f"{area} {point}"
    if point:
        return point
    if area:
        return area
    return single


def _place_tokens(place: dict | None) -> List[str]:
    if not isinstance(place, dict):
        return []
    tokens = []
    for key in ("place", "point", "area"):
        value = str(place.get(key) or "").strip()
        if value and value not in tokens:
            tokens.append(value)
    return tokens


def _remove_place_mentions(text: str, *places: dict | None) -> str:
    cleaned = str(text or "")
    for place in places:
        for token in _place_tokens(place):
            cleaned = cleaned.replace(token, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _build_safe_rewrite(
    raw_query: str,
    place_label: str,
    place_original: dict | None,
    explicit_mode: str | None,
    context_action: str | None,
) -> str:
    stripped = _strip_prompt_leak(raw_query).strip()
    stripped = re.sub(r"[?!.,]+$", "", stripped).strip()
    base = _remove_place_mentions(stripped, {"place": place_label}, place_original)
    category_label = MODE_LABELS.get(explicit_mode or "", "")
    if explicit_mode and _is_short_category_followup(stripped, explicit_mode):
        return f"{place_label} 근처 {category_label} 추천해줘"
    if context_action == "keep" and explicit_mode:
        return f"{place_label} 근처 {category_label} 추천해줘"
    if not base:
        if explicit_mode:
            return f"{place_label} 근처 {category_label} 추천해줘"
        return f"{place_label} 정보 알려줘"
    if explicit_mode and not _contains_mode_keyword(base, explicit_mode):
        base = f"{category_label} {base}".strip()
    if (
        not _has_recommend_phrase(base)
        and (
            explicit_mode
            or any(token in base for token in MODE_KEYWORDS.get("tourspot", ()))
            or any(hint in base for hint in RECOMMEND_LIKE_HINTS)
        )
    ):
        base = f"{base} 추천해줘"
    return f"{place_label} {base}".strip()


def _is_short_category_followup(query: str, explicit_mode: str | None) -> bool:
    if not explicit_mode:
        return False
    compact = re.sub(r"\s+", "", re.sub(r"[?!.,]+$", "", query))
    if len(compact) > 8:
        return False
    words = re.findall(r"[0-9A-Za-z가-힣]+", query)
    return len(words) <= 2 and detect_mode(None, query) == explicit_mode


def _extract_context_topic(context: dict) -> str | None:
    for key in ("last_query", "last_normalized_query"):
        raw = _strip_prompt_leak(str(context.get(key) or "")).strip()
        if not raw:
            continue
        text = re.sub(r"[?!.,]+$", "", raw).strip()
        match = re.search(r"(.+?)\s*뭐\s*할까", text)
        if match:
            prefix = match.group(1).strip()
            if prefix:
                return f"{prefix} 갈만한"
        match = re.search(r"(.+?)\s*어디\s*갈까", text)
        if match:
            prefix = match.group(1).strip()
            if prefix:
                return f"{prefix} 갈만한"
        if len(re.sub(r"\s+", "", text)) <= 16 and any(
            token in text for token in ("같이", "데이트", "엄마", "아빠", "친구", "아이", "연인")
        ):
            return text
    return None


def _build_recommend_seed(
    query: str,
    explicit_mode: str | None,
    context: dict,
    anaphora_detected: bool = False,
    has_explicit_place: bool = False,
) -> str | None:
    cleaned = re.sub(r"\s+", " ", _strip_prompt_leak(query)).strip()
    cleaned = re.sub(r"[?!.,]+$", "", cleaned).strip()
    if not cleaned:
        return None

    last_place = str(context.get("last_resolved_name") or "").strip()
    category_label = MODE_LABELS.get(explicit_mode or "", cleaned)

    if anaphora_detected and last_place:
        if explicit_mode:
            return f"{last_place} 근처 {category_label} 추천해줘"
        return f"{last_place} 근처 추천해줘"

    if explicit_mode and _is_short_category_followup(cleaned, explicit_mode):
        topic = _extract_context_topic(context)
        if topic:
            return f"{topic} {category_label} 추천해줘"
        if last_place and not has_explicit_place:
            return f"{last_place} 근처 {category_label} 추천해줘"
        return f"{category_label} 추천해줘"

    if explicit_mode and not _has_intent_signal(cleaned):
        if has_explicit_place:
            return f"{cleaned} 추천해줘"
        if last_place and cleaned == category_label:
            return f"{last_place} 근처 {category_label} 추천해줘"
        return f"{cleaned} 추천해줘"
    return None


def _collect_history(messages: Iterable[dict], query: str) -> List[dict]:
    history: List[dict] = []
    for msg in messages:
        role = str(msg.get("role") or "").strip()
        if role != "user":
            continue

        raw_content = str(msg.get("content") or "").strip()
        if not raw_content or _has_prompt_leak(raw_content):
            continue

        content = _strip_prompt_leak(raw_content)
        if not content or _is_meaningless(content):
            continue
        history.append({"role": role, "content": content})

    if history and history[-1]["content"] == query:
        history = history[:-1]
    return history


async def rewrite_query_node(state: GraphState) -> Dict:
    """Rewrite the user query using the already-decided place context."""
    raw_query = str(state.get("query", "")).strip()
    query = _strip_prompt_leak(raw_query)
    trace_query = raw_query or query

    if _is_meaningless(query):
        result = {"normalized_query": ""}
        append_node_trace_result(trace_query, "rewrite_query", result)
        return result

    context = state.get("context") or {}
    callbacks = state.get("callbacks")
    config = build_callbacks_config(callbacks)
    explicit_mode = state.get("explicit_mode")
    context_action = state.get("context_action")
    confirmed_place = state.get("place") or {}
    place_label = _place_to_label(confirmed_place)
    current_place = state.get("has_explicit_place")

    if is_expand_query(query):
        result = {"normalized_query": query}
        append_node_trace_result(trace_query, "rewrite_query", result)
        return result

    seed_rewrite = _build_safe_rewrite(
        raw_query=query,
        place_label=place_label,
        place_original=state.get("place_original"),
        explicit_mode=explicit_mode,
        context_action=context_action,
    )
    if place_label and (context_action == "keep" or _is_short_category_followup(query, explicit_mode)):
        result = {"normalized_query": seed_rewrite}
        append_node_trace_result(trace_query, "rewrite_query", result)
        return result

    history = _collect_history(state.get("messages") or [], query)
    if not _should_use_history(query, anaphora_detected=bool(state.get("anaphora_detected"))):
        history = []
    history = history[-3:]

    history_hint = ""
    if history:
        history_lines = [f"- {item['role']}: {item['content']}" for item in history]
        history_hint = "최근 대화 기록:\n" + "\n".join(history_lines)

    last_mode = context.get("last_mode")
    context_hint = (
        "문맥 정보: "
        f"확정된 기준 장소={place_label or '없음'}, "
        f"컨텍스트 결정={context_action or '없음'}, "
        f"현재 감지 카테고리={explicit_mode or '없음'}, "
        f"직전 카테고리={last_mode or '없음'}, "
        f"현재 턴에 새 장소 발견 여부={'예' if current_place else '아니오'}"
    )

    messages = [
        (
            "system",
            "너는 검색 질의를 정제하는 전문가야. 장소는 이미 앞 단계에서 확정되었다.\n"
            f"{context_hint}\n"
            f"{history_hint}\n"
            "핵심 규칙:\n"
            "1. 확정된 기준 장소를 반드시 그대로 사용하라. 다른 장소를 추가하거나 섞지 마라.\n"
            "2. 사용자의 의도와 카테고리만 정리하라. 장소 결정은 하지 마라.\n"
            "3. 현재 입력에 식당/맛집/카페/관광지 같은 카테고리 단어가 있으면 절대 삭제하지 마라.\n"
            "4. 짧은 후속 발화라면 추천 의도가 드러나는 완결 문장으로 보강하라.\n"
            "5. 기준 장소와 다른 이전 장소명이 응답에 들어가면 안 된다.\n"
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

    normalized = _strip_prompt_leak(normalized)
    previous_place = str(context.get("last_resolved_name") or "").strip()
    if place_label and place_label not in normalized:
        normalized = seed_rewrite
    if (
        place_label
        and current_place
        and previous_place
        and previous_place != place_label
        and previous_place in normalized
    ):
        normalized = seed_rewrite
    if explicit_mode and normalized and not _contains_mode_keyword(normalized, explicit_mode):
        normalized = seed_rewrite or query
    if _should_fallback_to_query(query, normalized):
        normalized = seed_rewrite or query

    if _is_meaningless(normalized):
        normalized = seed_rewrite or ""

    result = {"normalized_query": normalized}
    append_node_trace_result(trace_query, "rewrite_query", result)
    return result
