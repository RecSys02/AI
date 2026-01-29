from typing import List

from services.chat_nodes.callbacks import build_callbacks_config
from services.chat_nodes.intent import is_date_query, is_nearby_query
from services.chat_nodes.llm_clients import llm
from services.chat_nodes.state import GraphState, build_context
from utils.geo import append_node_trace_result


async def answer_node(state: GraphState):
    """Generate the final response or clarification based on retrieval results."""
    
    # 1. 원본 쿼리와 정제된 쿼리 설정
    raw_query = state.get("query", "")
    normalized_query = state.get("normalized_query")
    target_query = normalized_query if normalized_query else raw_query
    
    # 2. 예외 케이스 처리 (위치 검색 실패 등)
    if state.get("expand_failed"):
        final_text = "이전에 사용한 기준 위치가 없어서 범위를 넓힐 수 없어요. 기준 장소를 알려주세요."
        append_node_trace_result(raw_query, "answer", {"final": final_text})
        yield {"final": final_text}
        yield {"context": build_context(state)}
        return

    if state.get("anchor_failed"):
        place = state.get("resolved_name") or state.get("input_place")
        if not place:
            place_info = state.get("place") or {}
            place = place_info.get("point") or place_info.get("area")
        
        final_text = f"'{place}' 위치를 찾지 못했어요. 지점/역/건물명을 알려주세요." if place else "위치를 찾지 못했어요. 기준이 될 지점/역/건물명을 알려주세요."
        append_node_trace_result(raw_query, "answer", {"final": final_text})
        yield {"final": final_text}
        yield {"context": build_context(state)}
        return

    callbacks = state.get("callbacks")
    config = build_callbacks_config(callbacks)

    # 주변 추천 의도(정제된 쿼리 기준)인데 기준점이 없는 경우
    if is_nearby_query(target_query) and not state.get("anchor"):
        place = state.get("resolved_name") or state.get("input_place")
        if not place:
            place_info = state.get("place") or {}
            place = place_info.get("point") or place_info.get("area")
        
        final_text = f"'{place}'가 어느 지점을 말하는지 알려주세요. 기준 위치를 알려주시면 그 근처로 추천할게요." if place else "근처/주변 추천을 하려면 기준 위치가 필요해요. 지점/역/건물명을 알려주세요."
        append_node_trace_result(raw_query, "answer", {"final": final_text})
        yield {"final": final_text}
        yield {"context": build_context(state)}
        return

    retrievals = state.get("retrievals", [])
    
    # 검색 결과가 없는 경우
    if not retrievals:
        filter_applied = bool(state.get("anchor"))
        if filter_applied:
            location_name = state.get("resolved_name") or state.get("input_place") or "해당 지역"
            mode_used = state.get("mode") or "tourspot"
            category_label = {"tourspot": "놀거리", "restaurant": "맛집", "cafe": "카페"}.get(mode_used, "추천")
            final_text = f"{location_name} 근처에는 조건에 맞는 결과가 없어요. 반경을 넓혀서 다시 찾아볼까요, 아니면 {location_name}의 다른 {category_label}로 추천해드릴까요?"
        else:
            final_text = "검색된 결과가 없습니다."
        
        yield {"final": final_text}
        yield {"context": build_context(state)}
        return

    if state.get("debug"):
        yield {"debug": retrievals}

    # 3. 컨텍스트 구성 방식 (태그 기반 구조화)
    def _build_ctx(r: dict) -> str:
        meta = r.get("meta") or {}
        name = meta.get("name") or meta.get("title") or "장소"
        summary = meta.get("summary_one_sentence") or meta.get("description") or ""
        addr = meta.get("address") or meta.get("location", {}).get("addr1") or ""
        kw = ", ".join(map(str, meta.get("keywords", [])))
        
        rating = f"평점 {meta['starts']}" if meta.get("starts") else ""
        reviews = f"리뷰 {meta['counts']}" if meta.get("counts") else ""
        
        ctx_parts = [f"[장소명]: {name}", f"[설명]: {summary}"]
        if addr: ctx_parts.append(f"[주소]: {addr}")
        if kw: ctx_parts.append(f"[키워드]: {kw}")
        if rating or reviews: ctx_parts.append(f"[정보]: {rating} {reviews}".strip())
        
        return " | ".join(ctx_parts)

    context_str = "\n".join([f"- {_build_ctx(r)}" for r in retrievals])

    # 4. 출력 형식 및 기본 지침 정의
    resolved_name = state.get("resolved_name") if bool(state.get("anchor")) else "서울"
    
    FORMAT_INSTRUCTION = (
        "답변은 반드시 아래 형식을 지켜라:\n"
        "번호. [장소 이름]\n"
        "   - 특징: 한 줄 요약\n"
        "   - 추천 이유: 상세 설명\n"
    )

    system_base = (
        "너는 서울 전문 여행 가이드다. 다음 지침을 엄격히 준수하라.\n"
        f"1. 답변 서두에 기준 위치인 '{resolved_name}'를 언급하며 인사를 건넨다.\n"
        "2. 제공된 '후보 목록'에 있는 정보만 사용하며, 없는 장소는 절대 지어내지 않는다.\n"
        f"3. {FORMAT_INSTRUCTION}"
    )

    # 5. 유동적인 제약 조건 추가 (정제된 쿼리 기준)
    constraints = []
    if is_date_query(target_query):
        constraints.append("- 데이트 의도에 맞춰 로맨틱하고 분위기 좋은 점을 강조하여 추천 이유를 작성할 것.")
    if state.get("mode_unknown"):
        constraints.append("- 카테고리가 불분명하므로 관광지 위주로 추천했음을 알리고, 맛집/카페 등 선호 타입을 물어볼 것.")

    full_system_prompt = system_base + "\n" + "\n".join(constraints)

    # 6. Few-shot 예시와 함께 메시지 구성
    messages = [
        ("system", full_system_prompt),
        ("system", f"후보 목록:\n{context_str}"),
        ("user", "잠실역 근처 맛집 추천해줘"),
        ("assistant", (
            f"{resolved_name} 주변에서 즐거운 시간을 보내실 수 있는 곳들을 추천해 드립니다.\n\n"
            "1. [홀리차우 잠실점]\n"
            "   - 특징: 퓨전 중국요리 맛집\n"
            "   - 추천 이유: 롯데월드와 가까워 아이들과 방문하기 좋으며, 남녀노소 즐길 메뉴가 다양합니다.\n"
            "2. [옹솥 롯데백화점잠실점]\n"
            "   - 특징: 푸짐한 전골 요리\n"
            "   - 추천 이유: 곱창전골 등 다양한 메뉴와 푸짐한 밑반찬이 제공되어 가족 식사에 적합합니다."
        )),
        ("user", target_query) # LLM에게 정제된 쿼리 전달
    ]

    parts: List[str] = []
    async for chunk in llm.astream(messages, config=config):
        content = chunk.content
        if not content:
            continue
        parts.append(content)
        yield {"token": content}

    final_text = "".join(parts)
    append_node_trace_result(raw_query, "answer", {"final": final_text})
    yield {"final": final_text}
    yield {"context": build_context(state)}