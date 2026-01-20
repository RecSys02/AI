from typing import List

from services.chat_nodes.intent import is_date_query, is_nearby_query
from services.chat_nodes.llm_clients import llm
from services.chat_nodes.state import GraphState, build_context
from utils.geo import append_node_trace_result


async def answer_node(state: GraphState):
    if state.get("expand_failed"):
        final_text = "이전에 사용한 기준 위치가 없어서 범위를 넓힐 수 없어요. 기준 장소를 알려주세요."
        append_node_trace_result(state.get("query", ""), "answer", {"final": final_text})
        yield {"final": final_text}
        yield {"context": build_context(state)}
        return
    query = state.get("query", "")
    if is_nearby_query(query) and not (state.get("anchor") or state.get("admin_term")):
        place = state.get("resolved_name") or state.get("input_place")
        if not place:
            place_info = state.get("place") or {}
            place = place_info.get("point") or place_info.get("area")
        if place:
            final_text = f"'{place}'가 어느 지점을 말하는지 알려주세요. 기준 위치를 알려주시면 그 근처로 추천할게요."
        else:
            final_text = "근처/주변 추천을 하려면 기준 위치가 필요해요. 지점/역/건물명을 알려주세요."
        append_node_trace_result(state.get("query", ""), "answer", {"final": final_text})
        yield {"final": final_text}
        yield {"context": build_context(state)}
        return
    retrievals = state.get("retrievals", [])
    print("answer node : ", retrievals)
    if not retrievals:
        filter_applied = bool(state.get("anchor") or state.get("admin_term"))
        if filter_applied:
            resolved_name = state.get("resolved_name")
            input_place = state.get("input_place")
            location_name = resolved_name or input_place or "해당 지역"
            mode_used = state.get("mode") or "tourspot"
            category_label = {
                "tourspot": "놀거리",
                "restaurant": "맛집",
                "cafe": "카페",
            }.get(mode_used, "추천")
            yield {
                "final": (
                    f"{location_name} 근처에는 조건에 맞는 결과가 없어요. "
                    f"반경을 넓혀서 다시 찾아볼까요, 아니면 {location_name}의 다른 {category_label}로 추천해드릴까요?"
                )
            }
        else:
            yield {"final": "검색된 결과가 없습니다."}
        yield {"context": build_context(state)}
        return
    # retrieval 디버그 정보 포함해서 뭔지 확인하고 싶을때!
    if state.get("debug"):
        yield {"debug": retrievals}

    # 후보 컨텍스트: 이름 + 요약/설명 + 주소 + 키워드까지 포함
    def _build_ctx(r: dict) -> str:
        meta = r.get("meta") or {}
        name = meta.get("name") or meta.get("title") or "장소"
        summary = meta.get("summary_one_sentence") or meta.get("description") or meta.get("content") or ""
        addr = meta.get("address") or meta.get("location", {}).get("addr1") or ""
        kw = meta.get("keywords") or []
        kw_str = ", ".join([str(k) for k in kw]) if kw else ""
        popularity = []
        if meta.get("views") is not None:
            popularity.append(f"조회수 {meta['views']}")
        if meta.get("likes") is not None:
            popularity.append(f"좋아요 {meta['likes']}")
        if meta.get("bookmarks") is not None:
            popularity.append(f"북마크 {meta['bookmarks']}")
        rating_parts = []
        if meta.get("starts"):
            rating_parts.append(f"평점 {meta['starts']}")
        if meta.get("counts"):
            rating_parts.append(f"리뷰 {meta['counts']}")
        rating_str = ", ".join(rating_parts)
        pop_str = ", ".join(popularity)
        parts = [f"{name}", summary]
        if addr:
            parts.append(f"주소: {addr}")
        if kw_str:
            parts.append(f"키워드: {kw_str}")
        if pop_str:
            parts.append(f"인기: {pop_str}")
        if rating_str:
            parts.append(rating_str)
        return " ".join([p for p in parts if p])

    context = "\n".join([f"- {_build_ctx(r)}" for r in retrievals])
    messages = [
        (
            "system",
            "너는 서울 여행 가이드다. 아래 후보 목록에서만 선택해 한 줄 요약과 함께 추천해라. "
            "후보에 없는 장소는 절대 언급하지 말고, 후보 정보(이름/설명/키워드)를 활용해 답하라.",
        ),
        ("system", f"후보:\n{context}"),
    ]
    filter_applied = bool(state.get("anchor") or state.get("admin_term"))
    resolved_name = state.get("resolved_name") if filter_applied else None
    if resolved_name:
        messages.append(
            (
                "system",
                f"보정된 기준 지명은 '{resolved_name}'이다. 답변 서두에 이 지명만 언급하고, 이 기준으로 추천한다고 알려라.",
            )
        )
    if state.get("mode_unknown"):
        messages.append(
            (
                "system",
                "모드를 정확히 인식하지 못했다면 관광지 기준으로 임시 추천했으니, "
                "사용자가 카페/식당/관광지 중 원하는 카테고리를 답하도록 유도하라.",
            )
        )
    if is_date_query(query):
        messages.append(
            (
                "system",
                "사용자 의도는 데이트/커플이다. 분위기, 로맨틱함, 기념일/특별한 경험, 조용함 등을 우선 고려해 추천하라.",
            )
        )
    messages.append(("user", state.get("query", "")))

    parts: List[str] = []
    async for chunk in llm.astream(messages):
        content = chunk.content
        if not content:
            continue
        parts.append(content)
        yield {"token": content}

    final_text = "".join(parts)
    append_node_trace_result(state.get("query", ""), "answer", {"final": final_text})
    yield {"final": final_text}
    yield {"context": build_context(state)}
