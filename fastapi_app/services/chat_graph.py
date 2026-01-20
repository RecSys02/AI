import os
import json
from typing import Dict, List, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from services.retriever import retrieve
from utils.geo import (
    add_alias,
    build_alias_map,
    distance_to_centers_km,
    get_lat_lng,
    load_admin_aliases,
    load_anchor_cache,
    load_geo_centers,
    load_keyword_aliases,
    normalize_text,
    resolve_alias,
    save_admin_aliases,
    save_anchor_cache,
    append_place_debug,
    append_node_trace_result,
)
from utils.google_geocode import geocode_place_id
from utils.google_place_autocomplete import autocomplete_places

class GraphState(TypedDict):
    query: str
    normalized_query: str | None
    mode: str | None
    mode_detected: str | None
    mode_unknown: bool | None
    place: Dict | None
    anchor: Dict | None
    admin_term: str | None
    input_place: str | None
    resolved_name: str | None
    top_k: int | None
    history_place_ids: List[int]
    intent: str
    retrievals: List[dict]
    final: str
    messages: List[dict]
    debug: bool | None
    
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
DETECT_MODEL = os.getenv("DETECT_MODEL", CHAT_MODEL)
llm = ChatOpenAI(model=CHAT_MODEL, streaming=True, temperature=0.0)
# 모드 감지/리랭크용은 스트리밍 없이 호출
detect_llm = ChatOpenAI(model=DETECT_MODEL, streaming=False, temperature=0.0)


def _detect_intent(query: str) -> str:
    q = query.lower()
    # 추천 의도가 포함된 경우 "recommend" 반환
    recommend_terms = ["추천", "어디", "가볼", "뭐가 있어", "top", "best", "3개", "5곳"]

    return "recommend" if any(t in q for t in recommend_terms) else "general"

def _detect_mode(mode_hint: str | None, query: str) -> str:
    # 직접적으로 모드가 주어지면 우선 사용
    if mode_hint in {"tourspot", "cafe", "restaurant"}:
        return mode_hint
    # 힌트가 없으면 쿼리 기반으로 추론
    q = query.lower()
    cafe_terms = ["카페", "커피", "디저트", "브런치", "빵", "라떼", "tea", "티룸"]
    restaurant_terms = [
        "맛집",
        "식당",
        "레스토랑",
        "밥",
        "점심",
        "저녁",
        "고기",
        "파스타",
        "스테이크",
        "스시",
        "초밥",
        "회",
        "중식",
        "한식",
        "양식",
        "분식",
        "라멘",
        "라면",
        "피자",
        "버거",
        "삼겹살",
        "술집",
        "포차",
        "안주",
        "뷔페",
    ]
    # 기본값은 unknown, 카페/맛집 키워드가 있으면 해당 모드로 변경
    if any(t in q for t in cafe_terms):
        return "cafe"
    if any(t in q for t in restaurant_terms):
        return "restaurant"
    return "unknown"


async def _llm_detect_mode(query: str) -> str:
    """LLM으로 모드 분류 (tourspot/cafe/restaurant/unknown 중 하나만 반환)."""
    messages = [
        (
            "system",
            "다음 사용자 질문이 관광지(tourspot), 카페(cafe), 식당/맛집(restaurant) 중 어느 카테고리에 해당하는지 "
            "정확히 하나의 단어만 소문자로 답하라. 해당이 없으면 unknown만 답하라.",
        ),
        ("user", query),
    ]
    try:
        resp = await detect_llm.ainvoke(messages, max_tokens=5)
        mode = (resp.content or "").strip().lower()
        return mode if mode in {"tourspot", "cafe", "restaurant"} else "unknown"
    except Exception:
        return "unknown"

async def _llm_extract_place(query: str) -> Dict | None:
    """LLM으로 장소 후보 키워드를 최대한 너그럽게 추출한다."""
    messages = [
            (
                "system",
                "너는 사용자의 의도에서 '지리적 위치(지명)'만 추출하는 전문가야.\n"
                "다음 JSON 형식으로만 답하라: {\"area\": \"...\", \"point\": \"...\"}\n"
                "규칙:\n"
                "1. '음식 메뉴(김밥, 파스타, 떡볶이 등)'나 '장소의 종류(맛집, 카페, 놀거리)'는 절대 지명으로 추출하지 마라.\n"
                "2. area는 행정구역/지역명(예: 강남구, 신림동, 여의도), point는 구체 지점(역/대학교/아파트/빌딩/랜드마크/몰)로 분리하라.\n"
                "3. 둘 다 있으면 area와 point 모두 채워라. point가 없다면 point는 null로 두어라.\n"
                "4. 오타가 있더라도 문맥상 '지역/지점'이면 추출하되(예: 걍남 -> 걍남), 메뉴 이름은 무조건 배제하라.\n"
                "5. 지명이 없으면 반드시 {\"area\": null, \"point\": null}을 반환하라."
            ),
            ("user", f"입력 문장: {query}\n추출 결과: "),
    ]
    try:
        resp = await detect_llm.ainvoke(messages, max_tokens=40)
        raw = (resp.content or "").strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        area = data.get("area")
        point = data.get("point")
        area_val = area.strip() if isinstance(area, str) and area.strip() else None
        point_val = point.strip() if isinstance(point, str) and point.strip() else None
        if not area_val and not point_val:
            return None
        return {"area": area_val, "point": point_val}
    except Exception:
        return None

async def route_node(state: GraphState) -> Dict:
    intent = _detect_intent(state.get("query", ""))
    result = {"intent": intent}
    append_node_trace_result(state.get("query", ""), "route", result)
    return result # 변경된 부분만 반환

async def normalize_query_node(state: GraphState) -> Dict:
    query = state.get("query", "")
    messages = [
            (
                "system",
                "너는 검색 엔진을 위한 쿼리 최적화 전문가야. 사용자 질문을 아래 규칙에 따라 정규화하라.\n\n"
                "1. **구체적 지명 보존**: '반포 자이', '삼성의원', '강남역 1번 출구'와 같은 구체적인 건물, 아파트명, 지점 정보는 절대 생략하거나 광역 지명(예: 강남)으로 축소하지 마라.\n"
                "2. **의도 명확화**: '놀만한 거'는 '놀거리/명소'로, '맛있는 곳'은 '맛집/식당'으로 검색에 유리한 단어로 치환하라.\n"
                "3. **오타 수정**: '걍남' -> '강남', '강나' -> '강남' 등 명확한 오타는 수정하되, 고유 명사인지 확인하라.\n"
                "4. **메뉴 강조**: '짜장면', '방어' 같은 구체적 메뉴가 있다면 이를 문장의 핵심으로 유지하라.\n"
                "JSON 형식만 반환: {\"normalized_query\": \"...\"}"
            ),
            ("user", query),
    ]
    normalized = query
    try:
        resp = await detect_llm.ainvoke(messages, max_tokens=80)
        raw = (resp.content or "").strip()
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("normalized_query"):
            normalized = str(data["normalized_query"]).strip()
    except Exception:
        pass
    result = {"normalized_query": normalized}
    append_node_trace_result(query, "normalize_query", result)
    return result

def _slim_retrievals(items: List[dict]) -> List[dict]:
    slim = []
    for r in items:
        meta = r.get("meta") or {}
        slim.append(
            {
                "place_id": r.get("place_id"),
                "category": r.get("category"),
                "province": meta.get("province"),
                "name": meta.get("name") or meta.get("title"),
                "score": r.get("score"),
            }
        )
    return slim

async def extract_place_node(state: GraphState) -> Dict:
    query = state.get("normalized_query") or state.get("query", "")
    place = await _llm_extract_place(query)
    append_place_debug(
        {
            "query": query,
            "place": place,
        }
    )
    result = {"place": place}
    append_node_trace_result(query, "extract_place", result)
    return result

async def resolve_anchor_node(state: GraphState) -> Dict:
    place = state.get("place") or {}
    raw_point = (place.get("point") or "").strip()
    raw_area = (place.get("area") or "").strip()
    raw_place = raw_point or raw_area
    if raw_area and raw_point and raw_area not in raw_point:
        raw_place = f"{raw_area} {raw_point}"
    if not raw_place:
        result = {}
        append_node_trace_result(state.get("query", ""), "resolve_anchor", result)
        return result

    # Admin 판단은 suffix 규칙으로만
    lowered = raw_place.lower()
    admin_suffixes = ("시", "구", "동", "가", "로", "길", "대로")
    if not raw_point and lowered.endswith(admin_suffixes):
        admin_aliases = load_admin_aliases()
        alias_map = build_alias_map(admin_aliases)
        canonical = resolve_alias(raw_place, alias_map)
        if canonical != raw_place:
            if add_alias(admin_aliases, canonical, raw_place):
                save_admin_aliases(admin_aliases)
        result = {
            "admin_term": canonical,
            "place": {"place": canonical},
            "input_place": raw_place,
            "resolved_name": canonical,
        }
        append_node_trace_result(state.get("query", ""), "resolve_anchor", result)
        return result

    # Keyword 경로: alias -> geo_centers -> cache -> local search -> geocode
    keyword_aliases = load_keyword_aliases()
    alias_map = build_alias_map(keyword_aliases)
    canonical = resolve_alias(raw_place, alias_map)

    geo_centers = load_geo_centers()
    if canonical in geo_centers:
        entry = geo_centers[canonical] or {}
        centers = entry.get("centers") or []
        if centers:
            anchor = {
                "centers": centers,
                "radius_by_intent": entry.get("radius_by_intent") or {},
                "source": "geo_centers",
            }
            result = {
                "anchor": anchor,
                "place": {"place": canonical},
                "input_place": raw_place,
                "resolved_name": canonical,
            }
            append_node_trace_result(state.get("query", ""), "resolve_anchor", result)
            return result

    cache = load_anchor_cache()
    cache_key = normalize_text(canonical)
    cached = cache.get(cache_key)
    if cached and cached.get("lat") is not None and cached.get("lng") is not None:
        anchor = {
            "centers": [[cached["lat"], cached["lng"]]],
            "radius_by_intent": cached.get("radius_by_intent") or {},
            "source": "anchor_cache",
        }
        result = {
            "anchor": anchor,
            "place": {"place": canonical},
            "input_place": raw_place,
            "resolved_name": cached.get("resolved_name") or canonical,
        }
        append_node_trace_result(state.get("query", ""), "resolve_anchor", result)
        return result

    candidates = autocomplete_places(canonical, limit=5)
    selected = None
    food_types = {"restaurant", "cafe", "bar", "bakery", "food", "meal_takeaway", "meal_delivery"}
    priority1 = {"subway_station", "transit_station", "intersection"}
    priority2 = {"political", "sublocality", "locality"}
    priority3 = {
        "university",
        "school",
        "park",
        "tourist_attraction",
        "museum",
        "library",
        "shopping_mall",
        "stadium",
        "lodging",
        "point_of_interest",
        "establishment",
    }
    normalized_query = normalize_text(canonical)
    non_food_candidates = []
    anchor_candidates = []
    for item in candidates:
        types = set([t for t in item.get("types") or []])
        if types & food_types:
            continue
        non_food_candidates.append(item)
        if types & (priority1 | priority2 | priority3):
            anchor_candidates.append(item)

    def _select_by_priority(items: List[dict]) -> List[dict]:
        if not items:
            return []
        p1 = [c for c in items if set(c.get("types") or []) & priority1]
        if p1:
            return p1
        p2 = [c for c in items if set(c.get("types") or []) & priority2]
        if p2:
            return p2
        p3 = []
        for c in items:
            types = set(c.get("types") or [])
            if not (types & priority3):
                continue
            desc = normalize_text(c.get("description") or "")
            if normalized_query and normalized_query in desc:
                p3.append(c)
        return p3 if p3 else items

    selected_pool = _select_by_priority(anchor_candidates or non_food_candidates)
    geocode_attempts = []
    for item in selected_pool:
        place_id = item.get("place_id") or ""
        if not place_id:
            continue
        geo = geocode_place_id(place_id)
        geocode_attempts.append(
            {
                "place_id": place_id,
                "geocode": geo,
            }
        )
        if geo:
            lat = geo.get("lat")
            lng = geo.get("lng")
            if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
                if 37.4 <= lat <= 37.7 and 126.7 <= lng <= 127.2:
                    selected = {
                        "geo": geo,
                        "place_id": place_id,
                        "title": item.get("description") or "",
                    }
                    break

    append_place_debug(
        {
            "query": state.get("query", ""),
            "input_area": raw_area or None,
            "input_point": raw_point or None,
            "normalized": canonical,
            "autocomplete_candidates": candidates,
            "anchor_candidates": anchor_candidates,
            "geocode_attempts": geocode_attempts,
            "selected": selected,
        }
    )

    if selected:
        lat = selected["geo"].get("lat")
        lng = selected["geo"].get("lng")
        cache[cache_key] = {
            "lat": lat,
            "lng": lng,
            "address": selected["geo"].get("address") or "",
            "query": canonical,
            "resolved_name": selected.get("title") or "",
            "place_id": selected.get("place_id") or "",
            "radius_by_intent": {},
            "source": "google_autocomplete+geocode",
        }
        save_anchor_cache(cache)
        anchor = {
            "centers": [[lat, lng]],
            "radius_by_intent": {},
            "source": "google_autocomplete+geocode",
        }
        result = {
            "anchor": anchor,
            "place": {"place": canonical},
            "input_place": raw_place,
            "resolved_name": selected.get("title") or canonical,
        }
        append_node_trace_result(state.get("query", ""), "resolve_anchor", result)
        return result

    result = {"place": {"place": canonical}, "input_place": raw_place, "resolved_name": canonical}
    append_node_trace_result(state.get("query", ""), "resolve_anchor", result)
    return result

async def retrieve_node(state: GraphState) -> Dict:
    query = state.get("normalized_query") or state.get("query", "")
    mode_raw = _detect_mode(state.get("mode"), query)
    if mode_raw == "unknown":
        mode_raw = await _llm_detect_mode(query)
    mode_unknown = mode_raw == "unknown"
    mode_used = "tourspot" if mode_unknown else mode_raw
    requested_k = state.get("top_k")
    # 이전 방문 기록 아직 없음. 어떻게 뽑아올지 정해야함(기존 추천시스템에서 방문한 이력들 DB에서 뽑아오도록 해야할듯?)
    history_ids = state.get("history_place_ids") or []

    debug_flag = bool(state.get("debug"))
    # 후보를 retrieve 함수를 사용해 뽑아옴
    anchor = state.get("anchor") or {}
    admin_term = state.get("admin_term")
    centers = anchor.get("centers") or []
    radius_by_intent = anchor.get("radius_by_intent") or {}
    radius_km = float(radius_by_intent.get(mode_used, 2.0)) if centers else None
    hits = retrieve(
        query=query,
        mode=mode_used,
        top_k=max(requested_k, 20) if requested_k is not None else 20,
        history_place_ids=history_ids,
        debug=debug_flag,
        anchor_centers=centers or None,
        anchor_radius_km=radius_km,
        admin_term=admin_term,
    )
    result = {
        "retrievals": hits,
        "mode": mode_used,
        "mode_detected": mode_raw,
        "mode_unknown": mode_unknown,
    }
    append_node_trace_result(
        state.get("query", ""),
        "retrieve",
        {**result, "retrievals": _slim_retrievals(result["retrievals"])},
    )
    return result

async def apply_location_filter_node(state: GraphState) -> Dict:
    retrievals = state.get("retrievals") or []
    if not retrievals:
        result = {}
        append_node_trace_result(state.get("query", ""), "apply_location_filter", result)
        return result

    anchor = state.get("anchor") or {}
    admin_term = state.get("admin_term")
    mode_used = state.get("mode") or "tourspot"

    if anchor:
        centers = anchor.get("centers") or []
        radius_by_intent = anchor.get("radius_by_intent") or {}
        radius_km = float(radius_by_intent.get(mode_used, 1.5))
        filtered = []
        for r in retrievals:
            meta = r.get("meta") or {}
            lat, lng = get_lat_lng(meta)
            if lat is None or lng is None:
                continue
            dist_km = distance_to_centers_km(lat, lng, centers)
            if dist_km is not None and dist_km <= radius_km:
                filtered.append(r)
        result = {"retrievals": filtered}
        append_node_trace_result(
            state.get("query", ""),
            "apply_location_filter",
            {"retrievals": _slim_retrievals(result["retrievals"])},
        )
        return result

    if admin_term:
        term = str(admin_term).lower()
        filtered = []
        for r in retrievals:
            meta = r.get("meta") or {}
            addr_parts = [
                str(meta.get("city") or ""),
                str(meta.get("district") or ""),
                str(meta.get("dong") or ""),
                str(meta.get("road") or ""),
                str(meta.get("address") or meta.get("location", {}).get("addr1") or ""),
            ]
            addr_blob = " ".join(addr_parts).lower()
            if term in addr_blob:
                filtered.append(r)
        result = {"retrievals": filtered}
        append_node_trace_result(
            state.get("query", ""),
            "apply_location_filter",
            {"retrievals": _slim_retrievals(result["retrievals"])},
        )
        return result

    result = {}
    append_node_trace_result(state.get("query", ""), "apply_location_filter", result)
    return result

async def answer_node(state: GraphState):
    retrievals = state.get("retrievals", [])
    print("answer node : ",retrievals)
    if not retrievals:
        yield {"final": "검색된 결과가 없습니다."}
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
        ("system", f"추천 개수: {state['top_k']}\n후보:\n{context}") if state.get("top_k") else ("system", f"후보:\n{context}"),
    ]
    resolved_name = state.get("resolved_name")
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


async def rerank_node(state: GraphState) -> Dict:
    """LLM으로 상위 후보를 재선택한다."""
    retrievals = state.get("retrievals") or []
    if not retrievals:
        result = {"retrievals": []}
        append_node_trace_result(state.get("query", ""), "rerank", result)
        return result

    desired_k = state.get("top_k") or 5
    desired_k = max(1, min(desired_k, len(retrievals)))

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
        parts = [name, summary]
        if addr:
            parts.append(f"주소: {addr}")
        if kw_str:
            parts.append(f"키워드: {kw_str}")
        if pop_str:
            parts.append(f"인기: {pop_str}")
        if rating_str:
            parts.append(rating_str)
        return " ".join([p for p in parts if p])

    candidates_txt = "\n".join(
        [f"[id={i}] {_build_ctx(r)}" for i, r in enumerate(retrievals)]
    )
    messages = [
        (
            "system",
            f"사용자 질문과 아래 후보를 보고 가장 관련 높은 상위 {desired_k}개를 고르고, "
            "id만 JSON 배열로 반환해. 예: [0,2]. 다른 텍스트는 넣지 말 것.",
        ),
        ("system", f"후보:\n{candidates_txt}"),
        ("user", state.get("query", "")),
    ]

    raw = None
    try:
        resp = await detect_llm.ainvoke(messages, max_tokens=50)
        raw = resp.content or ""
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            idxs = []
            for v in parsed:
                try:
                    iv = int(v)
                except Exception:
                    continue
                if 0 <= iv < len(retrievals) and iv not in idxs:
                    idxs.append(iv)
            if idxs:
                selected = [retrievals[i] for i in idxs[:desired_k]]
                result = {"retrievals": selected}
                append_node_trace_result(
                    state.get("query", ""),
                    "rerank",
                    {
                        "raw": raw,
                        "idxs": idxs,
                        "retrievals": _slim_retrievals(result["retrievals"]),
                    },
                )
                return result
    except Exception:
        pass

    # 파싱 실패 시 fallback: 상위 desired_k 사용
    selected = retrievals[:desired_k]
    result = {"retrievals": selected}
    append_node_trace_result(
        state.get("query", ""),
        "rerank",
        {
            "raw": raw,
            "idxs": None,
            "retrievals": _slim_retrievals(result["retrievals"]),
        },
    )
    return result


async def general_answer_node(state: GraphState):
    query = state.get("query", "")
    mode_raw = _detect_mode(state.get("mode"), query)
    if mode_raw == "unknown":
        mode_raw = await _llm_detect_mode(query)
    mode_unknown = mode_raw == "unknown"
    mode_used = "tourspot" if mode_unknown else mode_raw
    history_place_ids: List[int] = state.get("history_place_ids") or []
    if state.get("debug"):
        # general에서도 같은 형태로 디버그 반환
        pass

    # 일반 질의도 데이터 기반으로 답하도록 간단히 검색 사용
    req_k = state.get("top_k")
    hits = retrieve(
        query=query,
        mode=mode_used,
        top_k=max(req_k, 5) if req_k is not None else 5,
        history_place_ids=history_place_ids,
    )
    if not hits:
        yield {"final": "관련 정보를 찾지 못했습니다."}
        return
    if state.get("debug"):
        yield {"debug": hits}

    def _build_ctx(r: dict) -> str:
        meta = r.get("meta") or {}
        name = meta.get("name") or meta.get("title") or "장소"
        desc = meta.get("summary_one_sentence") or meta.get("description") or meta.get("content") or ""
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
        parts = [name, desc]
        if addr:
            parts.append(f"주소: {addr}")
        if kw_str:
            parts.append(f"키워드: {kw_str}")
        if pop_str:
            parts.append(f"인기: {pop_str}")
        if rating_str:
            parts.append(rating_str)
        return " ".join([p for p in parts if p])

    context = "\n".join([f"- {_build_ctx(r)}" for r in hits])

    messages = [
        (
            "system",
            "너는 서울 여행/맛집/카페 정보를 안내하는 챗봇이다. "
            "반드시 아래 후보 정보만 활용해서 질문에 답해라. 후보 밖 내용은 말하지 마라.",
        ),
        ("system", f"후보 정보:\n{context}"),
    ]
    if mode_unknown:
        messages.append(
            (
                "system",
                "모드를 정확히 인식하지 못했다면 관광지 기준으로 임시로 답하고,"
                "사용자에게 식당/카페/관광지 중 원하는 카테고리를 물어본다.",
            )
        )
    messages.append(("user", query))

    parts: List[str] = []
    async for chunk in llm.astream(messages):
        content = chunk.content
        if not content:
            continue
        parts.append(content)
        yield {"token": content}

    final_text = "".join(parts)
    append_node_trace_result(state.get("query", ""), "general_answer", {"final": final_text})
    yield {"final": final_text}

# 그래프 구성
workflow = StateGraph(GraphState)
workflow.add_node("route", route_node)
workflow.add_node("normalize_query", normalize_query_node)
workflow.add_node("extract_place", extract_place_node)
workflow.add_node("resolve_anchor", resolve_anchor_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("apply_location_filter", apply_location_filter_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("answer", answer_node)
workflow.add_node("general_answer", general_answer_node)
workflow.set_entry_point("route")
# 조건부 엣지
workflow.add_conditional_edges(
    "route",
    lambda state: "normalize_query" if state["intent"] == "recommend" else "general_answer",
    {"normalize_query": "normalize_query", "general_answer": "general_answer"}
)

workflow.add_edge("normalize_query", "extract_place")
workflow.add_edge("extract_place", "resolve_anchor")
workflow.add_edge("resolve_anchor", "retrieve")
workflow.add_edge("retrieve", "apply_location_filter")
workflow.add_edge("apply_location_filter", "rerank")
workflow.add_edge("rerank", "answer")
workflow.add_edge("answer", END)
workflow.add_edge("general_answer", END)
chat_app = workflow.compile()
