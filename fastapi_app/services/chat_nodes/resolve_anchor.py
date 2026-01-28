from typing import Dict, List

from services.chat_nodes.state import GraphState
from utils.geo import (
    append_node_trace_result,
    append_place_debug,
    build_alias_map,
    load_admin_aliases,
    load_anchor_cache,
    load_geo_centers,
    load_keyword_aliases,
    normalize_text,
    resolve_alias,
    save_anchor_cache,
)
from utils.google_geocode import geocode_place_id
from utils.google_place_autocomplete import autocomplete_places

SEOUL_PREFIXES = ("서울특별시", "서울시", "서울")
REGION_HINTS = (
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
AREA_SUFFIXES = ("특별시", "광역시", "자치시", "자치구", "시", "군", "구", "도")
STATION_TYPE_HINTS = {"subway_station", "transit_station", "train_station", "light_rail_station"}
INTERSECTION_HINTS = ("교차로", "사거리")
PRIORITY1 = {
    "subway_station",
    "transit_station",
    "train_station",
    "bus_station",
    "light_rail_station",
    "intersection",
}
PRIORITY2_EXACT = {"political", "locality"}
PRIORITY2_PREFIXES = ("sublocality", "administrative_area_level")
PRIORITY3 = {
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


def _strip_prefix(token: str, prefixes: tuple[str, ...]) -> str | None:
    for prefix in prefixes:
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _extract_area_hint(text_norm: str) -> str | None:
    if not text_norm:
        return None
    for suffix in AREA_SUFFIXES:
        idx = text_norm.find(suffix)
        if idx > 0:
            candidate = text_norm[: idx + len(suffix)]
            if len(candidate) > len(suffix):
                return candidate
    return None


def _build_match_tokens(raw_place: str, raw_area: str, raw_point: str) -> List[str]:
    tokens: List[str] = []
    for part in (raw_place, raw_area, raw_point):
        norm = normalize_text(part or "")
        if norm and norm not in tokens:
            tokens.append(norm)
    extras: List[str] = []
    for token in list(tokens):
        stripped = _strip_prefix(token, SEOUL_PREFIXES)
        if stripped:
            extras.append(stripped)
        if token.endswith("역") and len(token) > 1:
            extras.append(token[:-1])
        if stripped and stripped.endswith("역") and len(stripped) > 1:
            extras.append(stripped[:-1])
    for extra in extras:
        if extra and extra not in tokens:
            tokens.append(extra)
    return tokens


def _build_region_tokens(raw_area: str | None, raw_place: str) -> List[str]:
    tokens: List[str] = []
    area_norm = normalize_text(raw_area or "")
    if area_norm:
        tokens.append(area_norm)
        if area_norm.endswith("시") and len(area_norm) > 1:
            tokens.append(area_norm[:-1])
        if area_norm.endswith("도") and len(area_norm) > 1:
            tokens.append(area_norm[:-1])
    if not tokens:
        q_norm = normalize_text(raw_place or "")
        for region in REGION_HINTS:
            if region in q_norm:
                tokens.append(region)
                break
    if not tokens:
        inferred = _extract_area_hint(normalize_text(raw_place or ""))
        if inferred:
            tokens.append(inferred)
            if inferred.endswith("시") and len(inferred) > 1:
                tokens.append(inferred[:-1])
            if inferred.endswith("도") and len(inferred) > 1:
                tokens.append(inferred[:-1])
    if not tokens:
        tokens.append("서울")
    uniq: List[str] = []
    for token in tokens:
        if token and token not in uniq:
            uniq.append(token)
    return uniq


def _build_station_variants(match_tokens: List[str]) -> List[str]:
    variants: List[str] = []
    for token in match_tokens:
        if token.endswith("역"):
            variants.append(token)
            if len(token) > 1:
                variants.append(token[:-1])
    uniq: List[str] = []
    for token in variants:
        if token and token not in uniq:
            uniq.append(token)
    return uniq


def _match_any(text_norm: str, tokens: List[str]) -> bool:
    return any(token and token in text_norm for token in tokens)


def _is_priority2(types: set[str]) -> bool:
    if types & PRIORITY2_EXACT:
        return True
    return any(t.startswith(PRIORITY2_PREFIXES) for t in types)


def _is_priority3(types: set[str]) -> bool:
    return bool(types & PRIORITY3)


def _score_candidate(
    item: dict,
    match_tokens: List[str],
    region_tokens: List[str],
    station_tokens: List[str],
    station_hint: bool,
    intersection_hint: bool,
) -> int:
    desc_norm = normalize_text(item.get("description") or "")
    types = set(item.get("types") or [])
    tokens = station_tokens if station_hint and station_tokens else match_tokens
    match_score = 0
    for token in tokens:
        if not token:
            continue
        if token == desc_norm:
            match_score = max(match_score, 6)
        elif token in desc_norm:
            match_score = max(match_score, 4)
        elif desc_norm and desc_norm in token:
            match_score = max(match_score, 2)
    score = match_score
    if station_hint and "역" in desc_norm:
        score += 1
    if station_hint and (types & STATION_TYPE_HINTS):
        score += 2
    if station_hint and "역점" in desc_norm:
        score -= 2
    if station_hint and "출구" in desc_norm:
        score += 1
    if station_hint:
        penalize_types = {
            "store",
            "restaurant",
            "food",
            "cafe",
            "bar",
            "bakery",
            "doctor",
            "health",
            "lodging",
            "meal_takeaway",
            "meal_delivery",
            "local_government_office",
            "city_hall",
        }
        if types & penalize_types:
            score -= 2
    if region_tokens and _match_any(desc_norm, region_tokens):
        score += 1
    if "intersection" in types and not intersection_hint:
        score -= 1
    return score


def _select_anchor_candidates(candidates: List[dict], normalized_query: str) -> List[dict]:
    if not candidates:
        return []
    p1 = [c for c in candidates if set(c.get("types") or []) & PRIORITY1]
    if p1:
        return p1
    p2 = [c for c in candidates if _is_priority2(set(c.get("types") or []))]
    if p2:
        return p2
    p3 = []
    for c in candidates:
        types = set(c.get("types") or [])
        if not _is_priority3(types):
            continue
        desc = normalize_text(c.get("description") or "")
        if normalized_query and normalized_query in desc:
            p3.append(c)
    return p3 if p3 else candidates


async def resolve_anchor_node(state: GraphState) -> Dict:
    """Resolve a textual place into a geographic anchor or admin term."""
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

    # Keyword path: alias -> geo_centers -> cache -> autocomplete -> geocode
    admin_suffixes = ("시", "구", "동", "가", "로", "길", "대로")
    keyword_aliases = load_keyword_aliases()
    alias_map = build_alias_map(keyword_aliases)
    if raw_place.endswith(admin_suffixes):
        admin_aliases = load_admin_aliases()
        alias_map.update(build_alias_map(admin_aliases))
    canonical = resolve_alias(raw_place, alias_map)
    # Build matching tokens to score autocomplete candidates.
    match_tokens = _build_match_tokens(raw_place, raw_area, raw_point)
    canonical_norm = normalize_text(canonical)
    if canonical_norm and canonical_norm not in match_tokens:
        match_tokens.append(canonical_norm)
    stripped = _strip_prefix(canonical_norm, SEOUL_PREFIXES)
    if stripped and stripped not in match_tokens:
        match_tokens.append(stripped)
    if canonical_norm.endswith("역") and len(canonical_norm) > 1:
        no_station = canonical_norm[:-1]
        if no_station not in match_tokens:
            match_tokens.append(no_station)
    if stripped and stripped.endswith("역") and len(stripped) > 1:
        no_station = stripped[:-1]
        if no_station not in match_tokens:
            match_tokens.append(no_station)
    region_tokens = _build_region_tokens(raw_area or None, canonical or raw_place)
    station_tokens = [token for token in match_tokens if token.endswith("역")]
    station_variants = _build_station_variants(match_tokens)

    # Exact configured centers take precedence over live lookups.
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

    # Cache lookup guards against stale/wrong anchors by token validation.
    cache = load_anchor_cache()
    cache_key = normalize_text(canonical)
    cached = cache.get(cache_key)
    if cached and cached.get("lat") is not None and cached.get("lng") is not None:
        cache_blob = " ".join(
            [
                str(cached.get("resolved_name") or ""),
                str(cached.get("address") or ""),
                str(cached.get("query") or ""),
            ]
        )
        cache_norm = normalize_text(cache_blob)
        cache_ok = True
        if station_tokens and not _match_any(cache_norm, station_tokens):
            cache_ok = False
        if region_tokens and not _match_any(cache_norm, region_tokens):
            cache_ok = False
        if match_tokens and not _match_any(cache_norm, match_tokens):
            cache_ok = False
        if cache_ok:
            anchor = {
                "centers": [[cached["lat"], cached["lng"]]],
                "radius_by_intent": cached.get("radius_by_intent") or {},
                "source": "anchor_cache",
            }
            resolved_name = cached.get("resolved_name") or canonical
            if station_tokens:
                resolved_name = canonical
            result = {
                "anchor": anchor,
                "place": {"place": canonical},
                "input_place": raw_place,
                "resolved_name": resolved_name,
            }
            append_node_trace_result(state.get("query", ""), "resolve_anchor", result)
            return result

    # Autocomplete is used only when cache/geo_centers do not resolve.
    candidates = autocomplete_places(canonical, limit=5, types=None)
    if not candidates:
        candidates = autocomplete_places(canonical, limit=5, types="(regions)")
    selected = None
    food_types = {"restaurant", "cafe", "bar", "bakery", "food", "meal_takeaway", "meal_delivery"}
    non_food_candidates = []
    anchor_candidates = []
    normalized_query = normalize_text(canonical)

    for item in candidates:
        types = set([t for t in item.get("types") or []])
        if types & food_types:
            continue
        non_food_candidates.append(item)
        if (types & PRIORITY1) or _is_priority2(types) or _is_priority3(types):
            anchor_candidates.append(item)

    base_candidates = _select_anchor_candidates(anchor_candidates or non_food_candidates, normalized_query)
    station_hint = False
    if station_tokens:
        station_filtered = []
        for item in base_candidates:
            desc_norm = normalize_text(item.get("description") or "")
            if _match_any(desc_norm, station_tokens):
                station_filtered.append(item)
        if station_filtered:
            base_candidates = station_filtered
            station_hint = True
    intersection_hint = any(hint in (state.get("query") or "") for hint in INTERSECTION_HINTS)
    scored = []
    for idx, item in enumerate(base_candidates):
        # Score candidates by token match quality and type hints.
        score = _score_candidate(
            item,
            match_tokens=match_tokens,
            region_tokens=region_tokens,
            station_tokens=station_tokens,
            station_hint=station_hint,
            intersection_hint=intersection_hint,
        )
        scored.append((score, idx, item))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected_pool = [item for _, _, item in scored]

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
                # Hard-bound to 서울권 to avoid far-off anchors.
                if 37.4 <= lat <= 37.7 and 126.7 <= lng <= 127.2:
                    selected = {
                        "geo": geo,
                        "place_id": place_id,
                        "title": item.get("description") or "",
                        "description": item.get("description") or "",
                        "types": item.get("types") or [],
                    }
                    break

    append_place_debug(
        {
            "query": state.get("query", ""),
            "input_area": raw_area or None,
            "input_point": raw_point or None,
            "normalized": canonical,
            "match_tokens": match_tokens,
            "region_tokens": region_tokens,
            "station_tokens": station_tokens,
            "station_variants": station_variants,
            "autocomplete_candidates": candidates,
            "anchor_candidates": anchor_candidates,
            "geocode_attempts": geocode_attempts,
            "selected": selected,
        }
    )

    if selected:
        # Save resolved anchor for faster future lookups.
        lat = selected["geo"].get("lat")
        lng = selected["geo"].get("lng")
        default_radius_by_intent = {"restaurant": 2.0, "cafe": 2.0, "tourspot": 3.0}
        cache_resolved_name = selected.get("title") or ""
        if station_tokens:
            cache_resolved_name = canonical
        cache[cache_key] = {
            "lat": lat,
            "lng": lng,
            "address": selected["geo"].get("address") or "",
            "query": canonical,
            "resolved_name": cache_resolved_name,
            "place_id": selected.get("place_id") or "",
            "radius_by_intent": default_radius_by_intent,
            "source": "google_autocomplete+geocode",
        }
        save_anchor_cache(cache)
        anchor = {
            "centers": [[lat, lng]],
            "radius_by_intent": default_radius_by_intent,
            "source": "google_autocomplete+geocode",
        }
        resolved_name = selected.get("title") or canonical
        if station_tokens:
            resolved_name = canonical
        result = {
            "anchor": anchor,
            "place": {"place": canonical},
            "input_place": raw_place,
            "resolved_name": resolved_name,
        }
        append_node_trace_result(state.get("query", ""), "resolve_anchor", result)
        return result

    result = {
        "place": {"place": canonical},
        "input_place": raw_place,
        "resolved_name": canonical,
        "anchor_failed": True,
    }
    append_node_trace_result(state.get("query", ""), "resolve_anchor", result)
    return result
