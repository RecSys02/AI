import time
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from services.stores import get_milvus_store, get_postgres_store

MODEL_NAME = "dragonkue/multilingual-e5-small-ko"
E5_QUERY_PREFIX = "query: "
ALPHA_DENSE = 0.6  # dense vs BM25 가중치
DENSE_CANDIDATE_MULTIPLIER = 50
BM25_CANDIDATE_MULTIPLIER = 50
MAX_CANDIDATES = 1000

SYNONYMS = {
    # 수족관/해양
    "aquarium": ["수족관", "아쿠아리움", "aquarium"],
    # 일식/스시
    "sushi": ["스시", "초밥", "오마카세", "사시미", "스시야", "초밥집", "카이센동", "텐동"],
    # 이탈리안/파스타/피자
    "pasta": ["파스타", "이탈리안", "스파게티", "리조또", "알리오올리오", "까르보나라", "토마토파스타", "크림파스타"],
    "pizza": ["피자", "pizzeria", "나폴리피자", "화덕피자"],
    # 카페/디저트
    "coffee": ["카페", "커피", "디저트", "브런치", "라떼", "티룸", "tea", "커피숍", "카페거리", "로스터리", "핸드드립", "스페셜티", "브런치카페"],
    "dessert": ["디저트", "케이크", "빙수", "도넛", "마카롱", "수제 디저트", "타르트", "쿠키", "젤라또", "아이스크림", "초콜릿", "수제 아이스크림", "수제 쿠키"],
    # 버거/브런치류
    "burger": ["버거", "햄버거", "버거집"],
    # 스테이크/양식 고기
    "steak": ["스테이크", "티본", "안심", "등심", "립아이"],
    # 한식 고기류
    "korean_bbq": ["삼겹살", "꽃등심", "소갈비", "차돌박이", "한우", "고기", "육회", "돼지갈비", "소고기구이", "숯불구이"],
    "gopchang": ["곱창", "막창", "대창", "양대창", "양곱창", "곱창구이"],
    # 면류
    "noodle": ["칼국수", "국수", "냉면", "라면", "라멘", "우동", "막국수", "쌀국수", "쫄면", "메밀", "비빔국수"],
    # 치킨류
    "chicken": ["치킨", "통닭", "닭갈비", "닭도리탕", "양념치킨", "후라이드"],
    # 분식/간편식
    "bunsik": ["분식", "떡볶이", "김밥", "라볶이", "순대", "튀김"],
    # 주류/바
    "izakaya": ["이자카야", "사케", "오뎅", "타코야끼", "덴뿌라"],
    "beer": ["맥주", "브루어리", "펍", "호프", "수제맥주", "바", "비어"],
    "wine": ["와인", "와인바", "와인 바"],
    "nightlife": ["바", "포차", "술집", "칵테일", "나이트", "라운지"],
    # 중식/기타
    "chinese": ["중식", "짜장면", "짬뽕", "탕수육", "중국집", "만두", "딤섬"],
    # 한정식/한식 세트
    "korean_set": ["한정식", "한식", "궁중", "백반", "정식"],
    # 해산물/회
    "seafood": ["해물", "회", "횟집", "참치", "오징어", "조개", "조개구이", "해산물"],
    "duck": ["오리", "오리구이", "오리 로스", "오리로스", "훈제오리", "유황오리", "오리백숙", "오리탕"],
    # 프렌치/양식
    "french": ["프렌치", "프랑스", "비스토로", "비스트로"],
    # 기타 이국/퓨전
    "fusion": ["퓨전", "남미", "브라질", "스페인", "남미요리"],
    # 관광/여행지(투어스팟)
    "tourspot": [
        "관광지", "명소", "핫플", "여행지", "공원", "산책", "산책로", "둘레길", "숲길",
        "한강", "강변", "호수", "계곡", "폭포", "전망대", "야경",
        "박물관", "미술관", "전시장", "전시", "아트센터",
        "사찰", "절", "성당", "교회", "성지", "한옥", "전통마을",
        "캠핑", "캠핑장", "글램핑", "피크닉", "도보코스", "트레킹", "등산", "산",
    ],
    # 카페 세부(베이커리 등)
    "bakery": ["베이커리", "빵집", "파티세리", "크루아상", "바게트"],
}

KEYWORD_GROUPS_BY_MODE = {
    "restaurant": [
        "sushi",
        "pasta",
        "pizza",
        "burger",
        "steak",
        "korean_bbq",
        "gopchang",
        "noodle",
        "chicken",
        "bunsik",
        "izakaya",
        "beer",
        "wine",
        "nightlife",
        "chinese",
        "korean_set",
        "seafood",
        "duck",
        "french",
        "fusion",
    ],
    "cafe": ["coffee", "dessert", "bakery"],
    "tourspot": ["tourspot", "aquarium"],
}


def _join_list(values: List) -> str:
    if not values:
        return ""
    return ", ".join([str(v).strip() for v in values if str(v).strip()])


def build_embedding_text_tourspot(poi: Dict) -> str:
    parts = []

    header = []
    if poi.get("name"):
        header.append(f"{poi['name']} (카테고리: 관광지)")
    if poi.get("summary_one_sentence"):
        header.append(poi["summary_one_sentence"])
    if header:
        parts.append(". ".join(header))

    themes = _join_list(poi.get("themes", []))
    mood = _join_list(poi.get("mood", []))
    indoor_outdoor = poi.get("indoor_outdoor")
    line = []
    if themes:
        line.append(f"테마: {themes}")
    if mood:
        line.append(f"분위기: {mood}")
    if indoor_outdoor:
        line.append(f"실내/실외: {indoor_outdoor}")
    if line:
        parts.append(". ".join(line))

    visitor_type = _join_list(poi.get("visitor_type", []))
    keywords = _join_list(poi.get("keywords", []))
    line = []
    if visitor_type:
        line.append(f"대상: {visitor_type}")
    if keywords:
        line.append(f"키워드: {keywords}")
    if line:
        parts.append(". ".join(line))

    activity = poi.get("activity") or {}
    activity_label = None
    if isinstance(activity, dict):
        activity_label = activity.get("label")
    if not activity_label and isinstance(activity, dict) and activity.get("level") is not None:
        activity_label = activity.get("level")
    best_time = _join_list(poi.get("best_time", []))
    ideal_schedule = poi.get("ideal_schedule_position")
    line = []
    if activity_label:
        line.append(f"활동 강도: {activity_label}")
    if best_time:
        line.append(f"추천 시간: {best_time}")
    if ideal_schedule:
        line.append(str(ideal_schedule))
    if line:
        parts.append(". ".join(line))

    if poi.get("photospot") is True:
        parts.append("포토스팟: 있음")

    if not parts:
        pid = poi.get("poi_id", "unknown")
        return f"관광지 {pid}"

    return " ".join(parts)


def build_embedding_text_food(poi: Dict) -> str:
    parts = []
    title = poi.get("title") or poi.get("name")
    if title:
        parts.append(f"이름: {title}")
    if poi.get("category"):
        parts.append(f"카테고리: {poi['category']}")
    if poi.get("content"):
        parts.append(poi["content"])
    if poi.get("description"):
        parts.append(poi["description"])
    kws = _join_list(poi.get("keywords", []))
    if kws:
        parts.append(f"키워드: {kws}")
    return " ".join(parts) or (title or "음식/카페")

MODE_CONFIG: Dict[str, Tuple[Callable[[Dict], str], str]] = {
    "tourspot": (build_embedding_text_tourspot, "tourspot"),
    "cafe": (build_embedding_text_food, "cafe"),
    "restaurant": (build_embedding_text_food, "restaurant"),
}

def _select_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    device = _select_device()
    model = SentenceTransformer(MODEL_NAME, device=device)
    if device == "mps":
        model = model.half()
    return model


def _get_lat_lng(meta: Dict):
    lat = meta.get("lat") or meta.get("latitude")
    lng = meta.get("lng") or meta.get("lon") or meta.get("longitude")
    nested = meta.get("meta") if isinstance(meta.get("meta"), dict) else None
    if (lat is None or lng is None) and nested:
        lat = lat or nested.get("lat") or nested.get("latitude")
        lng = lng or nested.get("lng") or nested.get("lon") or nested.get("longitude")
    if (lat is None or lng is None) and isinstance(meta.get("location"), dict):
        loc = meta["location"]
        lat = loc.get("lat") or loc.get("latitude") or lat
        lng = loc.get("lng") or loc.get("lon") or loc.get("longitude") or lng
    if (lat is None or lng is None) and nested and isinstance(nested.get("location"), dict):
        loc = nested["location"]
        lat = loc.get("lat") or loc.get("latitude") or lat
        lng = loc.get("lng") or loc.get("lon") or loc.get("longitude") or lng
    try:
        return float(lat), float(lng)
    except Exception:
        return None, None


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _distance_to_centers_km(lat: float, lng: float, centers: List[List[float]]) -> Optional[float]:
    if not centers:
        return None
    lat1 = np.deg2rad(lat)
    lon1 = np.deg2rad(lng)
    min_km = None
    for c in centers:
        if not c or len(c) != 2:
            continue
        lat2 = np.deg2rad(c[0])
        lon2 = np.deg2rad(c[1])
        dist = _haversine_km(lat1, lon1, lat2, lon2)
        if min_km is None or dist < min_km:
            min_km = dist
    return min_km


def _needs_keyword_filter(query_text: str, mode: str) -> Tuple[bool, List[str]]:
    q_lower = query_text.lower()
    matched_terms: List[str] = []
    group_keys = KEYWORD_GROUPS_BY_MODE.get(mode, [])
    for key in group_keys:
        group = SYNONYMS.get(key, [])
        if any(term.lower() in q_lower for term in group):
            matched_terms.extend(group)
    return (len(matched_terms) > 0, matched_terms if matched_terms else [])


def _has_any_keyword(meta: Dict, terms: List[str]) -> bool:
    target_parts = []
    for key in ["name", "title", "summary_one_sentence", "description", "overview", "keywords"]:
        val = meta.get(key)
        if isinstance(val, list):
            target_parts.extend([str(v) for v in val])
        elif val:
            target_parts.append(str(val))
    blob = " ".join(target_parts).lower()
    return any(t.lower() in blob for t in terms)


def _build_query_text(query: str, history_names: List[str]) -> str:
    if history_names:
        recent = ", ".join(history_names[:5])
        return f"{query} (최근 방문: {recent})"
    return query


def retrieve(
    query: str,
    mode: str = "tourspot",
    top_k: int = 1,
    history_place_ids: Optional[List[int]] = None,
    debug: bool = False,
    anchor_centers: Optional[List[List[float]]] = None,
    anchor_radius_km: Optional[float] = None,
    timings: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    if mode not in MODE_CONFIG:
        raise ValueError(f"지원하지 않는 mode: {mode}")
    t_start = time.perf_counter()
    t0 = time.perf_counter()
    milvus = get_milvus_store()
    pg = get_postgres_store()
    t1 = time.perf_counter()
    if timings is not None:
        timings["load_store_ms"] = round((t1 - t0) * 1000, 2)
    t0 = time.perf_counter()
    model = _load_model()
    t1 = time.perf_counter()
    if timings is not None:
        timings["load_model_ms"] = round((t1 - t0) * 1000, 2)

    # 쿼리 텍스트를 history 정보로 강화
    history_place_ids = history_place_ids or []
    history_names = pg.fetch_names(history_place_ids, category=mode)
    qtext = _build_query_text(query, history_names)
    qtext_embed = f"{E5_QUERY_PREFIX}{qtext}"
    t0 = time.perf_counter()
    qvec = model.encode([qtext_embed], normalize_embeddings=True)[0]
    t1 = time.perf_counter()
    if timings is not None:
        timings["encode_ms"] = round((t1 - t0) * 1000, 2)
    t0 = time.perf_counter()
    dense_k = min(max(top_k * DENSE_CANDIDATE_MULTIPLIER, top_k), MAX_CANDIDATES)
    dense_hits = milvus.search(mode, qvec, top_k=dense_k)
    t1 = time.perf_counter()
    if timings is not None:
        timings["dense_search_ms"] = round((t1 - t0) * 1000, 2)
    if not dense_hits:
        if timings is not None:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return []
    dense_score_by_id = {pid: score for pid, score in dense_hits}

    t0 = time.perf_counter()
    bm25_k = min(max(top_k * BM25_CANDIDATE_MULTIPLIER, top_k), MAX_CANDIDATES)
    bm25_score_by_id = pg.fts_scores(qtext, mode, limit=bm25_k)
    t1 = time.perf_counter()
    if timings is not None:
        timings["bm25_ms"] = round((t1 - t0) * 1000, 2)

    t0 = time.perf_counter()
    candidate_ids = list(dict.fromkeys(list(dense_score_by_id.keys()) + list(bm25_score_by_id.keys())))
    if not candidate_ids:
        if timings is not None:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return []
    meta_map = pg.fetch_meta(candidate_ids, category=mode)
    distance_by_id = {}
    if anchor_centers and anchor_radius_km is not None:
        filtered_ids = []
        for pid in candidate_ids:
            row = meta_map.get(pid, {})
            lat, lng = _get_lat_lng(row or {})
            if lat is None or lng is None:
                continue
            dist = _distance_to_centers_km(lat, lng, anchor_centers)
            if dist is None:
                continue
            distance_by_id[pid] = dist
            if dist <= anchor_radius_km:
                filtered_ids.append(pid)
        candidate_ids = filtered_ids
        if not candidate_ids:
            if timings is not None:
                timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return []
    dense_scores = np.array([dense_score_by_id.get(pid, 0.0) for pid in candidate_ids], dtype=float)
    dense_norm = (dense_scores + 1.0) / 2.0
    bm25_scores = np.array([bm25_score_by_id.get(pid, 0.0) for pid in candidate_ids], dtype=float)
    max_bm25 = bm25_scores.max() if bm25_scores.size else 0.0
    bm25_norm = bm25_scores / max_bm25 if max_bm25 > 0 else np.zeros_like(bm25_scores)
    scores = ALPHA_DENSE * dense_norm + (1 - ALPHA_DENSE) * bm25_norm
    idxs_all = scores.argsort()[::-1]
    t1 = time.perf_counter()
    if timings is not None:
        timings["rank_ms"] = round((t1 - t0) * 1000, 2)

    # 키워드 기반 필터 (수족관 등 특정 도메인 키워드가 있을 때만)
    t0 = time.perf_counter()
    use_filter, terms = _needs_keyword_filter(qtext, mode)
    filtered_idxs = []
    if use_filter:
        for i in idxs_all:
            pid = int(candidate_ids[i])
            meta_row = meta_map.get(pid, {})
            meta = meta_row.get("meta") if isinstance(meta_row, dict) else {}
            if _has_any_keyword(meta, terms):
                filtered_idxs.append(i)
            if len(filtered_idxs) >= top_k:
                break
        if not filtered_idxs:
            t1 = time.perf_counter()
            if timings is not None:
                timings["keyword_filter_ms"] = round((t1 - t0) * 1000, 2)
                timings["total_ms"] = round((t1 - t_start) * 1000, 2)
            return []
    # 필터 결과가 없으면 전체 점수 순으로 fallback
    if filtered_idxs:
        idxs = filtered_idxs[:top_k]
    else:
        idxs = idxs_all[:top_k]
    t1 = time.perf_counter()
    if timings is not None:
        timings["keyword_filter_ms"] = round((t1 - t0) * 1000, 2)

    t0 = time.perf_counter()
    results: List[Dict] = []
    for i in idxs:
        pid = int(candidate_ids[i])
        meta_row = meta_map.get(pid, {})
        meta = meta_row.get("meta") if isinstance(meta_row, dict) else {}
        results.append(
            {
                "place_id": pid,
                "category": mode,
                "score": float(scores[i]),
                **(
                    {
                        "score_dense": float(dense_scores[i]),
                        "score_bm25": float(bm25_scores[i]) if bm25_scores.size else None,
                        "distance_km": float(distance_by_id.get(pid)) if distance_by_id else None,
                    }
                    if debug
                    else {}
                ),
                "meta": meta,
            }
        )
    t1 = time.perf_counter()
    if timings is not None:
        timings["build_results_ms"] = round((t1 - t0) * 1000, 2)
        timings["total_ms"] = round((t1 - t_start) * 1000, 2)
    return results
