import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# 프로젝트 루트 (AI/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDING_JSON_DIR = PROJECT_ROOT / "data" / "embedding_json"
MODEL_NAME = "BAAI/bge-m3"
ALPHA_DENSE = 0.6  # dense vs BM25 가중치

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
    if poi.get("name"):
        parts.append(f"장소명: {poi['name']}")
    if poi.get("summary_one_sentence"):
        parts.append(poi["summary_one_sentence"])
    themes = _join_list(poi.get("themes", []))
    if themes:
        parts.append(f"테마: {themes}")
    mood = _join_list(poi.get("mood", []))
    if mood:
        parts.append(f"분위기: {mood}")
    visitor_type = _join_list(poi.get("visitor_type", []))
    if visitor_type:
        parts.append(f"방문객 유형: {visitor_type}")
    best_time = _join_list(poi.get("best_time", []))
    if best_time:
        parts.append(f"추천 방문 시간: {best_time}")
    if poi.get("duration"):
        parts.append(f"체류 시간: {poi['duration']}")
    activity = poi.get("activity") or {}
    if isinstance(activity, dict) and activity.get("level") is not None:
        parts.append(f"활동 강도: {activity['level']}")
    if poi.get("indoor_outdoor"):
        parts.append(f"실내/실외: {poi['indoor_outdoor']}")
    if poi.get("photospot") is True:
        parts.append("포토스팟이 있는 장소")
    elif poi.get("photospot") is False:
        parts.append("포토스팟 위주의 장소는 아님")
    keywords = _join_list(poi.get("keywords", []))
    if keywords:
        parts.append(f"키워드: {keywords}")
    avoid_for = _join_list(poi.get("avoid_for", []))
    if avoid_for:
        parts.append(f"비추천 대상: {avoid_for}")
    if poi.get("ideal_schedule_position"):
        parts.append(f"일정 추천 위치: {poi['ideal_schedule_position']}")
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


def _load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("입력 JSON은 list 형태여야 합니다.")
    return data


def _tokenize(text: str) -> List[str]:
    return text.lower().replace("\n", " ").split()


def _get_lat_lng(meta: Dict):
    lat = meta.get("lat") or meta.get("latitude")
    lng = meta.get("lng") or meta.get("lon") or meta.get("longitude")
    if (lat is None or lng is None) and isinstance(meta.get("location"), dict):
        loc = meta["location"]
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


@lru_cache(maxsize=None)
def _load_split(mode: str):
    builder, _ = MODE_CONFIG[mode]
    emb = np.load(EMBEDDING_DIR / f"embeddings_{mode}.npy")
    keys = np.load(EMBEDDING_DIR / f"keys_{mode}.npy", allow_pickle=True)
    meta_list = _load_json(EMBEDDING_JSON_DIR / f"embedding_{mode}.json")
    id_to_meta = {int(item["place_id"]): item for item in meta_list if "place_id" in item}
    texts = [builder(m) for m in meta_list]
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    return emb, keys, id_to_meta, bm25


def _build_query_text(query: str, mode: str, history_place_ids: Optional[List[int]], id_to_meta: Dict[int, Dict]) -> str:
    history_place_ids = history_place_ids or []
    names = [id_to_meta[pid]["name"] for pid in history_place_ids if pid in id_to_meta and id_to_meta[pid].get("name")]
    if names:
        recent = ", ".join(names[:5])
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
    admin_term: Optional[str] = None,
) -> List[Dict]:
    if mode not in MODE_CONFIG:
        raise ValueError(f"지원하지 않는 mode: {mode}")
    embeddings, keys, id_to_meta, bm25 = _load_split(mode)
    id_to_idx = {int(keys[i][2]): i for i in range(len(keys))}

    # pre-filter by anchor/admin before scoring
    candidate_idxs = list(range(len(keys)))
    filtered_pids = None
    filter_applied = False
    if anchor_centers and anchor_radius_km is not None:
        filter_applied = True
        filtered_pids = []
        for pid, meta in id_to_meta.items():
            lat, lng = _get_lat_lng(meta)
            if lat is None or lng is None:
                continue
            dist = _distance_to_centers_km(lat, lng, anchor_centers)
            if dist is not None and dist <= anchor_radius_km:
                filtered_pids.append(pid)
    elif admin_term:
        filter_applied = True
        term = str(admin_term).lower()
        filtered_pids = []
        for pid, meta in id_to_meta.items():
            addr_parts = [
                str(meta.get("city") or ""),
                str(meta.get("district") or ""),
                str(meta.get("dong") or ""),
                str(meta.get("road") or ""),
                str(meta.get("address") or meta.get("location", {}).get("addr1") or ""),
            ]
            addr_blob = " ".join(addr_parts).lower()
            if term in addr_blob:
                filtered_pids.append(pid)
    if filter_applied:
        if not filtered_pids:
            return []
        candidate_idxs = [id_to_idx[pid] for pid in filtered_pids if pid in id_to_idx]
        if not candidate_idxs:
            return []
    model = _load_model()

    # 쿼리 텍스트를 history 정보로 강화
    qtext = _build_query_text(query, mode, history_place_ids, id_to_meta)
    qvec = model.encode([qtext], normalize_embeddings=True)[0]
    dense_scores = embeddings[candidate_idxs] @ qvec
    dense_norm = (dense_scores + 1.0) / 2.0  # [-1,1] -> [0,1]

    bm25_scores = None
    bm25_norm = np.zeros_like(dense_norm)
    if bm25 is not None:
        bm25_scores = bm25.get_scores(_tokenize(qtext))
        max_bm25 = bm25_scores.max() if bm25_scores is not None else 0.0
        if max_bm25 > 0:
            bm25_norm = bm25_scores / max_bm25
        bm25_norm = bm25_norm[candidate_idxs]

    scores = ALPHA_DENSE * dense_norm + (1 - ALPHA_DENSE) * bm25_norm
    idxs_all = scores.argsort()[::-1]

    # 키워드 기반 필터 (수족관 등 특정 도메인 키워드가 있을 때만)
    use_filter, terms = _needs_keyword_filter(qtext, mode)
    filtered_idxs = []
    if use_filter:
        for i in idxs_all:
            pid = int(keys[candidate_idxs[i]][2])
            meta = id_to_meta.get(pid, {})
            if _has_any_keyword(meta, terms):
                filtered_idxs.append(i)
            if len(filtered_idxs) >= top_k:
                break
        if not filtered_idxs:
            return []
    # 필터 결과가 없으면 전체 점수 순으로 fallback
    if filtered_idxs:
        idxs = filtered_idxs[:top_k]
    else:
        idxs = idxs_all[:top_k]

    results: List[Dict] = []
    for i in idxs:
        base_i = candidate_idxs[i]
        pid = int(keys[base_i][2])
        meta = id_to_meta.get(pid, {})
        results.append(
            {
                "place_id": pid,
                "category": mode,
                "score": float(scores[i]),
                **(
                    {
                        "score_dense": float(dense_scores[i]),
                        "score_bm25": float(bm25_scores[base_i]) if bm25_scores is not None else None,
                    }
                    if debug
                    else {}
                ),
                "meta": meta,
            }
        )
    return results
