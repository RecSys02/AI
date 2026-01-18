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


def _needs_keyword_filter(query_text: str) -> Tuple[bool, List[str]]:
    keywords = ["수족관", "아쿠아리움", "aquarium"]
    q_lower = query_text.lower()
    hit = [k for k in keywords if k in q_lower]
    return (len(hit) > 0, keywords)


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
) -> List[Dict]:
    if mode not in MODE_CONFIG:
        raise ValueError(f"지원하지 않는 mode: {mode}")
    embeddings, keys, id_to_meta, bm25 = _load_split(mode)
    model = _load_model()

    # 쿼리 텍스트를 history 정보로 강화
    qtext = _build_query_text(query, mode, history_place_ids, id_to_meta)
    qvec = model.encode([qtext], normalize_embeddings=True)[0]
    dense_scores = embeddings @ qvec
    dense_norm = (dense_scores + 1.0) / 2.0  # [-1,1] -> [0,1]

    bm25_scores = None
    bm25_norm = np.zeros_like(dense_norm)
    if bm25 is not None:
        bm25_scores = bm25.get_scores(_tokenize(qtext))
        max_bm25 = bm25_scores.max() if bm25_scores is not None else 0.0
        if max_bm25 > 0:
            bm25_norm = bm25_scores / max_bm25

    scores = ALPHA_DENSE * dense_norm + (1 - ALPHA_DENSE) * bm25_norm
    idxs_all = scores.argsort()[::-1]

    # 키워드 기반 필터 (수족관 등 특정 도메인 키워드가 있을 때만)
    use_filter, terms = _needs_keyword_filter(qtext)
    filtered_idxs = []
    if use_filter:
        for i in idxs_all:
            pid = int(keys[i][2])
            meta = id_to_meta.get(pid, {})
            if _has_any_keyword(meta, terms):
                filtered_idxs.append(i)
            if len(filtered_idxs) >= top_k:
                break
    # 필터 결과가 없으면 전체 점수 순으로 fallback
    if filtered_idxs:
        idxs = filtered_idxs[:top_k]
    else:
        idxs = idxs_all[:top_k]

    results: List[Dict] = []
    for i in idxs:
        pid = int(keys[i][2])
        meta = id_to_meta.get(pid, {})
        results.append(
            {
                "place_id": pid,
                "category": mode,
                "score": float(scores[i]),
                **(
                    {
                        "score_dense": float(dense_scores[i]),
                        "score_bm25": float(bm25_scores[i]) if bm25_scores is not None else None,
                    }
                    if debug
                    else {}
                ),
                "meta": meta,
            }
        )
    return results
