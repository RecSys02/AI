# -*- coding: utf-8 -*-
"""
POI analysis 데이터를 기반으로
- 임베딩 텍스트 생성
- poi_embeddings_tourspot.npy
- poi_keys_tourspot.npy  (region, category, place_id)
를 생성하는 스크립트
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# =========================
# Config
# =========================
INPUT_JSONL = "../../data/embedding_json/embedding_tourspot.json"

OUT_DIR = Path("../../data/embeddings")
EMB_PATH = OUT_DIR / "poi_embeddings_tourspot.npy"
KEYS_PATH = OUT_DIR / "poi_keys_tourspot.npy"

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16


# =========================
# Utils
# =========================
def join_list(values: List[Any]) -> str:
    if not values:
        return ""
    return ", ".join([str(v).strip() for v in values if str(v).strip()])


def build_embedding_text(poi: Dict[str, Any]) -> str:
    """
    POI 하나를 임베딩용 텍스트로 변환
    (의미 중심 필드만 사용)
    """
    parts = []

    # 1. 장소명
    if poi.get("name"):
        parts.append(f"장소명: {poi['name']}")

    # 2. 한 줄 요약
    if poi.get("summary_one_sentence"):
        parts.append(poi["summary_one_sentence"])

    # 3. 테마
    themes = join_list(poi.get("themes", []))
    if themes:
        parts.append(f"테마: {themes}")

    # 4. 분위기
    mood = join_list(poi.get("mood", []))
    if mood:
        parts.append(f"분위기: {mood}")

    # 5. 방문자 유형
    visitor_type = join_list(poi.get("visitor_type", []))
    if visitor_type:
        parts.append(f"방문객 유형: {visitor_type}")

    # 6. 추천 방문 시간
    best_time = join_list(poi.get("best_time", []))
    if best_time:
        parts.append(f"추천 방문 시간: {best_time}")

    # 7. 체류 시간
    if poi.get("duration"):
        parts.append(f"체류 시간: {poi['duration']}")

    # 8. 활동 강도
    activity = poi.get("activity") or {}
    if isinstance(activity, dict) and activity.get("level") is not None:
        parts.append(f"활동 강도: {activity['level']}")

    # 9. 실내/실외
    if poi.get("indoor_outdoor"):
        parts.append(f"실내/실외: {poi['indoor_outdoor']}")

    # 10. 포토스팟
    if poi.get("photospot") is True:
        parts.append("포토스팟이 있는 장소")
    elif poi.get("photospot") is False:
        parts.append("포토스팟 위주의 장소는 아님")

    # 11. 키워드
    keywords = join_list(poi.get("keywords", []))
    if keywords:
        parts.append(f"키워드: {keywords}")

    # 12. 비추천 대상
    avoid_for = join_list(poi.get("avoid_for", []))
    if avoid_for:
        parts.append(f"비추천 대상: {avoid_for}")

    # 13. 일정 배치 추천
    if poi.get("ideal_schedule_position"):
        parts.append(f"일정 추천 위치: {poi['ideal_schedule_position']}")

    # fallback
    if not parts:
        pid = poi.get("poi_id", "unknown")
        return f"관광지 {pid}"

    return " ".join(parts)


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("입력 JSON은 list 형태여야 합니다.")

    return data



# =========================
# Main
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pois = load_json(INPUT_JSONL)
    print(f"[INFO] POI count: {len(pois)}")

    # 1. 임베딩 텍스트 생성
    texts = [build_embedding_text(p) for p in pois]

    # 2. 임베딩 ↔ DB 매핑 키 (region, category, place_id)
    poi_keys = np.array(
        [
            (p["region"], p["category"], int(p["place_id"]))
            for p in pois
        ],
        dtype=object
    )

    # 3. device 선택
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[INFO] device={device}")

    # 4. 임베딩 생성
    model = SentenceTransformer(MODEL_NAME, device=device)
    if device == "mps":
        model = model.half()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    # 5. 저장
    np.save(EMB_PATH, embeddings)
    np.save(KEYS_PATH, poi_keys)

    print(f"✅ Saved embeddings: {EMB_PATH} shape={embeddings.shape}")
    print(f"✅ Saved poi_keys: {KEYS_PATH} shape={poi_keys.shape}")


if __name__ == "__main__":
    main()
