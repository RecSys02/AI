# -*- coding: utf-8 -*-
"""
POI 임베딩 생성 스크립트
- tourspot(기존 로직) / food(카페/레스토랑) 모드 분리
- 입력 JSON에서 임베딩 텍스트 생성 후 numpy 배열 저장
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Callable, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# 기본 경로
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOUR_PATH = ROOT / "data" / "embedding_json" / "embedding_tourspot.json"
DEFAULT_RESTAURANT_PATH = ROOT / "data" / "embedding_json" / "embedding_restaurant.json"
DEFAULT_CAFE_PATH = ROOT / "data" / "embedding_json" / "embedding_cafe.json"
OUT_DIR = ROOT / "data" / "embeddings"

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16


# =========================
# Utils
# =========================
def join_list(values: List[Any]) -> str:
    if not values:
        return ""
    return ", ".join([str(v).strip() for v in values if str(v).strip()])


def build_embedding_text_tourspot(poi: Dict[str, Any]) -> str:
    """
    관광지용 임베딩 텍스트
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

    if not parts:
        pid = poi.get("poi_id", "unknown")
        return f"관광지 {pid}"

    return " ".join(parts)


def build_embedding_text_food(poi: Dict[str, Any]) -> str:
    """
    카페/레스토랑용 임베딩 텍스트
    """
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
    kws = join_list(poi.get("keywords", []))
    if kws:
        parts.append(f"키워드: {kws}")

    return " ".join(parts) or (title or "음식/카페")


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("입력 JSON은 list 형태여야 합니다.")
    return data


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Build POI embeddings (tourspot / cafe / restaurant).")
    parser.add_argument("--mode", choices=["tourspot", "cafe", "restaurant"], default="tourspot")
    parser.add_argument("--input", type=Path, help="입력 JSON 경로 (기본: mode별 기본값)")
    parser.add_argument(
        "--output-prefix",
        help="출력 파일 접두사 (poi_embeddings_<prefix>.npy / poi_keys_<prefix>.npy)",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # 기본 prefix는 파일명에 바로 붙습니다. (예: embeddings_<prefix>.npy)
    mode_config: Dict[str, Tuple[Path, str, Callable[[Dict[str, Any]], str]]] = {
        "tourspot": (DEFAULT_TOUR_PATH, "tourspot", build_embedding_text_tourspot),
        "restaurant": (DEFAULT_RESTAURANT_PATH, "restaurant", build_embedding_text_food),
        "cafe": (DEFAULT_CAFE_PATH, "cafe", build_embedding_text_food),
    }

    default_input, default_prefix, builder = mode_config[args.mode]
    input_path = args.input or default_input
    prefix = args.output_prefix or default_prefix

    emb_path = OUT_DIR / f"embeddings_{prefix}.npy"
    keys_path = OUT_DIR / f"keys_{prefix}.npy"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pois = load_json(input_path)
    print(f"[INFO] mode={args.mode} input={input_path} count={len(pois)}")

    # 임베딩 텍스트
    texts = [builder(p) for p in pois]

    # 키 (province, category, place_id)
    poi_keys = np.array(
        [(p.get("province"), p.get("category"), int(p.get("place_id"))) for p in pois],
        dtype=object,
    )

    device = select_device()
    print(f"[INFO] device={device}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    if device == "mps":
        model = model.half()

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    np.save(emb_path, embeddings)
    np.save(keys_path, poi_keys)

    print(f"✅ Saved embeddings: {emb_path} shape={embeddings.shape}")
    print(f"✅ Saved poi_keys: {keys_path} shape={poi_keys.shape}")


if __name__ == "__main__":
    main()
