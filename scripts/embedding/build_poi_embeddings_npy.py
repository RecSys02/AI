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

MODEL_NAME = "dragonkue/multilingual-e5-small-ko"
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

    # 1. 이름 + 요약
    header = []
    name = poi.get("name")
    summary = poi.get("summary_one_sentence")
    if name:
        header.append(f"이름: {name}")
    if name or summary:
        header.append("카테고리: 관광지")
    if summary:
        header.append(summary)
    if header:
        parts.append(". ".join(header))

    # 2. 테마/분위기/실내외
    themes = join_list(poi.get("themes", []))
    mood = join_list(poi.get("mood", []))
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

    # 3. 대상/키워드
    visitor_type = join_list(poi.get("visitor_type", []))
    keywords = join_list(poi.get("keywords", []))
    line = []
    if visitor_type:
        line.append(f"대상: {visitor_type}")
    if keywords:
        line.append(f"키워드: {keywords}")
    if line:
        parts.append(". ".join(line))

    # 4. 활동/시간/일정
    activity = poi.get("activity") or {}
    activity_label = None
    if isinstance(activity, dict):
        activity_label = activity.get("label")
    if not activity_label and isinstance(activity, dict) and activity.get("level") is not None:
        activity_label = activity.get("level")
    best_time = join_list(poi.get("best_time", []))
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

    # 5. 포토스팟 (있는 경우만)
    if poi.get("photospot") is True:
        parts.append("포토스팟: 있음")

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
    # texts = [builder(p) for p in pois]
    texts = [f"passage: {builder(p)}" for p in pois]
    
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
