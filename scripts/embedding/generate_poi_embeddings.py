# generate_poi_embeddings.py
# -*- coding: utf-8 -*-

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# ===== 설정 =====
INPUT_PATH = "./data/poi_analysis_ai_shopping.jsonl"  
OUTPUT_PATH = "./data/poi_embeddings_shopping.jsonl"              # 임베딩 결과 저장 파일
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16


def build_poi_profile_text(poi: dict) -> str:
    """
    poi_analysis_ai_attraction.jsonl 의 필드를 이용해서
    임베딩용 설명 텍스트를 한 덩어리로 만드는 함수.
    (한국어 문장으로 만들어서 bge-m3에 넣기)
    """
    parts = []

    # 한 줄 요약이 있으면 제일 앞에
    summary = poi.get("summary_one_sentence")
    if summary:
        parts.append(summary)

    # 테마
    themes = poi.get("themes") or []
    if themes:
        parts.append("주요 테마: " + ", ".join(themes))

    # 분위기
    moods = poi.get("mood") or []
    if moods:
        parts.append("분위기: " + ", ".join(moods))

    # 방문자 타입
    vtypes = poi.get("visitor_type") or []
    if vtypes:
        parts.append("주 방문객 유형: " + ", ".join(vtypes))

    # 추천 방문 시간대
    best_time = poi.get("best_time") or []
    if best_time:
        parts.append("방문 추천 시간대: " + ", ".join(best_time))

    # 체류 시간
    duration = poi.get("duration")
    if duration:
        parts.append(f"평균 체류 시간: {duration}")

    # 활동 강도
    activity = poi.get("activity_level")
    if activity:
        parts.append(f"활동 강도: {activity}")

    # 실내/실외
    indoor_outdoor = poi.get("indoor_outdoor")
    if indoor_outdoor:
        parts.append(f"실내/실외: {indoor_outdoor}")

    # 포토스팟 여부
    photospot = poi.get("photospot")
    if photospot is True:
        parts.append("포토스팟이 있어 사진 찍기 좋은 장소입니다.")
    elif photospot is False:
        parts.append("특별한 포토스팟이 중심은 아닌 장소입니다.")

    # 키워드
    keywords = poi.get("keywords") or []
    if keywords:
        parts.append("키워드: " + ", ".join(keywords))

    # 피하면 좋을 타입
    avoid_for = poi.get("avoid_for") or []
    if avoid_for:
        parts.append("다음과 같은 여행자에게는 비추천: " + ", ".join(avoid_for))

    # 일정에서 어디에 넣기 좋은지
    ideal_pos = poi.get("ideal_schedule_position")
    if ideal_pos:
        parts.append(f"일정 배치 추천: {ideal_pos}")

    # 아무것도 없을 경우 fallback
    if not parts:
        poi_id = poi.get("id")
        return f"ID {poi_id}인 관광지에 대한 정보가 거의 없습니다."

    # 한 문장 덩어리로 합치기
    return " ".join(parts)


def load_pois(path: str):
    pois = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pois.append(json.loads(line))
    return pois


def main():
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)

    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    pois = load_pois(str(input_path))
    print(f"[INFO] POI 수: {len(pois)}")

    # ----- 디바이스 선택 (CUDA / MPS / CPU) -----
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[INFO] 모델 로드 중: {MODEL_NAME} (device={device})")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # (선택) MPS일 때 half()로 조금 더 가볍게
    if device == "mps":
        model = model.half()

    # ----- POI 설명 텍스트 생성 -----
    poi_texts = []
    for p in pois:
        text = build_poi_profile_text(p)
        poi_texts.append(text)

    print("[INFO] 임베딩 계산 시작")

    # ----- 임베딩 계산 -----
    embeddings = model.encode(
        poi_texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,   # 코사인 유사도 바로 사용 가능
        show_progress_bar=True,
    )

    embeddings = embeddings.astype("float32")

    # ----- 결과 저장 (jsonl) -----
    with open(output_path, "w", encoding="utf-8") as fout:
        for poi, text, vec in zip(pois, poi_texts, embeddings):
            out_obj = {
                "poi_id": poi.get("id"),
                "profile_text": text,
                "embedding_model": MODEL_NAME,
                "embedding": vec.tolist(),
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"[INFO] 저장 완료 → {output_path.resolve()}")


if __name__ == "__main__":
    main()
