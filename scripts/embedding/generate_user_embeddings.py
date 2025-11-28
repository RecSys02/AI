# generate_user_embeddings.py
# -*- coding: utf-8 -*-

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# ===== 설정 =====
INPUT_PATH = "./data/user/user_profile.jsonl"            
OUTPUT_PATH = "./data/user_embeddings.jsonl"         # 임베딩 결과 저장 파일
MODEL_NAME = "BAAI/bge-m3"            
BATCH_SIZE = 16


def build_poi_profile_text(poi: dict) -> str:
    """
    raw + AI 태그가 합쳐진 POI 딕셔너리를 받아
    BGE 임베딩용 한 덩어리 설명 텍스트로 변환한다.
    """

    parts = []

    # 1) 기본 메타 정보
    name = poi.get("name")
    gu_name = poi.get("gu_name")
    if name and gu_name:
        parts.append(f"{gu_name}에 위치한 '{name}' 장소입니다.")
    elif name:
        parts.append(f"'{name}' 장소입니다.")

    # 카테고리 문자열
    category = poi.get("category") or {}
    cat1_name = category.get("cat1_name")
    cat2_name = category.get("cat2_name")
    cat3_name = category.get("cat3_name")

    cat_parts = [c for c in [cat1_name, cat2_name, cat3_name] if c]
    if cat_parts:
        parts.append("카테고리: " + " > ".join(cat_parts))

    # 2) AI 한 줄 요약
    summary = poi.get("summary_one_sentence")
    if summary:
        parts.append(summary)

    # 3) AI 태그 기반 프로필
    themes = poi.get("themes") or []
    if themes:
        parts.append("주요 테마: " + ", ".join(themes))

    moods = poi.get("mood") or []
    if moods:
        parts.append("분위기: " + ", ".join(moods))

    vtypes = poi.get("visitor_type") or []
    if vtypes:
        parts.append("주 방문객 유형: " + ", ".join(vtypes))

    best_time = poi.get("best_time") or []
    if best_time:
        parts.append("방문 추천 시간대: " + ", ".join(best_time))

    duration = poi.get("duration")
    if duration:
        parts.append(f"평균 체류 시간: {duration}")

    # activity: dict / string 둘 다 대응
    activity = poi.get("activity") or poi.get("activity_level")
    if isinstance(activity, dict):
        label = activity.get("label")
        level = activity.get("level")
        if label and level:
            parts.append(f"활동 강도: {label} (레벨 {level})")
        elif label:
            parts.append(f"활동 강도: {label}")
    elif isinstance(activity, str):
        parts.append(f"활동 강도: {activity}")

    indoor_outdoor = poi.get("indoor_outdoor")
    if indoor_outdoor:
        parts.append(f"실내/실외: {indoor_outdoor}")

    photospot = poi.get("photospot")
    if photospot is True:
        parts.append("포토스팟이 있어 사진 찍기 좋은 장소입니다.")
    elif photospot is False:
        parts.append("특별한 포토스팟이 중심은 아닌 장소입니다.")

    keywords = poi.get("keywords") or []
    if keywords:
        parts.append("키워드: " + ", ".join(keywords))

    avoid_for = poi.get("avoid_for") or []
    if avoid_for:
        parts.append("다음과 같은 여행자에게는 비추천: " + ", ".join(avoid_for))

    ideal_pos = poi.get("ideal_schedule_position")
    if ideal_pos:
        parts.append(f"일정 배치 추천: {ideal_pos}")

    # 4) raw overview 일부도 살짝 포함 (선택)
    overview = poi.get("overview")
    if overview:
        # 너무 길까 걱정되면 앞부분만 잘라써도 됨
        short_ov = overview[:400]
        parts.append("장소 설명: " + short_ov)

    # 아무것도 없을 경우 fallback
    if not parts:
        poi_id = poi.get("poi_id") or poi.get("id")
        return f"ID {poi_id}인 관광지에 대한 정보가 거의 없습니다."

    return " ".join(parts)


def load_users(path: str):
    users = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            users.append(json.loads(line))
    return users


def main():
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)

    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    users = load_users(str(input_path))
    print(f"[INFO] 유저 수: {len(users)}")

    # ----- BGE 모델 로드 -----
    # device 선택
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[INFO] 모델 로드 중: {MODEL_NAME} (device={device})")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # MPS 최적화
    if device == "mps":
        model = model.half()


    # ----- 프로필 텍스트 생성 -----
    profile_texts = []
    for u in users:
        text = build_user_profile_text(u)
        profile_texts.append(text)

    print("[INFO] 임베딩 계산 시작")

    # ----- 임베딩 계산 -----
    embeddings = model.encode(
        profile_texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,   # 나중에 코사인 유사도 바로 사용 가능
        show_progress_bar=True,
    )

    # float32로 다운캐스팅 (용량 절약용)
    embeddings = embeddings.astype("float32")

    # ----- 결과 저장 (jsonl) -----
    with open(output_path, "w", encoding="utf-8") as fout:
        for user, text, vec in zip(users, profile_texts, embeddings):
            out_obj = {
                "user_id": user.get("user_id"),
                "city": user.get("city"),
                "profile_text": text,
                "embedding_model": MODEL_NAME,
                "embedding": vec.tolist(),  # JSON으로 저장하기 위해 list로 변환
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"[INFO] 저장 완료 → {output_path.resolve()}")


if __name__ == "__main__":
    main()
