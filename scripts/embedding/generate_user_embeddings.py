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
    

def build_user_profile_text(user: dict) -> str:
    """
    user_profile.jsonl의 필드를 이용해
    임베딩용 user_profile 한 문장을 만드는 함수.
    """
    parts = []

    city = user.get("city")
    if city:
        parts.append(f"{city}에서 여행을 계획 중인 사용자입니다.")

    companion = user.get("companion_type") or []
    if companion:
        parts.append("동행 유형: " + ", ".join(companion))

    themes = user.get("preferred_themes") or []
    if themes:
        parts.append("선호 테마: " + ", ".join(themes))

    moods = user.get("preferred_moods") or []
    if moods:
        parts.append("선호 분위기: " + ", ".join(moods))

    budget = user.get("budget")
    if budget:
        parts.append(f"예산 수준: {budget}.")

    avoid = user.get("avoid") or []
    if avoid:
        parts.append("피하고 싶은 것: " + ", ".join(avoid))

    activity = user.get("activity_level")
    if activity:
        parts.append(f"활동 강도 선호: {activity}.")

    flow = user.get("schedule_flow")
    if flow:
        parts.append(f"전체 일정 흐름: {flow}를 선호합니다.")

    # 아무 것도 없을 경우 fallback
    if not parts:
        return "여행 취향 정보가 거의 없는 사용자입니다."

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
