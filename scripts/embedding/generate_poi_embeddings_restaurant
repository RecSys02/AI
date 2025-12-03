# generate_poi_embeddings.py
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# ===== 설정 =====
INPUT_PATH = "cafe_data_with_ids.jsonl"
OUTPUT_PATH = "poi_embeddings_cafe_filtered.jsonl"
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16

# [필터링 기준] 이 기준 미달이면 아예 임베딩 생성 안 함 (데이터 제외)
MIN_REVIEW_COUNT = 5       # 최소 리뷰 수 (예: 5개 미만은 폐업/비인기라 가정하고 제외)
MIN_DESC_LENGTH = 10       # 설명글 최소 길이 (너무 짧으면 매칭 품질 저하)

def parse_count_str(count_str: str) -> int:
    """
    "(38)" 또는 "38" 형태의 문자열에서 숫자만 추출
    """
    if not count_str:
        return 0
    # 숫자 외의 문자 제거 (괄호, 콤마 등)
    clean_str = re.sub(r'[^0-9]', '', str(count_str))
    if not clean_str:
        return 0
    return int(clean_str)

def get_popularity_context(reviews: int, views: int, likes: int) -> str:
    """
    숫자 데이터를 임베딩 모델이 이해할 수 있는 '문장'으로 변환
    """
    parts = []
    
    # 리뷰 수에 따른 표현
    if reviews >= 100:
        parts.append("많은 방문자 리뷰가 증명하는 검증된 맛집입니다.")
    elif reviews >= 50:
        parts.append("방문자들의 리뷰가 꽤 쌓인 인기 장소입니다.")
    
    # 조회수에 따른 표현
    if views >= 10000:
        parts.append("사람들의 관심도가 매우 높고 조회수가 많은 핫플레이스입니다.")
    elif views >= 3000:
        parts.append("사람들이 많이 검색해보는 관심 장소입니다.")

    # 좋아요 수에 따른 표현
    if likes >= 50:
        parts.append("많은 사람들이 좋아요를 누른 선호도 높은 곳입니다.")

    return " ".join(parts)

def build_poi_profile_text(poi: dict) -> str:
    """
    데이터를 분석하여 임베딩용 텍스트 생성.
    조건 미달 시 None을 반환하여 호출 측에서 건너뛰게 함.
    """
    
    # 1. 숫자 데이터 파싱
    raw_counts = poi.get("counts", "0")
    reviews = parse_count_str(raw_counts)
    views = int(poi.get("views", 0))
    likes = int(poi.get("likes", 0))
    
    description = poi.get("description", "").strip()
    
    # ===== [중요] 필터링 로직 =====
    # 리뷰 수가 기준 미달이거나, 설명이 너무 짧으면 생성 포기 (None 반환)
    if reviews < MIN_REVIEW_COUNT:
        return None
    if len(description) < MIN_DESC_LENGTH:
        return None

    # 2. 텍스트 조립
    parts = []

    # 제목과 카테고리 (기본 정보)
    title = poi.get("title", "")
    category = poi.get("content", "") # restaurant 등
    parts.append(f"{title}은(는) {category}입니다.")

    # 설명글 (가장 중요)
    parts.append(description)

    # 인기도 문맥 주입 (숫자 -> 텍스트 변환)
    pop_context = get_popularity_context(reviews, views, likes)
    if pop_context:
        parts.append(pop_context)

    # 키워드
    keywords = poi.get("keywords", "")
    if keywords:
        # 키워드가 콤마로 구분된 문자열로 들어오는 경우 처리
        parts.append(f"특징 및 분위기: {keywords}")

    # 주소 (지역 정보)
    address = poi.get("address", "")
    if address:
        parts.append(f"위치: {address}")

    return " ".join(parts)


def load_pois(path: str):
    pois = []
    with open(path, "r", encoding="utf-8") as f:
        # 파일 전체가 JSON 배열([...]) 형태인지, JSONL(줄바꿈)인지 확인 필요
        # 제공해주신 포맷은 JSON 배열 내의 객체들 같으므로, 
        # 만약 JSONL이라면 아래 로직 유지, JSON List라면 json.load(f) 사용해야 함.
        # 여기서는 기존 코드(JSONL 가정)를 따르되, 예외처리 추가
        for line in f:
            line = line.strip()
            if not line: continue
            
            # 쉼표로 끝나는 라인 처리 (JSON array 복사붙여넣기 시 흔한 오류 방지)
            if line.endswith(","): 
                line = line[:-1]
                
            try:
                pois.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pois


def main():
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)

    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    # 데이터 로드 (JSONL 형식 가정, 만약 통짜 JSON이면 수정 필요)
    # 제공해주신 샘플이 리스트 형태라 json.load가 맞을 수 있으나 
    # 기존 코드 흐름 상 JSONL 처럼 처리합니다.
    try:
        raw_pois = load_pois(str(input_path))
    except:
        # 혹시 통짜 JSON 파일일 경우 대비
        with open(input_path, "r", encoding='utf-8') as f:
            raw_pois = json.load(f)

    print(f"[INFO] 원본 데이터 수: {len(raw_pois)}")

    # ----- 텍스트 생성 및 필터링 -----
    valid_pois = []
    valid_texts = []

    for p in raw_pois:
        text = build_poi_profile_text(p)
        if text is not None:  # None이 아니면(필터 통과하면) 저장
            valid_pois.append(p)
            valid_texts.append(text)
    
    print(f"[INFO] 필터링 후 임베딩 대상 수: {len(valid_pois)} (제외됨: {len(raw_pois) - len(valid_pois)})")

    if not valid_pois:
        print("[WARN] 유효한 데이터가 없습니다. 종료합니다.")
        return

    # ----- 모델 로드 -----
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[INFO] 모델 로드 중: {MODEL_NAME} (device={device})")
    model = SentenceTransformer(MODEL_NAME, device=device)
    if device == "mps":
        model = model.half()

    print("[INFO] 임베딩 계산 시작")

    # ----- 임베딩 계산 -----
    embeddings = model.encode(
        valid_texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = embeddings.astype("float32")

    # ----- 결과 저장 -----
    with open(output_path, "w", encoding="utf-8") as fout:
        for poi, text, vec in zip(valid_pois, valid_texts, embeddings):
            # 원본 데이터의 키 보존 및 메타데이터 추가
            out_obj = poi.copy() 
            out_obj["profile_text_for_embedding"] = text
            out_obj["embedding"] = vec.tolist()
            
            # 필요하다면 리뷰 수 등 숫자도 정제해서 저장 (후처리를 위해)
            out_obj["parsed_reviews"] = parse_count_str(poi.get("counts"))
            
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"[INFO] 저장 완료 → {output_path.resolve()}")


if __name__ == "__main__":
    main()
