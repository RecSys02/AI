# exp_recommend_top10_poi.py
# -*- coding: utf-8 -*-

"""
User–POI 임베딩 기반 추천 테스트 스크립트.

- user_embeddings.jsonl의 각 사용자 임베딩을 불러와
- POI 임베딩 리스트와 코사인 유사도를 계산하고
- 사용자별 Top 10 POI ID 목록을 생성하여 user_top10_poi.jsonl로 저장

💡 모델 개발/평가를 위한 실험용 추천 코드이며,
   운영 추천 로직/랭킹 모델/스케줄링 로직은 포함되지 않음.
"""


import json
import numpy as np
from pathlib import Path

USER_EMB_PATH = "./data/user_embeddings.jsonl"
POI_FILES = [
    "./data/poi_embeddings_attraction.jsonl",
    "./data/poi_embeddings_shopping.jsonl"
]
OUTPUT_PATH = "./data/user_top10_poi.jsonl"


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def cosine_similarity(a, b):
    """a: (d,) b: (n,d) → 반환: (n,)"""
    return np.dot(b, a) / (np.linalg.norm(b, axis=1) * np.linalg.norm(a) + 1e-8)


def main():
    # ------ Load user ------
    users = load_jsonl(USER_EMB_PATH)

    # ------ Load POI from multiple files ------
    pois = []
    for file in POI_FILES:
        pois += load_jsonl(file)

    print(f"[INFO] users: {len(users)} / pois(total): {len(pois)}")

    # ------ 벡터 행렬화 ------
    poi_ids = [p["poi_id"] for p in pois]
    poi_matrix = np.array([p["embedding"] for p in pois], dtype=np.float32)

    results = []

    for user in users:
        user_id = user["user_id"]
        user_vec = np.array(user["embedding"], dtype=np.float32)

        # similarity 계산
        scores = cosine_similarity(user_vec, poi_matrix)

        # Top 10 인덱스
        top_idx = np.argsort(scores)[::-1][:10]

        top10_ids = [poi_ids[i] for i in top_idx]

        results.append({
            "user_id": user_id,
            "top10_poi": top10_ids
        })

    # 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[INFO] 저장 완료 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
