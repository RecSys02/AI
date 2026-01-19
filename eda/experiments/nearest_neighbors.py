# nearest_neighbors.py
# -*- coding: utf-8 -*-
 
"""
POI 임베딩 유사도 탐색 스크립트.

- 저장된 임베딩 JSONL 파일을 로드하고
- 코사인 유사도를 기반으로 TARGET_POI_ID와 가장 유사한 TOP-K POI를 출력
- 임베딩 값의 NaN / Inf / Zero-norm 여부를 점검하는 디버그 기능 포함
- 추천 모델 품질 점검 및 EDA 목적의 실험용 코드
"""

import json
import numpy as np

INPUT_PATH = "./data/poi_embeddings_sum.jsonl"   # 임베딩 저장된 jsonl 파일 경로
TOPK = 10
TARGET_POI_ID = "1606096" 


def load_embeddings(path):
    poi_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            emb = obj.get("embedding")
            if emb is None:
                continue

            poi_list.append({
                "poi_id": obj.get("poi_id"),
                "profile_text": obj.get("profile_text"),
                "embedding": np.array(emb, dtype=np.float32),
            })
    return poi_list


def cosine_sim_matrix(X):
    # L2 normalize
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return Xn @ Xn.T   # (N, N)

def debug_check_embeddings(pois):
    X = np.stack([p["embedding"] for p in pois], axis=0)  # (N, d)

    norms = np.linalg.norm(X, axis=1)
    print("===== [DEBUG] Embedding Stats =====")
    print("[DEBUG] total POIs       :", len(pois))
    print("[DEBUG] min norm         :", norms.min())
    print("[DEBUG] max norm         :", norms.max())
    print("[DEBUG] zero-norm count  :", np.sum(norms == 0))
    print("[DEBUG] has NaN          :", np.isnan(X).any())
    print("[DEBUG] has Inf          :", np.isinf(X).any())

    # 어떤 poi들이 문제인지까지 보고 싶으면:
    zero_idx = np.where(norms == 0)[0]
    if len(zero_idx) > 0:
        print("\n[DEBUG] Zero-norm POIs (first 20):")
        for i in zero_idx[:20]:
            print(f"  - idx={i}, poi_id={pois[i]['poi_id']}")

    bad_idx = np.unique(np.where(~np.isfinite(X))[0])
    if len(bad_idx) > 0:
        print("\n[DEBUG] NaN/Inf 포함 POIs (first 20):")
        for i in bad_idx[:20]:
            print(f"  - idx={i}, poi_id={pois[i]['poi_id']}")



def main():
    pois = load_embeddings(INPUT_PATH)
    print(f"[INFO] Loaded embeddings: {len(pois)} POIs")
    # ===========================    
    debug_check_embeddings(pois)
    # ===========================
    id_to_index = {p["poi_id"]: i for i, p in enumerate(pois)}

    if TARGET_POI_ID not in id_to_index:
        print(f"[ERROR] target poi_id '{TARGET_POI_ID}' not found.")
        return

    idx = id_to_index[TARGET_POI_ID]
    print(f"[TARGET] poi_id: {TARGET_POI_ID}")
    print(f"         text  : {pois[idx]['profile_text']}")
    
    # ---- similarity vector 계산 ----
    X = np.stack([p["embedding"] for p in pois], axis=0)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    # ---- 전체 similarity 분포 분석 ----
    print("\n===== [ANALYSIS] 전체 cosine similarity 분포 =====")
    S = Xn @ Xn.T     # (N, N)
    upper = S[np.triu_indices_from(S, k=1)]
    print("mean sim:", upper.mean())
    print("std sim :", upper.std())
    print("min sim :", upper.min())
    print("max sim :", upper.max())
    print("==============================================\n")
    sims = Xn.dot(Xn[idx])

    sims[idx] = -1  # 자기 자신 제외
    

    # Top-K 인덱스
    top_idx = sims.argsort()[::-1][:TOPK]

    print("\n===== 🔥 TOP-K Similar POIs =====")
    for rank, j in enumerate(top_idx, start=1):
        print(f"{rank:2d}. sim={sims[j]:.4f} | poi_id={pois[j]['poi_id']}")
        print(f"     text: {pois[j]['profile_text']}\n")


if __name__ == "__main__":
    main()
