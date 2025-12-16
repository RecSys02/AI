# recommend.py
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.user_text_builder import build_user_profile_text

MODEL_NAME = "BAAI/bge-m3"
model = SentenceTransformer(MODEL_NAME)

poi_embeddings = np.load("embedding/poi_embeddings.npy")      # (N, D)
poi_ids = np.load("embedding/poi_ids.npy", allow_pickle=True) # (N,)

def recommend_pois(user, top_k: int = 10, return_latency: bool = False):
    t0 = time.perf_counter()

    # 1️⃣ 사용자 텍스트 생성
    user_text = build_user_profile_text(user)
    t1 = time.perf_counter()

    # 2️⃣ 사용자 임베딩
    user_vec = model.encode(
        [user_text],
        normalize_embeddings=True
    )[0]
    t2 = time.perf_counter()

    # 3️⃣ cosine similarity
    scores = np.dot(poi_embeddings, user_vec)
    t3 = time.perf_counter()

    # 4️⃣ top-k 추출
    top_indices = scores.argsort()[::-1][:top_k]
    t4 = time.perf_counter()

    results = [
        {
            "poi_id": str(poi_ids[idx]),
            "score": float(scores[idx])
        }
        for idx in top_indices
    ]

    latency = {
        "build_user_text": round(t1 - t0, 4),
        "encode_user": round(t2 - t1, 4),
        "similarity": round(t3 - t2, 4),
        "topk": round(t4 - t3, 4),
        "total": round(t4 - t0, 4),
    }

    if return_latency:
        return results, latency

    return results
