from typing import List

from sentence_transformers import SentenceTransformer

from services.scorers import (
    build_cafe_scorer,
    build_restaurant_scorer,
    build_tourspot_scorer,
)
from utils.user_text_builder import build_user_profile_text

class RecommendService:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")
        self.scorers = [
            build_tourspot_scorer(),
            build_cafe_scorer(),
            build_restaurant_scorer(),
        ]

    def recommend(self, user, top_k_per_category: int = 10) -> List[dict]:
        user_text = build_user_profile_text(user)
        user_vec = self.model.encode(
            [user_text],
            normalize_embeddings=True,
        )[0]

        per_category = {}
        for scorer in self.scorers:
            per_category[scorer.name] = scorer.topk(user_vec, top_k_per_category)

        # 결과를 카테고리별로 리스트로 묶어 반환
        recommendations: List[dict] = []
        for category, items in per_category.items():
            recommendations.append({"category": category, "items": items})

        return recommendations
