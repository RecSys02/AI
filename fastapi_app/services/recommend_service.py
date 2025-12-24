from typing import List

from sentence_transformers import SentenceTransformer

from services.scorers import (
    build_cafe_scorer,
    build_restaurant_scorer,
    build_tourspot_scorer,
)
from utils.user_text_builder import (
    build_cafe_text,
    build_restaurant_text,
    build_tourspot_text,
)

class RecommendService:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")
        self.scorers = [
            build_tourspot_scorer(),
            build_cafe_scorer(),
            build_restaurant_scorer(),
        ]
        self.text_builders = {
            "tourspot": build_tourspot_text,
            "cafe": build_cafe_text,
            "restaurant": build_restaurant_text,
        }

    def recommend(self, user, top_k_per_category: int = 10, distance_max_km: float = 3.0) -> List[dict]:
        per_category = {}
        for scorer in self.scorers:
            builder = self.text_builders.get(scorer.name)
            user_text = builder(user) if builder else ""
            user_vec = self.model.encode(
                [user_text],
                normalize_embeddings=True,
            )[0]
            # 거리/recency 보너스는 selected(또는 last_selected_pois)가 있을 때만 사용
            selected = getattr(user, "selectedPlaces", None) or getattr(user, "last_selected_pois", None)
            recent_place_ids = []
            if selected:
                recent_place_ids = [
                    poi.id for poi in selected if getattr(poi, "category", None) == scorer.name
                ]

            per_category[scorer.name] = scorer.topk(
                user_vec,
                top_k=top_k_per_category,
                recent_place_ids=recent_place_ids,
                recent_weight=0.3,
                distance_max_km=distance_max_km,
            )

        # 결과를 카테고리별로 리스트로 묶어 반환
        recommendations: List[dict] = []
        for category, items in per_category.items():
            recommendations.append({"category": category, "items": items})

        return recommendations
