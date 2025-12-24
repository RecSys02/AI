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
        # 카테고리별 가중치/스케일 설정
        self.category_weights = {
            "tourspot": {
                "recent_weight": 0.3,
                "distance_weight": 0.2,
                "distance_scale_km": 5.0,
            },
            "restaurant": {
                "recent_weight": 0.3,
                "distance_weight": 0.25,
                "distance_scale_km": 4.0,
            },
            "cafe": {
                "recent_weight": 0.3,
                "distance_weight": 1.0,  # 카페는 거리 영향↑
                "distance_scale_km": 2.0, # 가까운 곳을 더 선호
            },
        }

    def recommend(self, user, top_k_per_category: int = 10, distance_max_km: float = 3.0, debug: bool = False) -> List[dict]:
        per_category = {}
        selected_all = getattr(user, "selectedPlaces", None) or getattr(user, "last_selected_pois", None) or []
        history_all = getattr(user, "historyPlaces", None) or []
        history_ids = {p.place_id for p in history_all if getattr(p, "place_id", None) is not None}
        # 거리 계산은 마지막 선택 장소 1개 기준(카테고리 무관)
        distance_place_ids = []
        if selected_all:
            last = selected_all[-1]
            if getattr(last, "place_id", None) is not None:
                distance_place_ids = [last.place_id]

        for scorer in self.scorers:
            builder = self.text_builders.get(scorer.name)
            user_text = builder(user) if builder else ""
            user_vec = self.model.encode(
                [user_text],
                normalize_embeddings=True,
            )[0]
            # 거리/recency 보너스는 selected(또는 last_selected_pois)가 있을 때만 사용
            selected = selected_all
            recent_place_ids = []
            if selected:
                recent_place_ids = [
                    poi.place_id
                    for poi in selected
                    if getattr(poi, "category", None) == scorer.name and getattr(poi, "place_id", None) is not None
                ]

            weights = self.category_weights.get(scorer.name, {})
            per_category[scorer.name] = scorer.topk(
                user_vec,
                top_k=top_k_per_category,
                recent_place_ids=recent_place_ids,
                distance_place_ids=distance_place_ids,
                recent_weight=weights.get("recent_weight", 0.3),
                distance_weight=weights.get("distance_weight", 0.2),
                distance_scale_km=weights.get("distance_scale_km", 5.0),
                distance_max_km=distance_max_km,
                debug=debug,
            )
            # 방문 이력(place_id 기준) 제외
            per_category[scorer.name] = [
                r for r in per_category[scorer.name] if r["place_id"] not in history_ids
            ][:top_k_per_category]

        # 결과를 카테고리별로 리스트로 묶어 반환
        recommendations: List[dict] = []
        for category, items in per_category.items():
            recommendations.append({"category": category, "items": items})

        return recommendations
