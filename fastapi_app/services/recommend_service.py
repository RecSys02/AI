from typing import List

from sentence_transformers import SentenceTransformer

from services.reranker import RerankerService
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
        self.reranker = RerankerService()  # Reranker 서비스 초기화
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
                "distance_weight": 0.1,
                "distance_scale_km": 5.0,
            },
            "restaurant": {
                "recent_weight": 0.3,
                "distance_weight": 0.3,  # 식당은 거리 영향↑
                "distance_scale_km": 2.0,  # 가까운 곳을 더 선호
            },
            "cafe": {
                "recent_weight": 0.3,
                "distance_weight": 0.3,  # 카페는 거리 영향↑
                "distance_scale_km": 2.0,  # 가까운 곳을 더 선호
            },
        }

    def _format_response(self, recommendations: List[dict], debug: bool = False) -> List[dict]:
        if debug:
            return recommendations

        # 사용자가 요청한 최종 응답 필드
        public_fields = [
            "category",
            "region",
            "place_id",
            "score",
        ]
        
        cleaned_recommendations = []
        for category_group in recommendations:
            cleaned_items = []
            for item in category_group["items"]:
                # 'province'를 'region'으로 매핑하고, 없는 필드는 None으로 처리
                cleaned_item = {
                    "category": item.get("category"),
                    "region": item.get("region") or item.get("province"),
                    "place_id": item.get("place_id"),
                    "score": item.get("score"),
                }
                # 요청된 필드만 포함하도록 다시 필터링 (혹시 모를 None 값 등 제외)
                final_item = {key: cleaned_item.get(key) for key in public_fields}
                cleaned_items.append(final_item)
            
            cleaned_recommendations.append(
                {"category": category_group["category"], "items": cleaned_items}
            )
        
        return cleaned_recommendations

    def recommend(
        self, user, top_k_per_category: int = 10, distance_max_km: float = 3.0, debug: bool = False
    ) -> List[dict]:
        per_category = {}
        selected_all = getattr(user, "selected_places", None) or getattr(user, "last_selected_pois", None) or []
        history_all = getattr(user, "history_places", None) or []
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
            # recency 보너스는 같은 카테고리의 방문 이력 기준
            recent_place_ids = [
                poi.place_id
                for poi in history_all
                if getattr(poi, "category", None) == scorer.name and getattr(poi, "place_id", None) is not None
            ]

            weights = self.category_weights.get(scorer.name, {})
            # 1. 1차 후보군 생성 (Scoring)
            candidates = scorer.topk(
                user_vec,
                top_k=top_k_per_category * 2,  # Reranker를 위해 더 많은 후보군 확보
                recent_place_ids=recent_place_ids,
                distance_place_ids=distance_place_ids,
                recent_weight=weights.get("recent_weight", 0.3),
                distance_weight=weights.get("distance_weight", 0.2),
                distance_scale_km=weights.get("distance_scale_km", 5.0),
                distance_max_km=distance_max_km,
                debug=debug,
                user_text=user_text,  # 음식 카테고리 필터링을 위한 사용자 텍스트 전달
            )
            # 2. 방문 이력 제외
            filtered_candidates = [r for r in candidates if r["place_id"] not in history_ids]

            # 3. 2차 후보군 생성 (Reranking)
            reranked_candidates = self.reranker.rerank(user, filtered_candidates, debug=debug)

            # 4. 최종 top_k 만큼 잘라서 결과 저장
            per_category[scorer.name] = reranked_candidates[:top_k_per_category]

        # 결과를 카테고리별로 리스트로 묶어 반환
        recommendations: List[dict] = []
        for category, items in per_category.items():
            recommendations.append({"category": category, "items": items})

        # 5. 최종 응답 포맷 정리
        return self._format_response(recommendations, debug=debug)
