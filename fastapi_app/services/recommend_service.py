import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

from openai import OpenAI
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
                "distance_weight": 0.1,
                "distance_scale_km": 5.0,
            },
            "restaurant": {
                "recent_weight": 0.3,
                "distance_weight": 0.3, # 식당은 거리 영향↑
                "distance_scale_km": 2.0, # 가까운 곳을 더 선호
            },
            "cafe": {
                "recent_weight": 0.3,
                "distance_weight": 0.3,  # 카페는 거리 영향↑
                "distance_scale_km": 2.0, # 가까운 곳을 더 선호
            },
        }
        # OpenAI 클라이언트 초기화
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # CSV 로깅 설정
        self.enable_csv_logging = os.getenv("RERANK_CSV_LOG", "false").lower() == "true"
        if self.enable_csv_logging:
            self.csv_log_path = Path(__file__).resolve().parents[2] / "logs" / "rerank_comparison.csv"
            self.csv_log_path.parent.mkdir(parents=True, exist_ok=True)
            # CSV 헤더 생성 (파일이 없을 때만)
            if not self.csv_log_path.exists():
                with open(self.csv_log_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "user_id", "category", "place_id",
                        "name", "rank_before", "rank_after", "rank_change",
                        "score", "score_base", "score_recent", "score_distance"
                    ])

    # 음식 종류 강제 필터링 제거: 복수 선호 타입 지원 및 유연한 추천을 위해
    # 임베딩 텍스트(user_text_builder.py)에 "세계음식 > 양식" 형태로 선호도 포함
    # LLM reranking에서 선호도를 고려하여 자연스럽게 가중치 반영

    def _log_rerank_to_csv(self, user, category: str, candidates: List[dict], reranked: List[dict], ranked_indices: List[int], top_k: int):
        """CSV 파일에 리랭킹 비교 로그 저장"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user_id = getattr(user, "user_id", "unknown")

            with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for idx in range(min(top_k, len(reranked))):
                    item = reranked[idx]
                    original_idx = ranked_indices[idx]
                    original_candidate = candidates[original_idx]

                    rank_before = original_idx + 1
                    rank_after = idx + 1
                    rank_change = rank_before - rank_after

                    meta = original_candidate.get("_meta", {})
                    name = meta.get("name", "N/A")

                    writer.writerow([
                        timestamp,
                        user_id,
                        category,
                        item["place_id"],
                        name,
                        rank_before,
                        rank_after,
                        rank_change,
                        item.get("score", 0),
                        item.get("score_base", 0),
                        item.get("score_recent", 0),
                        item.get("score_distance", 0)
                    ])
        except Exception as e:
            print(f"[RERANK] Failed to log to CSV: {e}")

    def _llm_rerank(self, user, category: str, candidates: List[dict], top_k: int = 10, debug: bool = False) -> List[dict]:
        """
        LLM을 사용하여 후보 POI들을 사용자 맥락에 맞게 재정렬

        Args:
            user: 사용자 정보 객체
            category: POI 카테고리 (tourspot/cafe/restaurant)
            candidates: 재정렬할 후보 POI 리스트
            top_k: 최종 반환할 개수
            debug: 디버그 모드 (리랭킹 전 순서 정보 포함)

        Returns:
            재정렬된 POI 리스트 (최대 top_k개)
        """
        if len(candidates) <= top_k:
            # 내부 메타데이터 제거 후 반환
            result = [{k: v for k, v in item.items() if k != "_meta"} for item in candidates]
            # debug 모드에서는 원본 순서 정보 추가
            if debug:
                for idx, item in enumerate(result):
                    item["rank_before_rerank"] = idx + 1
                    item["rank_after_rerank"] = idx + 1
            return result

        # 사용자 프로필 텍스트 생성
        user_profile_parts = []

        if hasattr(user, 'preferred_themes') and user.preferred_themes:
            user_profile_parts.append(f"선호 테마: {', '.join(user.preferred_themes)}")
        if hasattr(user, 'preferred_moods') and user.preferred_moods:
            user_profile_parts.append(f"선호 분위기: {', '.join(user.preferred_moods)}")
        if hasattr(user, 'activity_level') and user.activity_level:
            user_profile_parts.append(f"활동 수준: {user.activity_level}")
        if hasattr(user, 'companion') and user.companion:
            user_profile_parts.append(f"동행: {', '.join(user.companion)}")
        if hasattr(user, 'budget') and user.budget:
            user_profile_parts.append(f"예산: {user.budget}")
        if hasattr(user, 'avoid') and user.avoid:
            user_profile_parts.append(f"회피 사항: {', '.join(user.avoid)}")

        if category == "restaurant" and hasattr(user, 'preferred_restaurant_types') and user.preferred_restaurant_types:
            user_profile_parts.append(f"선호 음식점 종류: {', '.join(user.preferred_restaurant_types)}")
        elif category == "cafe" and hasattr(user, 'preferred_cafe_types') and user.preferred_cafe_types:
            user_profile_parts.append(f"선호 카페 종류: {', '.join(user.preferred_cafe_types)}")

        user_profile = "\n".join(user_profile_parts) if user_profile_parts else "특별한 선호도 없음"

        # 음식 종류 사전 필터링 (복수 선호 타입 지원)
        filtered_candidates = candidates
        if category in ["restaurant", "cafe"]:
            # 사용자 선호 음식/카페 종류 추출
            preferred_types = []
            if category == "restaurant" and hasattr(user, 'preferred_restaurant_types') and user.preferred_restaurant_types:
                preferred_types = user.preferred_restaurant_types
            elif category == "cafe" and hasattr(user, 'preferred_cafe_types') and user.preferred_cafe_types:
                preferred_types = user.preferred_cafe_types

            if preferred_types:
                # 음식 종류 매핑 (다양한 표현 지원)
                type_mapping = {
                    "한식": ["한국음식", "한식"],
                    "한국음식": ["한국음식", "한식"],
                    "중식": ["중식", "중국음식"],
                    "중국음식": ["중식", "중국음식"],
                    "일식": ["일식", "일본음식"],
                    "일본음식": ["일식", "일본음식"],
                    "양식": ["양식", "서양음식"],
                    "서양음식": ["양식", "서양음식"],
                    # 카페 타입 추가
                    "커피전문점": ["커피전문점", "커피"],
                    "디저트": ["디저트", "디저트 위주"],
                    "디저트 위주": ["디저트", "디저트 위주"],
                    "베이커리": ["베이커리", "빵"],
                }

                # 사용자가 선호하는 모든 키워드 수집
                allowed_keywords = set()
                for pref in preferred_types:
                    # 쉼표로 구분된 경우 분리
                    pref_items = [p.strip() for p in pref.split(',')]
                    for item in pref_items:
                        if item in type_mapping:
                            allowed_keywords.update(type_mapping[item])
                        else:
                            allowed_keywords.add(item)

                # 필터링 수행
                filtered = []
                for poi in candidates:
                    meta = poi.get("_meta", {})
                    content = meta.get("content", "").strip()

                    # content가 없으면 포함 (필터링 안함)
                    if not content:
                        filtered.append(poi)
                        continue

                    # content에 선호 키워드가 하나라도 포함되어 있으면 포함
                    content_lower = content.lower()
                    matched = any(keyword.lower() in content_lower for keyword in allowed_keywords)

                    if matched:
                        filtered.append(poi)
                    else:
                        # 필터링된 항목 로그
                        name = meta.get("name", "N/A")
                        print(f"[RERANK] Filtered out: {name} (food_type: {content})")

                # 필터링 후 충분한 개수가 남아있는지 확인
                if len(filtered) >= top_k:
                    filtered_candidates = filtered
                    print(f"[RERANK] {category}: Filtered {len(candidates)} → {len(filtered_candidates)} candidates (preferred: {', '.join(preferred_types)})")
                else:
                    # 필터 결과가 부족하면 원본 유지 (과도한 필터링 방지)
                    print(f"[RERANK] {category}: Not enough candidates after filtering ({len(filtered)} < {top_k}), keeping original list")
                    filtered_candidates = candidates

        # 후보 POI 정보 간략화 (place_id와 주요 정보만)
        candidates = filtered_candidates
        poi_list = []
        for idx, poi in enumerate(candidates):
            poi_info = {"index": idx, "place_id": poi["place_id"]}
            # scorer에서 반환한 메타데이터가 있다면 활용
            if "_meta" in poi:
                meta = poi["_meta"]
                if "name" in meta:
                    poi_info["name"] = meta["name"]
                if "summary_one_sentence" in meta:
                    poi_info["summary"] = meta["summary_one_sentence"]
                if "description" in meta:
                    poi_info["description"] = meta.get("description", "")[:200]  # 길이 제한
                if "themes" in meta:
                    poi_info["themes"] = meta.get("themes", [])
                if "keywords" in meta:
                    poi_info["keywords"] = meta.get("keywords", [])
                if "category" in meta:
                    poi_info["category"] = meta["category"]
                # 음식점/카페의 경우 content 필드 추가 (음식 종류 정보)
                if "content" in meta and category in ["restaurant", "cafe"]:
                    poi_info["food_type"] = meta["content"]
            poi_list.append(poi_info)

        # LLM 프롬프트 구성
        category_name = {"restaurant": "음식점", "cafe": "카페", "tourspot": "관광지"}.get(category, category)

        # 음식점/카페인 경우 음식 종류 필터링 지침 추가
        food_type_instruction = ""
        if category in ["restaurant", "cafe"]:
            food_type_instruction = """
**중요: 음식 종류 필터링 규칙**
1. 사용자가 "한식", "한국음식" 등을 선호한다면, food_type이 "중식", "일식", "양식" 등 다른 음식 종류인 후보는 반드시 순위를 낮춰야 합니다.
2. 사용자의 선호 음식 종류와 일치하는 후보를 우선적으로 선택하세요.
3. food_type 필드가 있는 경우, 이를 가장 중요한 판단 기준으로 삼으세요.
4. 예: 사용자가 한식을 선호하는데 "세계음식 > 중식"이라고 표시된 음식점은 순위를 대폭 낮춰야 합니다.
"""

        prompt = f"""다음은 사용자에게 추천할 {category_name} 후보 목록입니다.
사용자의 선호도와 맥락을 고려하여 가장 적합한 순서대로 상위 {top_k}개를 선택하여 재정렬해주세요.

[사용자 프로필]
{user_profile}
{food_type_instruction}
[후보 목록]
{json.dumps(poi_list, ensure_ascii=False, indent=2)}

응답은 반드시 다음 JSON 형식으로만 출력해주세요:
{{"ranked_indices": [index1, index2, ..., index{top_k}]}}

ranked_indices는 위 후보 목록의 index 값들을 재정렬한 배열입니다."""

        try:
            response = self.openai_client.chat.completions.create(
                model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "당신은 여행 POI 추천 전문가입니다. 사용자의 선호도를 분석하여 최적의 장소를 추천합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            ranked_indices = result.get("ranked_indices", [])

            # 유효성 검증
            if not ranked_indices or not all(isinstance(i, int) and 0 <= i < len(candidates) for i in ranked_indices):
                # LLM 응답이 잘못된 경우 원본 순서 유지 (내부 메타데이터 제거)
                result = [{k: v for k, v in item.items() if k != "_meta"} for item in candidates[:top_k]]
                if debug:
                    for idx, item in enumerate(result):
                        item["rank_before_rerank"] = idx + 1
                        item["rank_after_rerank"] = idx + 1
                return result

            # 재정렬된 결과 생성 (내부 메타데이터 제거)
            reranked = []
            for new_rank, original_idx in enumerate(ranked_indices, start=1):
                if original_idx < len(candidates):
                    item = {k: v for k, v in candidates[original_idx].items() if k != "_meta"}
                    rank_before = original_idx + 1
                    rank_after = new_rank
                    rank_change = rank_before - rank_after  # 양수면 순위 상승, 음수면 하락

                    if debug:
                        item["rank_before_rerank"] = rank_before
                        item["rank_after_rerank"] = rank_after
                    reranked.append(item)

            # 콘솔 로그 출력
            print(f"\n[RERANK] {category.upper()} - Reranking Results:")
            print(f"{'Rank':>4} {'→':>3} {'Rank':>4} | {'Change':>6} | {'Place ID':>8} | {'Name':<30} | {'Score'}")
            print("-" * 95)
            for idx, item in enumerate(reranked[:top_k], start=1):
                original_candidate = candidates[ranked_indices[idx-1]]
                rank_before = ranked_indices[idx-1] + 1
                rank_change = rank_before - idx
                change_symbol = "↑" if rank_change > 0 else "↓" if rank_change < 0 else "="
                name = original_candidate.get("_meta", {}).get("name", "N/A")[:30]
                score = item.get("score", 0)
                print(f"{rank_before:>4} {change_symbol:>3} {idx:>4} | {rank_change:>+6} | {item['place_id']:>8} | {name:<30} | {score:.4f}")

            # CSV 로그 저장
            if self.enable_csv_logging:
                self._log_rerank_to_csv(user, category, candidates, reranked, ranked_indices, top_k)

            return reranked[:top_k]

        except Exception as e:
            # LLM 호출 실패 시 원본 순서 유지 (내부 메타데이터 제거)
            print(f"[RERANK] LLM reranking failed for {category}: {e}")
            result = [{k: v for k, v in item.items() if k != "_meta"} for item in candidates[:top_k]]
            if debug:
                for idx, item in enumerate(result):
                    item["rank_before_rerank"] = idx + 1
                    item["rank_after_rerank"] = idx + 1
            return result

    def recommend(self, user, top_k_per_category: int = 10, distance_max_km: float = 3.0, debug: bool = False) -> List[dict]:
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
            # 임베딩 기반으로 top-15 추출 (reranking을 위한 후보군)
            # 음식 종류는 임베딩 텍스트에 포함되어 자연스럽게 가중치 반영
            initial_k = 15
            per_category[scorer.name] = scorer.topk(
                user_vec,
                top_k=initial_k,
                recent_place_ids=recent_place_ids,
                distance_place_ids=distance_place_ids,
                recent_weight=weights.get("recent_weight", 0.3),
                distance_weight=weights.get("distance_weight", 0.2),
                distance_scale_km=weights.get("distance_scale_km", 5.0),
                distance_max_km=distance_max_km,
                debug=debug,
                include_meta=True,  # LLM reranking을 위해 메타데이터 포함
            )
            # 방문 이력(place_id 기준) 제외
            candidates = [
                r for r in per_category[scorer.name] if r["place_id"] not in history_ids
            ][:initial_k]

            # LLM reranking으로 최종 top-10 선택
            per_category[scorer.name] = self._llm_rerank(
                user=user,
                category=scorer.name,
                candidates=candidates,
                top_k=top_k_per_category,
                debug=debug
            )

        # 결과를 카테고리별로 리스트로 묶어 반환
        recommendations: List[dict] = []
        for category, items in per_category.items():
            recommendations.append({"category": category, "items": items})

        return recommendations
