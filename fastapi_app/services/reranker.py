# fastapi_app/services/reranker.py
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRanker  # [변경] Classifier -> Ranker
from models.user_input import UserInput

logger = logging.getLogger(__name__)

class RerankerService:
    def __init__(self, model_path: str = "data/models/catboost_model.cbm"):
        # __file__ 기준 프로젝트 루트 경로 계산
        project_root = Path(__file__).resolve().parents[2]
        self.model_path = project_root / model_path
        self.model = self._load_model()

    def _load_model(self) -> Optional[CatBoostRanker]:
        if not self.model_path.exists():
            logger.warning(f"Reranker model not found at {self.model_path}, reranking will be skipped.")
            return None
        try:
            # [변경] Ranker로 로드해야 함
            model = CatBoostRanker()
            model.load_model(str(self.model_path))
            logger.info(f"Reranker model loaded from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            return None

    def _feature_engineering(self, user: UserInput, candidates: List[Dict]) -> pd.DataFrame:
        """
        Reranking 모델에 사용할 피처를 생성합니다.
        학습 단계(train_reranker.py)와 동일한 전처리를 수행해야 합니다.
        """
        features = []
        for c in candidates:
            # Debug 정보에서 two-tower 점수들을 추출
            debug_info = c.get("debug", {})

            # Two-tower 시스템 점수 (debug에 있으면 추출, 없으면 기본값)
            base_score = debug_info.get("score_base", c.get("score", 0.5))
            recent_score = debug_info.get("score_recent", 0.0)
            distance_score = debug_info.get("score_distance", 0.0)
            distance_km = c.get("distance_km", 5.0)

            # JSON 메타데이터에서 추가 정보 추출
            keywords = c.get("keywords", [])
            keyword_count = len(keywords) if isinstance(keywords, list) else 0

            # Content 필드 분석 (restaurant, cafe에서 중요)
            content = c.get("content", "")
            content_length = len(str(content)) if content else 0
            has_content = 1 if content_length > 0 else 0

            # 카테고리 확인
            category = c.get("category", "")
            is_restaurant_or_cafe = 1 if category in ["restaurant", "cafe"] else 0
            content_weight = is_restaurant_or_cafe * content_length

            features.append({
                "poi_id": c["place_id"],
                # Two-tower 점수
                "base_score": base_score,
                "recent_score": recent_score,
                "distance_score": distance_score,
                "distance_km": distance_km,
                # 기본 메타데이터
                "rating": c.get("rating", 3.5),
                "review_count": c.get("review_count", 0),
                # JSON 파일의 추가 정보
                "views": c.get("views", 0),
                "likes": c.get("likes", 0),
                "bookmarks": c.get("bookmarks", 0),
                "keyword_count": keyword_count,
                # Content 필드 특징 (restaurant, cafe 강화)
                "content_length": content_length,
                "has_content": has_content,
                "is_restaurant_or_cafe": is_restaurant_or_cafe,
                "content_weight": content_weight,
            })

        df = pd.DataFrame(features)

        # 결측치 처리 (학습 코드와 동일)
        df['base_score'] = df['base_score'].fillna(0.5)
        df['recent_score'] = df['recent_score'].fillna(0.0)
        df['distance_score'] = df['distance_score'].fillna(0.0)
        df['distance_km'] = df['distance_km'].fillna(5.0)
        df['rating'] = df['rating'].fillna(3.5)
        df['review_count'] = df['review_count'].fillna(0)
        df['views'] = df['views'].fillna(0)
        df['likes'] = df['likes'].fillna(0)
        df['bookmarks'] = df['bookmarks'].fillna(0)
        df['keyword_count'] = df['keyword_count'].fillna(0)
        df['content_length'] = df['content_length'].fillna(0)
        df['has_content'] = df['has_content'].fillna(0)
        df['is_restaurant_or_cafe'] = df['is_restaurant_or_cafe'].fillna(0)
        df['content_weight'] = df['content_weight'].fillna(0)

        # 파생 특징 생성 (학습 코드와 동일)
        df['total_score'] = df['base_score'] + df['recent_score'] + df['distance_score']
        df['popularity_score'] = np.log1p(df['views']) + np.log1p(df['likes']) * 2 + np.log1p(df['bookmarks']) * 3
        df['engagement_rate'] = (df['likes'] + df['bookmarks']) / (df['views'] + 1)

        # Content 관련 파생 특징 (restaurant, cafe 강화)
        df['content_boost'] = df['is_restaurant_or_cafe'] * df['has_content'] * 2.0
        df['content_popularity'] = df['is_restaurant_or_cafe'] * df['popularity_score'] * (1 + df['has_content'])

        return df

    def rerank(self, user: UserInput, candidates: List[Dict], debug: bool = False) -> List[Dict]:
        if not self.model:
            logger.info("Reranker model not loaded, skipping reranking.")
            return candidates

        if not candidates:
            return []

        # 1. 피처 생성
        feature_df = self._feature_engineering(user, candidates)

        # 2. 모델 피처 순서 맞추기
        try:
            model_features = self.model.feature_names_
        except AttributeError:
             # 모델 로드 직후 feature_names_ 접근이 안 될 경우 대비
            logger.warning("Model feature names not found. Skipping rerank.")
            return candidates

        feature_df_for_pred = pd.DataFrame()
        
        for col in model_features:
            if col in feature_df.columns:
                feature_df_for_pred[col] = feature_df[col]
            else:
                # 모델이 요구하는 피처가 없으면 기본값 (학습 때 사용한 기본값과 맞춰야 함)
                # logger.warning(f"Feature '{col}' not found, using default 0.")
                feature_df_for_pred[col] = 0

        # 3. Reranking 점수 예측
        try:
        # 1) Ranker의 Raw Score 예측 (음수 나올 수 있음, 예: -1.5, 0.2, 3.1 ...)
            raw_scores = self.model.predict(feature_df_for_pred)
        
        # 2) [핵심] Sigmoid 함수로 0~1 사이 확률값처럼 변환
        # 수식: 1 / (1 + e^-x)
        # raw_scores가 numpy array이므로 한 번에 연산됩니다.
            prob_scores = 1 / (1 + np.exp(-raw_scores))
        
        # 3) 동점 방지용 Random Jitter (0.00001 ~ 0.00005)
            jitter = np.random.uniform(0.00001, 0.00005, size=len(prob_scores))
        
        # 4) [선택] 리뷰 수 가산점 (Rule-base)
        # 리뷰가 많을수록 점수를 조금 더 줌 (Log scale)
        # 리뷰 0개일 때 에러 방지를 위해 fillna(0)
            log_reviews = np.log1p(feature_df["review_count"].fillna(0)) * 0.01
        
        # 5) 최종 점수 합산
            feature_df["rerank_score"] = prob_scores + jitter + log_reviews

        except Exception as e:
            logger.error(f"Reranking prediction failed: {e}")
            return candidates

        # 4. 점수 업데이트 및 재정렬
        poi_map = {c["place_id"]: c for c in candidates}
        
        for _, row in feature_df.iterrows():
            poi_id = row["poi_id"]
            if poi_id in poi_map:
                new_score = float(row["rerank_score"]) # numpy float -> python float
                
                if debug:
                    if "debug" not in poi_map[poi_id]:
                        poi_map[poi_id]["debug"] = {}
                    poi_map[poi_id]["debug"]["rerank_score"] = new_score
                    poi_map[poi_id]["debug"]["original_score"] = poi_map[poi_id].get("score", 0)
                
                # 최종 점수 업데이트
                poi_map[poi_id]["score"] = new_score

        # 점수 높은 순 정렬
        reranked_candidates = sorted(
            poi_map.values(), key=lambda x: x.get("score", 0.0), reverse=True
        )

        logger.info(f"Reranked {len(candidates)} candidates.")
        return reranked_candidates