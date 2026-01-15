import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import train_test_split

# --- 경로 설정 ---
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "embedding_json"
MODEL_DIR = ROOT / "data" / "models"
OUTPUT_MODEL_PATH = MODEL_DIR / "catboost_model.cbm"

# --- 유틸리티 함수 ---
def load_all_poi_data() -> pd.DataFrame:
    """/data/embedding_json/ 폴더의 모든 POI JSON을 읽어 하나의 DataFrame으로 합칩니다."""
    all_pois = []
    # 데이터가 없다면 빈 리스트 처리에 대한 방어 코드 필요하지만, 여기선 있다고 가정
    for json_path in DATA_DIR.glob("*.json"):
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            all_pois.extend(data)
    
    df = pd.DataFrame(all_pois)

    # 시뮬레이션을 위한 가상 평점/리뷰 수 추가
    if 'rating' not in df.columns:
        df['rating'] = np.random.uniform(3.5, 5.0, size=len(df)).round(2)
    if 'review_count' not in df.columns:
        df['review_count'] = np.random.randint(1, 500, size=len(df))
    
    # 누락된 값 기본값으로 채우기
    df['rating'] = df['rating'].fillna(3.5)
    df['review_count'] = df['review_count'].fillna(0)
    
    print(f"전체 POI 데이터 로드 완료: {len(df)}개")
    return df

def generate_synthetic_interactions(poi_df: pd.DataFrame, num_sessions: int) -> pd.DataFrame:
    print(f"가상 상호작용 데이터 생성 시작 (세션 수: {num_sessions})...")
    interactions = []
    poi_list = poi_df.to_dict('records')

    for i in range(num_sessions):
        impressions = random.sample(poi_list, k=min(20, len(poi_list)))

        # 가상 two-tower 점수 부여
        for p in impressions:
            p['_base_score'] = np.random.uniform(0.5, 1.0)
            p['_recent_score'] = np.random.uniform(0.0, 0.3)
            p['_distance_score'] = np.random.uniform(0.0, 0.2)
            p['_distance_km'] = np.random.uniform(0.5, 10.0)

        # 가중치 계산 (평점, 리뷰수, two-tower 점수 기반)
        weights = [
            p['rating'] * np.log1p(p['review_count']) * (p['_base_score'] ** 2)
            for p in impressions
        ]

        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights] if total_weight > 0 else None

        if probabilities:
            clicked_poi = random.choices(impressions, weights=probabilities, k=1)[0]
        else:
            clicked_poi = random.choice(impressions)

        for p in impressions:
            interactions.append({
                "session_id": i,
                "poi_id": p["place_id"],
                "label": 1 if p["place_id"] == clicked_poi["place_id"] else 0,
                "base_score": p['_base_score'],
                "recent_score": p['_recent_score'],
                "distance_score": p['_distance_score'],
                "distance_km": p['_distance_km']
            })

    print(f"가상 데이터 생성 완료: {len(interactions)}개")
    return pd.DataFrame(interactions)

def load_interaction_data(
    poi_df: pd.DataFrame,
    interaction_file_path: Optional[Path] = None,
    synthetic_sessions: int = 2000
) -> pd.DataFrame:
    if interaction_file_path and interaction_file_path.exists():
        print(f"실제 상호작용 데이터 로드: {interaction_file_path}")
        df = pd.read_csv(interaction_file_path)
        print(f"로드 완료: {len(df)}개 상호작용")
        return df
    else:
        print("실제 상호작용 데이터 파일을 찾을 수 없어 가상 데이터를 생성합니다.")
        return generate_synthetic_interactions(poi_df, num_sessions=synthetic_sessions)


def feature_engineering(interactions_df: pd.DataFrame, poi_df: pd.DataFrame):
    """상호작용 데이터와 POI 메타데이터를 결합하여 피처를 생성합니다."""

    # place_id를 기준으로 두 데이터프레임 조인
    df = pd.merge(interactions_df, poi_df, left_on="poi_id", right_on="place_id")

    # Two-tower 시스템 점수 특징
    feature_columns = [
        'base_score',          # Two-tower 기본 유사도 점수
        'recent_score',        # Recency 보너스 점수
        'distance_score',      # 거리 보너스 점수
        'distance_km',         # 실제 거리 (km)
        'rating',              # 평점
        'review_count',        # 리뷰 수
        'views',               # 조회수
        'likes',               # 좋아요 수
        'bookmarks',           # 북마크 수
    ]

    # keyword 개수 특징 추가
    if 'keywords' in df.columns:
        df['keyword_count'] = df['keywords'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        feature_columns.append('keyword_count')

    # content 필드 길이 특징 (restaurant, cafe에서 중요)
    if 'content' in df.columns:
        df['content_length'] = df['content'].fillna('').astype(str).apply(len)
        df['has_content'] = (df['content'].fillna('').astype(str).str.len() > 0).astype(int)
        feature_columns.extend(['content_length', 'has_content'])

    # 카테고리별 content 가중치 (restaurant, cafe는 높게)
    if 'category' in df.columns and 'content' in df.columns:
        df['is_restaurant_or_cafe'] = df['category'].isin(['restaurant', 'cafe']).astype(int)
        df['content_weight'] = df['is_restaurant_or_cafe'] * df['content_length']
        feature_columns.extend(['is_restaurant_or_cafe', 'content_weight'])

    # 필요한 컬럼만 남기기
    available_features = [col for col in feature_columns if col in df.columns]
    df = df[available_features + ['label', 'session_id']].copy()

    # 결측치 처리
    df['base_score'] = df['base_score'].fillna(0.5)
    df['recent_score'] = df['recent_score'].fillna(0.0)
    df['distance_score'] = df['distance_score'].fillna(0.0)
    df['distance_km'] = df['distance_km'].fillna(5.0)  # 기본 거리 5km
    df['rating'] = df['rating'].fillna(3.5)
    df['review_count'] = df['review_count'].fillna(0)
    df['views'] = df['views'].fillna(0) if 'views' in df.columns else 0
    df['likes'] = df['likes'].fillna(0) if 'likes' in df.columns else 0
    df['bookmarks'] = df['bookmarks'].fillna(0) if 'bookmarks' in df.columns else 0
    if 'keyword_count' in df.columns:
        df['keyword_count'] = df['keyword_count'].fillna(0)
    if 'content_length' in df.columns:
        df['content_length'] = df['content_length'].fillna(0)
    if 'has_content' in df.columns:
        df['has_content'] = df['has_content'].fillna(0)
    if 'is_restaurant_or_cafe' in df.columns:
        df['is_restaurant_or_cafe'] = df['is_restaurant_or_cafe'].fillna(0)
    if 'content_weight' in df.columns:
        df['content_weight'] = df['content_weight'].fillna(0)

    # 파생 특징 생성
    df['total_score'] = df['base_score'] + df['recent_score'] + df['distance_score']
    df['popularity_score'] = np.log1p(df['views']) + np.log1p(df['likes']) * 2 + np.log1p(df['bookmarks']) * 3
    df['engagement_rate'] = (df['likes'] + df['bookmarks']) / (df['views'] + 1)

    # Content 관련 파생 특징 (restaurant, cafe 강화)
    if 'is_restaurant_or_cafe' in df.columns:
        # restaurant/cafe인 경우 content가 있으면 점수 부스트
        df['content_boost'] = df['is_restaurant_or_cafe'] * df['has_content'] * 2.0
        # restaurant/cafe인 경우 popularity와 content 결합
        df['content_popularity'] = df['is_restaurant_or_cafe'] * df['popularity_score'] * (1 + df['has_content'])
        available_features.extend(['content_boost', 'content_popularity'])

    available_features.extend(['total_score', 'popularity_score', 'engagement_rate'])

    # Ranker 학습을 위해 데이터를 session_id 기준으로 정렬
    df = df.sort_values(by='session_id').reset_index(drop=True)

    print("피처 엔지니어링 완료. 생성된 피처:", available_features)
    print(f"학습 데이터 크기: {len(df)} rows")

    return df, available_features

def train_model(feature_df: pd.DataFrame, feature_columns: List[str]):
    """CatBoostRanker 학습 및 저장"""
    
    X = feature_df[feature_columns]
    
    # 데이터 분할
    groups = feature_df['session_id']
    unique_sessions = groups.unique()
    train_sessions, val_sessions = train_test_split(unique_sessions, test_size=0.2, random_state=42)
    
    train_df = feature_df[feature_df['session_id'].isin(train_sessions)]
    val_df = feature_df[feature_df['session_id'].isin(val_sessions)]

    # [수정] cat_features는 이제 빈 리스트 [] 가 됩니다.
    cat_features_indices = [] 

    # CatBoost Pool 생성
    train_pool = Pool(
        data=train_df[feature_columns],
        label=train_df['label'],
        group_id=train_df['session_id'],
        cat_features=cat_features_indices
    )
    
    val_pool = Pool(
        data=val_df[feature_columns],
        label=val_df['label'],
        group_id=val_df['session_id'],
        cat_features=cat_features_indices
    )

    print("\n--- Ranker 모델 학습 시작 ---")
    model = CatBoostRanker(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='YetiRank',
        eval_metric='NDCG:top=10',
        random_seed=42,
        verbose=100
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=50,
    )
    
    # 모델 저장
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUTPUT_MODEL_PATH))
    print(f"\n[SUCCESS] 모델 학습 완료 및 저장: {OUTPUT_MODEL_PATH}")
    
    # 피처 중요도 출력
    print("\n--- 피처 중요도 ---")
    importance_values = model.get_feature_importance(data=train_pool)
    
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance_values
    }).sort_values('importance', ascending=False)
    
    print(feature_importance_df)

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    print("Reranker 모델 학습 파이프라인 시작")

    REAL_INTERACTION_DATA_PATH: Optional[Path] = None 
    
    # 1. POI 데이터 로드
    poi_df = load_all_poi_data()
    
    # 2. 상호작용 데이터 로드 또는 생성
    interactions_df = load_interaction_data(
        poi_df,
        interaction_file_path=REAL_INTERACTION_DATA_PATH,
        synthetic_sessions=2000
    )
    
    # 3. 피처 엔지니어링
    feature_df, feature_columns = feature_engineering(interactions_df, poi_df)
    
    # 4. 모델 학습 및 저장
    train_model(feature_df, feature_columns)