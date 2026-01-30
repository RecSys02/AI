# POI 데이터/임베딩 파이프라인

## 0) 전체 흐름 요약
- TourAPI/식신 수집 -> 정규화 -> (AI 태깅/보강) -> `embedding_*.json` -> 임베딩 생성 -> 추천 스코어링
- 핵심 산출물은 `data/embedding_json/embedding_{mode}.json`와 `data/embeddings/*.npy`

## 1) 관광지 데이터 (TourAPI)
### 1-1. 원천 수집
- TourAPI로 서울 관광지 데이터를 수집
- 내부적으로 `data/processed/poi_merged.json` 형태로 모은 뒤 정규화에 사용

### 1-2. 정규화 (필드 축약/일관화)
- 스크립트: `scripts/poi/normalize_attraction_poi.py`
- 입력: `data/processed/poi_merged.json`
- 출력: `data/interim/attraction_poi_normalized.json`
- 주요 매핑/필드
  - 기본: `poi_id`, `type`, `sub_type`, `name`, `address`, `gu_name`
  - 좌표: `location.lat`, `location.lng`
  - 이미지: `media.firstimage`
  - 설명: `overview`
  - 통계: `naver.naver_rating`, `naver.naver_visitor_reviews`
  - 속성 묶음(`attributes`)
    - `duration`, `activity`, `photospot`, `indoor_outdoor`
    - `keywords`, `summary`, `themes`, `mood`, `visitor_type`, `best_time`
    - `parking` (가능한 경우에만)

### 1-3. AI 태깅/요약(임베딩 친화 가공)
- 스크립트: `scripts/poi/generate_poi_ai_analysis.py`
- 입력: `data/interim/poi_inter.jsonl` (정규화 결과를 집계한 intermediate)
- 출력: `data/processed/poi_analysis.jsonl`
- 생성 필드
  - `themes`, `mood`, `visitor_type`, `best_time`, `best_time_flags`
  - `duration`, `activity`(label/level), `photospot`, `indoor_outdoor`
  - `keywords`, `summary_one_sentence`, `avoid_for`, `ideal_schedule_position`
- 네이버 평점/리뷰는 참고용으로만 사용하며 출력 JSON에는 직접 포함하지 않음

### 1-4. embedding JSON 구성
- 최종 결과는 `data/embedding_json/embedding_tourspot.json`에 통합
- 이 파일은 좌표/메타 조회 및 임베딩 입력용으로 사용됨

## 2) 식당/카페 데이터 (식신)
### 2-1. 원천 수집
- 식신 데이터 크롤링
- 입력 예시
  - `data/raw/rest_data.json`
  - `data/raw/cafe_data.json`

### 2-2. 정규화 (필드 축약/일관화)
- 레스토랑
  - 스크립트: `scripts/poi/normalize_restaurant_poi.py`
  - 출력: `data/interim/restaurant_poi_normalized.json`
- 카페
  - 스크립트: `scripts/poi/normalize_cafe_poi.py`
  - 출력: `data/interim/cafe_poi_normalized.json`
- 주요 매핑/필드
  - 기본: `poi_id`, `type`, `sub_type`(카페는 content에서 추출), `name`, `address`
  - 좌표: `location.lat`, `location.lng`
  - 이미지: `imglinks`
  - 설명: `description`
  - 통계: `views`, `likes`, `bookmarks`, `starts`(rating), `counts`(review_count)

### 2-3. (선택) AI 키워드 보강
- 스크립트: `scripts/poi/generate_food_keywords_ai.py`
- 입력: `data/embedding_json/embedding_{category}.json`
- 출력: `data/processed/*_ai_keywords.json`
- 음식 분류어는 제외하고, 분위기/경험 중심 키워드 5개를 생성

### 2-4. embedding JSON 구성
- 최종 결과는 `data/embedding_json/embedding_cafe.json`, `embedding_restaurant.json`에 통합
- 이 파일은 좌표/메타 조회 및 임베딩 입력용으로 사용됨

## 3) 임베딩 생성
### 3-1. 임베딩 모델
- 모델: `BAAI/bge-m3`

### 3-2. 임베딩 생성 스크립트
- 스크립트: `scripts/embedding/build_poi_embeddings_npy.py`
- 입력: `data/embedding_json/embedding_{mode}.json`
- 출력:
  - `data/embeddings/embeddings_{mode}.npy`
  - `data/embeddings/keys_{mode}.npy`

### 3-3. 임베딩 텍스트 구성 규칙
- 관광지(`build_embedding_text_tourspot`)
  - `name`, `summary_one_sentence`, `themes`, `mood`, `visitor_type`, `best_time`
  - `duration`, `activity.level`, `indoor_outdoor`, `photospot`
  - `keywords`, `avoid_for`, `ideal_schedule_position`
- 음식/카페(`build_embedding_text_food`)
  - `name/title`, `category`, `content`, `description`, `keywords`

## 4) 데이터 아티팩트 사용처
- `data/embedding_json/embedding_{mode}.json`
  - 좌표/메타 로딩용
  - 추천에서 `coords_path`로 사용 (`fastapi_app/services/scorers/base.py`)
- `data/embeddings/embeddings_{mode}.npy`
  - 벡터 유사도 계산용
- `data/embeddings/keys_{mode}.npy`
  - `(province, category, place_id)` 키 매핑
  - 추천 스코어링에서 place_id 인덱싱에 사용

## 5)외부 지표 보강
- Google Places로 평점/리뷰 수 보강
  - `scripts/poi/add_googleplace/add_googleplace_id_tourspot.py`
  - `scripts/poi/add_googleplace/add_googleplace_id_cafe_restaurant.py`
- Google 평점 기반 인지도 점수 계산
  - `scripts/poi/add_popularity_scores.py`
