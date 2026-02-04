
# POI 데이터/임베딩 파이프라인

## 1. 전체 흐름 요약
- **프로세스**: TourAPI/식신 수집 → 정규화 → (AI 태깅/보강) → `embedding_*.json` → 임베딩 생성 → 추천 스코어링
- **핵심 산출물**:
    - `data/embedding_json/embedding_{mode}.json` (메타데이터)
    - `data/embeddings/*.npy` (임베딩 벡터)


## 2. 관광지 데이터 (TourAPI)

### 2-1. 데이터 처리 과정
1. **원천 수집**: TourAPI 서울 관광지 데이터 수집 (`data/processed/poi_merged.json`)
2. **정규화 (Normalization)**
    - **스크립트**: `scripts/poi/normalize_attraction_poi.py`
    - **입력**: `data/processed/poi_merged.json`
    - **출력**: `data/interim/attraction_poi_normalized.json`
    - **주요 필드**: `poi_id`, `type`, `name`, `address`, `location`(lat/lng), `media`, `overview` 등,
3. **AI 태깅 및 요약**
    - **스크립트**: `scripts/poi/generate_poi_ai_analysis.py`
    - **목적**: 임베딩 친화적 가공
    - **생성 필드**:
        - `themes`, `mood`, `visitor_type`, `best_time`
        - `duration`, `activity`, `photospot`, `indoor_outdoor`
        - `keywords`, `summary_one_sentence`, `ideal_schedule_position`

### 2-2. 최종 JSON 구성
- **파일 경로**: `data/embedding_json/embedding_tourspot.json`
- **포함 내용**: 좌표/메타 조회 및 임베딩 입력용 데이터

---

## 3. 식당/카페 데이터 (식신)

### 3-1. 데이터 처리 과정
1. **원천 수집**: 식신 크롤링 데이터 (`rest_data.json`, `cafe_data.json`)
2. **정규화 (Normalization)**
    - **레스토랑 스크립트**: `scripts/poi/normalize_restaurant_poi.py`
    - **카페 스크립트**: `scripts/poi/normalize_cafe_poi.py`
    - **주요 필드**: `poi_id`, `type`, `sub_type`, `name`, `address`, `imglinks`, `description`, `stats`(views, likes 등)
3. **AI 키워드 보강**,
    - **스크립트**: `scripts/poi/generate_food_keywords_ai.py`
    - **특징**: 음식 분류어 제외, **분위기/경험 중심 키워드 5개** 생성

### 3-2. 최종 JSON 구성
- **파일 경로**: `data/embedding_json/embedding_cafe.json`, `embedding_restaurant.json`

---

## 4. 임베딩 생성 (Embedding)

### 4-1. 모델 및 스크립트
- **모델**: `dragonkue/multilingual-e5-small-ko`
- **스크립트**: `scripts/embedding/build_poi_embeddings_npy.py`
- **출력 파일**:
    - `data/embeddings/embeddings_{mode}.npy` (벡터)
    - `data/embeddings/keys_{mode}.npy` (인덱스 키 매핑)

### 4-2. 임베딩 텍스트 구성 규칙 (Vector Input)
임베딩 벡터를 만들 때 아래 필드들을 조합하여 텍스트로 변환합니다.

| 카테고리 | 포함 필드 |
| :--- | :--- |
| **관광지** | `name`, `summary_one_sentence`, `themes`, `mood`, `indoor_outdoor`, `visitor_type`, `keywords`, `activity.label`(없으면 `activity.level`), `best_time`, `ideal_schedule_position`, `photospot`(true일 때만) |
| **음식/카페** | `title`(없으면 `name`), `category`, `content`, `description`, `keywords` |

---

## 5. 데이터 활용 및 외부 지표

### 5-1. 산출물 사용처
- **JSON 파일 (`embedding_{mode}.json`)**: Postgres 메타 적재용 (`scripts/db/ingest_poi_meta.py`)
- **NPY 파일 (`embeddings_{mode}.npy`)**: Milvus 벡터 적재용 (`scripts/milvus/ingest_poi_embeddings.py`)
- **NPY 파일 (`keys_{mode}.npy`)**: `(province, category, place_id)` 키 매핑
  - Milvus 적재 시 place_id 매핑에 사용

### 5-2. 외부 지표 보강
- **Google Places**: 평점 및 리뷰 수 보강 (`add_googleplace_id_*.py`)
- **인기도 점수**: Google 평점 기반 `popularity_score` 계산 (`add_popularity_scores.py`)

---

## 6. 최종 저장 데이터 스키마 (Summary)

모든 최종 JSON(`embedding_{category}.json`)은 아래 구조를 따릅니다.

### 공통 필드
- `place_id`, `poi_id`, `category`, `province`
- `name`
- `city`, `district`, `road`
- `google` (외부 평점), `popularity_score` (계산된 인기도)

### 카테고리별 상세 필드
- **관광지**
  - 위치/주소: `location` (`addr1`, `addr2`, `zipcode`, `lat`, `lng`)
  - TourAPI 원본: `overview`, `intro`, `media`, `contenttypeid`, `poi_type`, `type`, `gu_name`
  - AI 보강: `summary_one_sentence`, `themes`, `mood`, `visitor_type`, `activity`, `best_time`, `best_time_flags`, `keywords`, `indoor_outdoor`, `photospot`, `duration`, `avoid_for`, `ideal_schedule_position`
  - 외부/매칭: `naver`, `naver_match`
- **식당/카페**
  - 위치/주소: `address`, `latitude`, `longitude`
  - 원천/메타: `content`, `description`, `categories`, `links`, `counts`, `likes`, `views`, `bookmarks`, `starts`
  - 이미지: `imglinks` (restaurant는 `images`도 포함)
  - 추가 키워드: `keywords` (AI 보강), `sicksin_keywords` (카페에만 존재)
