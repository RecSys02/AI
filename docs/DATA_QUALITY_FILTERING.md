# 데이터 품질 필터링

## 개요

크롤링 오류나 데이터 오염으로 인한 잘못된 POI 데이터를 자동으로 제외하는 규칙 기반 필터링 시스템입니다.

**2단계 필터링**:
1. **임베딩 생성 시 필터링**: 잘못된 데이터가 임베딩 파일에 포함되지 않도록 사전 차단
2. **추천 시스템 런타임 필터링**: topk 선정 과정에서 실시간으로 유효성 검증

## 문제 사례

### 잘못된 데이터 예시

```json
{
  "place_id": 887,
  "category": "restaurant",
  "province": "seoul",
  "name": "내 위치",  // ❌ 금칙어
  "description": "가볼만한 아담한 퓨전포차♥",
  "address": "서울특별시 서초구 양재2동 279-9",
  "content": "한국음식 > 닭구이/찜/탕/갈비",
  "keywords": ["아담한 공간", "친근한 분위기"]
}
```

이런 데이터가 임베딩되면:
- 높은 순위로 추천될 수 있음
- 사용자 경험 저하
- 시스템 신뢰도 하락

## 필터링 규칙

### 1. 필수 필드 검증

모든 POI는 다음 필드가 필수입니다:

```python
# 필수 필드
- place_id: POI 고유 ID
- name 또는 title: 장소명
```

**제외 사유**:
- `place_id 누락`
- `이름 누락`

### 2. 금칙어 검사 (이름)

장소명에 다음 단어가 포함되면 제외됩니다:

```python
forbidden_names = [
    "내 위치",
    "현재 위치",
    "unknown",
    "test",
    "테스트",
    "샘플",
]
```

**제외 사유**: `금칙어 포함: '내 위치'`

### 3. 카페/레스토랑 전용 검증

#### 3-1. content 필드 필수

```python
if mode in ["cafe", "restaurant"]:
    if not poi.get("content"):
        # 제외
```

**제외 사유**: `content 필드 누락`

#### 3-2. 유효하지 않은 content 값

```python
invalid_contents = [
    "평가중",
    "정보없음",
    "unknown",
    "n/a",
    ".",
    "-",
]
```

**제외 사유**: `유효하지 않은 content: '평가중'`

#### 3-3. content 길이 검증

```python
if len(content) < 2:
    # 제외
```

**제외 사유**: `content 너무 짧음: '.'`

#### 3-4. 카테고리 키워드 검증

content 필드에 해당 카테고리 관련 키워드가 최소 1개 이상 있어야 합니다.

**레스토랑 키워드**:
```python
["음식", "식당", "요리", "맛집", "한식", "중식", "일식", "양식", "세계음식"]
```

**카페 키워드**:
```python
["카페", "커피", "디저트", "베이커리"]
```

**제외 사유**: `content에 'restaurant' 관련 키워드 없음: '기타 > 잡화'`

## 필터링 방식

### 1단계: 임베딩 생성 시 필터링

**위치**: [build_poi_embeddings_npy.py:37-114, 182-207](../scripts/embedding/build_poi_embeddings_npy.py#L37-L114)

임베딩 파일 생성 전에 원본 JSON 데이터를 검증하여 유효한 POI만 임베딩합니다.

**사용 방법**:
```bash
# 레스토랑 임베딩 생성
python scripts/embedding/build_poi_embeddings_npy.py --mode restaurant

# 카페 임베딩 생성
python scripts/embedding/build_poi_embeddings_npy.py --mode cafe

# 관광지 임베딩 생성
python scripts/embedding/build_poi_embeddings_npy.py --mode tourspot
```

**콘솔 출력 예시**:
```
[INFO] mode=restaurant input=data/embedding_json/embedding_restaurant.json count=1523

[FILTER] 데이터 유효성 검사 시작...
  ❌ [제외] place_id=887, name='내 위치', reason='금칙어 포함: '내 위치''
  ❌ [제외] place_id=1024, name='테스트 식당', reason='금칙어 포함: '테스트''
  ❌ [제외] place_id=1155, name='맛집', reason='content 필드 누락'
  ❌ [제외] place_id=1203, name='카페 ABC', reason='content에 'restaurant' 관련 키워드 없음: '카페 > 디저트''

[FILTER] 필터링 완료:
  - 원본 데이터: 1523개
  - 유효한 데이터: 1519개
  - 제외된 데이터: 4개

[INFO] device=cuda
✅ Saved embeddings: data/embeddings/embeddings_restaurant.npy shape=(1519, 1024)
✅ Saved poi_keys: data/embeddings/keys_restaurant.npy shape=(1519,)
```

### 2단계: 런타임 필터링 (topk 선정 시)

**위치**: [base.py:14-81, 251-300](../fastapi_app/services/scorers/base.py#L14-L81)

추천 시스템이 topk 후보를 선정할 때 실시간으로 메타데이터를 검증합니다. 이는 임베딩 생성 이후에 추가된 데이터나 업데이트된 메타데이터를 검증하는 안전장치입니다.

**동작 방식**:
1. 점수 기반으로 정렬된 후보 중 상위부터 순회
2. 각 POI의 메타데이터를 `_is_valid_poi_meta()` 함수로 검증
3. 유효하지 않은 POI는 제외하고 다음 후보 확인
4. 요청한 개수(top_k)만큼 유효한 POI를 모을 때까지 계속 확인 (최대 top_k × 3개까지)

**콘솔 출력 예시**:
```
[SCORER] Filtered out: place_id=887, name='내 위치', content='한국음식 > 닭구이/찜/탕/갈비'
[SCORER] Filtered out: place_id=1024, name='테스트 식당', content='한국음식 > 한정식'
[SCORER] restaurant: Checked 12 items, filtered out 2, returned 10
```

**특징**:
- 임베딩 파일을 재생성하지 않아도 즉시 적용
- FastAPI 서버 재시작 시 자동 적용
- 오버헤드 최소화: 유효한 후보 10개를 찾기 위해 최대 30개만 확인

## 필터링 로직 위치

### 임베딩 생성 시 필터링

**파일**: [build_poi_embeddings_npy.py](../scripts/embedding/build_poi_embeddings_npy.py)

**함수**: `is_valid_poi(poi: Dict[str, Any], mode: str) -> Tuple[bool, str]`

**라인**: 37-114

### 런타임 필터링

**파일**: [base.py](../fastapi_app/services/scorers/base.py)

**함수**: `_is_valid_poi_meta(meta: dict, category: str) -> bool`

**라인**: 14-81

**적용 위치**: `EmbeddingScorer.topk()` 메서드 내부 (251-300)

## 필터링 규칙 수정 방법

### 금칙어 추가

**임베딩 생성 시**: [build_poi_embeddings_npy.py:58-65](../scripts/embedding/build_poi_embeddings_npy.py#L58-L65)
**런타임**: [base.py:33-40](../fastapi_app/services/scorers/base.py#L33-L40)

두 파일 모두 수정해야 완전한 필터링이 적용됩니다.

```python
forbidden_names = [
    "내 위치",
    "현재 위치",
    "unknown",
    "test",
    "테스트",
    "샘플",
    # 새로운 금칙어 추가
    "임시",
    "temp",
]
```

### 카테고리 키워드 확장

**임베딩 생성 시**: [build_poi_embeddings_npy.py:97-100](../scripts/embedding/build_poi_embeddings_npy.py#L97-L100)
**런타임**: [base.py:70-73](../fastapi_app/services/scorers/base.py#L70-L73)

두 파일 모두 수정해야 완전한 필터링이 적용됩니다.

```python
expected_keywords = {
    "cafe": ["카페", "커피", "디저트", "베이커리", "차", "음료"],  # 추가
    "restaurant": ["음식", "식당", "요리", "맛집", "한식", "중식", "일식", "양식", "세계음식", "전통음식"],  # 추가
}
```

## 필터링 통계 확인

필터링 통계는 임베딩 생성 시 자동으로 출력됩니다:

```
[FILTER] 필터링 완료:
  - 원본 데이터: 1523개
  - 유효한 데이터: 1519개
  - 제외된 데이터: 4개
```

## 제외된 데이터 검토

제외된 데이터를 검토하려면 콘솔 출력을 파일로 저장:

```bash
python scripts/embedding/build_poi_embeddings_npy.py --mode restaurant > embedding_log.txt 2>&1
```

그 후 `embedding_log.txt`에서 `[제외]` 키워드로 검색하여 확인할 수 있습니다.

## 장점 및 주의사항

### 2단계 필터링의 장점

1. **이중 안전장치**: 임베딩 생성 시 + 런타임에서 모두 필터링
2. **즉시 적용**: 런타임 필터링은 서버 재시작만으로 적용 (임베딩 재생성 불필요)
3. **유연성**: 임베딩 파일을 다시 만들지 않고도 필터링 규칙 수정 가능
4. **효율성**: 잘못된 데이터가 임베딩되지 않아 저장 공간 절약

### 주의사항

1. **원본 JSON 파일은 수정되지 않음**: 필터링은 임베딩/추천 과정에서만 적용되며, 원본 데이터는 그대로 유지됩니다.

2. **기존 임베딩 파일 덮어쓰기**: 동일한 모드로 재생성 시 기존 `.npy` 파일이 덮어써집니다.

3. **필터링 후 POI 개수 변경**: 필터링으로 인해 임베딩 파일의 POI 개수가 원본 JSON보다 적을 수 있습니다.

4. **두 파일 동기화 필요**: 필터링 규칙을 수정할 때는 `build_poi_embeddings_npy.py`와 `base.py` 모두 수정해야 합니다.

5. **런타임 오버헤드**: topk 선정 시 유효성 검증으로 약간의 오버헤드가 있지만, 최대 30개(top_k×3)만 확인하므로 미미합니다.

## 예외 처리

### 모든 POI가 제외된 경우

```python
if not valid_pois:
    print("[ERROR] 유효한 POI가 없습니다. 임베딩을 생성할 수 없습니다.")
    return
```

이 경우 임베딩 파일이 생성되지 않으며, 원본 데이터를 검토해야 합니다.

## 확장 가능성

### 추가 검증 규칙 예시

```python
# 4. 주소 검증
if not poi.get("address") or len(poi["address"]) < 5:
    return False, "유효하지 않은 주소"

# 5. 설명 최소 길이
description = poi.get("description", "")
if len(description) < 10:
    return False, "설명이 너무 짧음"

# 6. 키워드 개수
keywords = poi.get("keywords", [])
if len(keywords) < 2:
    return False, "키워드가 너무 적음"
```

이런 규칙들을 `is_valid_poi()` 함수에 추가할 수 있습니다.
