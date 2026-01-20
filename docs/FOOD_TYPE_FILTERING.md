# 음식 종류 필터링

## 개요

사용자가 "한식"을 선호한다고 입력했는데 "중식" 음식점이 추천되는 문제를 해결하기 위한 필터링 기능입니다.

## 동작 방식

### 1단계: 파이썬 사전 필터링

LLM reranking 전에 사용자 선호 음식 종류와 맞지 않는 후보를 제거합니다.

**위치**: [recommend_service.py:71-133](../fastapi_app/services/recommend_service.py#L71-L133)

**필터링 로직 (포함 방식)**:
```python
# 사용자가 선호하는 음식 종류 추출
if user.preferred_restaurant_types = ["양식"]:
    # "양식" 또는 "서양음식" 키워드가 포함된 것만 허용
    "세계음식 > 양식" → 포함 ✅ ("양식" 키워드 포함)
    "한국음식 > 냉면" → 제외 ❌ ("양식" 키워드 없음)
    "세계음식 > 중식" → 제외 ❌ ("양식" 키워드 없음)
    "세계음식 > 일식" → 제외 ❌ ("양식" 키워드 없음)
    "퓨전음식 > 브런치" → 제외 ❌ ("양식" 키워드 없음)
```

**핵심 원리**:
- 사용자가 "양식"을 선호하면 → "양식", "서양음식" 키워드가 **포함된** 것만 허용
- **포함(inclusive)** 필터링: 정확히 선호한 음식 종류만

**음식 종류 매핑**:
| 사용자 입력 | 매칭 키워드 |
|------------|-----------|
| 한식, 한국음식 | 한국음식, 한식 |
| 중식, 중국음식 | 중식, 중국음식 |
| 일식, 일본음식 | 일식, 일본음식 |
| 양식, 서양음식 | 양식, 서양음식 |

### 2단계: LLM 재정렬

1단계 필터링 후 남은 후보를 LLM이 사용자 맥락에 맞게 재정렬합니다.

**프롬프트 지침**:
```
**중요: 음식 종류 필터링 규칙**
1. 사용자가 "한식", "한국음식" 등을 선호한다면,
   food_type이 "중식", "일식", "양식" 등 다른 음식 종류인 후보는
   반드시 순위를 낮춰야 합니다.
2. 사용자의 선호 음식 종류와 일치하는 후보를 우선적으로 선택하세요.
3. food_type 필드가 있는 경우, 이를 가장 중요한 판단 기준으로 삼으세요.
```

## 필터링 예시

### 케이스 1: 양식 선호 사용자

**입력**:
```json
{
  "user_id": 123,
  "preferred_restaurant_types": ["양식"],
  "region": "seoul"
}
```

**후보 (top-15)**:
```
1. 이탈리안 레스토랑 (세계음식 > 양식) - 점수: 0.85 ✅ "양식" 포함
2. 마라탕 전문점 (세계음식 > 중식) - 점수: 0.82 ❌ "양식" 없음
3. 프렌치 비스트로 (세계음식 > 양식) - 점수: 0.80 ✅ "양식" 포함
4. 스시야 (세계음식 > 일식) - 점수: 0.78 ❌ "양식" 없음
5. 토속촌 (한국음식 > 삼계탕) - 점수: 0.75 ❌ "양식" 없음
6. 스테이크하우스 (세계음식 > 양식) - 점수: 0.73 ✅ "양식" 포함
```

**필터링 후 (양식만)**:
```
1. 이탈리안 레스토랑 (세계음식 > 양식) - 점수: 0.85
2. 프렌치 비스트로 (세계음식 > 양식) - 점수: 0.80
3. 스테이크하우스 (세계음식 > 양식) - 점수: 0.73
...
```

**콘솔 로그**:
```
[RERANK] Filtered out: 마라탕 전문점 (food_type: 세계음식 > 중식)
[RERANK] Filtered out: 스시야 (food_type: 세계음식 > 일식)
[RERANK] Filtered out: 토속촌 (food_type: 한국음식 > 삼계탕)
```

### 케이스 2: 선호도 없는 사용자

**입력**:
```json
{
  "user_id": 456,
  "preferred_restaurant_types": null,
  "region": "seoul"
}
```

**동작**: 필터링 없이 LLM reranking만 수행

## 안전장치

### 1. 과도한 필터링 방지

```python
if len(filtered_candidates) >= top_k:
    candidates = filtered_candidates
else:
    # 필터 후 top_k보다 적으면 원본 유지
    candidates = candidates
```

필터링 결과가 요청한 개수(top_k, 기본 10개)보다 적으면 필터링을 적용하지 않습니다.

### 2. content 필드 없는 경우

```python
if not content:
    # content가 없으면 포함 (필터링 안함)
    filtered.append(poi)
```

메타데이터에 `content` 필드가 없는 POI는 필터링하지 않고 포함시킵니다.

## 데이터 구조

### embedding_restaurant.json

```json
{
  "place_id": 1,
  "name": "을밀대 평양냉면",
  "content": "한국음식 > 냉면",  // 이 필드를 사용
  "description": "...",
  "keywords": ["전통의 맛", "..."]
}
```

### POI 내부 구조 (_meta)

```python
{
  "place_id": 1,
  "category": "restaurant",
  "score": 0.85,
  "_meta": {
    "name": "을밀대 평양냉면",
    "content": "한국음식 > 냉면",  // 필터링에 사용
    "description": "...",
    "keywords": [...]
  }
}
```

## 확장 가능성

### 더 많은 음식 종류 추가

[recommend_service.py:94-103](../fastapi_app/services/recommend_service.py#L94-L103)의 `type_mapping` 딕셔너리에 추가:

```python
type_mapping = {
    "한식": ["한국음식", "한식"],
    "중식": ["중식", "중국음식"],
    "일식": ["일식", "일본음식"],
    "양식": ["양식", "서양음식"],
    # 새로운 종류 추가
    "태국음식": ["태국음식", "타이"],
    "베트남음식": ["베트남음식", "쌀국수"],
}
```

### 카페 음료 종류 필터링

동일한 로직이 `cafe` 카테고리에도 적용됩니다:
- `preferred_cafe_types`: ["커피전문점", "디저트카페", ...]
- `content` 필드 예시: "카페 > 커피전문점", "카페 > 베이커리"

## 디버깅

필터링된 항목은 콘솔에 자동 출력됩니다:

```bash
[RERANK] Filtered out: 마라탕 전문점 (food_type: 세계음식 > 중식)
[RERANK] Filtered out: 스시야 (food_type: 세계음식 > 일식)
```

필터링 통계를 확인하려면:
```python
print(f"[RERANK] Filtered: {len(candidates) - len(filtered)} out of {len(candidates)}")
```
