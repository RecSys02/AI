# LLM Reranking 비교 로그

## 개요

LLM 기반 reranking의 효과를 측정하기 위해 리랭킹 전후 순위 변화를 추적합니다.

## 로그 방식

### 1. 콘솔 로그 (기본 활성화)

모든 추천 요청에서 자동으로 콘솔에 리랭킹 결과를 출력합니다.

**출력 예시:**
```
[RERANK] TOURSPOT - Reranking Results:
Rank  →  Rank | Change | Place ID |             Name             | Score
-----------------------------------------------------------------------------------------------
   3  ↑     1 |     +2 |      624 | 북촌한옥마을                      | 0.5594
   1  ↓     2 |     -1 |      456 | 경복궁                           | 0.5634
   2  =     3 |      0 |     1044 | 창덕궁                           | 0.5598
   5  ↑     4 |     +1 |      569 | 인사동                           | 0.5574
```

**컬럼 설명:**
- **Rank → Rank**: 리랭킹 전 순위 → 리랭킹 후 순위
- **Change**: 순위 변동 (양수 = 상승 ↑, 음수 = 하락 ↓, 0 = 유지 =)
- **Place ID**: POI 고유 ID
- **Name**: 장소명
- **Score**: 임베딩 기반 원래 점수

### 2. CSV 로그 (선택적)

CSV 파일로 상세한 리랭킹 로그를 저장할 수 있습니다.

**활성화 방법:**

`.env` 파일에 다음 설정 추가:
```bash
RERANK_CSV_LOG=true
```

**저장 위치:**
```
logs/rerank_comparison.csv
```

**CSV 컬럼:**
| 컬럼명 | 설명 |
|--------|------|
| timestamp | 요청 시각 |
| user_id | 사용자 ID |
| category | POI 카테고리 (tourspot/cafe/restaurant) |
| place_id | POI ID |
| name | 장소명 |
| rank_before | 리랭킹 전 순위 |
| rank_after | 리랭킹 후 순위 |
| rank_change | 순위 변동 (양수=상승, 음수=하락) |
| score | 총 점수 |
| score_base | 기본 임베딩 점수 |
| score_recent | 최근 방문 보너스 |
| score_distance | 거리 보너스 |

**CSV 예시:**
```csv
timestamp,user_id,category,place_id,name,rank_before,rank_after,rank_change,score,score_base,score_recent,score_distance
2025-01-16 14:30:15,123,tourspot,624,북촌한옥마을,3,1,2,0.5594,0.52,0.03,0.01
2025-01-16 14:30:15,123,tourspot,456,경복궁,1,2,-1,0.5634,0.53,0.02,0.01
```

## 분석 방법

### 순위 변동 분석

```python
import pandas as pd

# CSV 로드
df = pd.read_csv('logs/rerank_comparison.csv')

# 순위 상승한 항목 (LLM이 더 적합하다고 판단)
promoted = df[df['rank_change'] > 0]
print(f"평균 순위 상승: {promoted['rank_change'].mean():.2f}")

# 순위 하락한 항목
demoted = df[df['rank_change'] < 0]
print(f"평균 순위 하락: {demoted['rank_change'].mean():.2f}")

# 카테고리별 변동 패턴
print(df.groupby('category')['rank_change'].describe())
```

### 사용자별 리랭킹 효과

```python
# 사용자별 평균 순위 변동
user_stats = df.groupby('user_id').agg({
    'rank_change': ['mean', 'std', 'count']
})
print(user_stats)
```

### 점수 vs 순위 변동 상관관계

```python
import matplotlib.pyplot as plt

# 원래 점수가 낮았지만 LLM이 높게 평가한 케이스
plt.scatter(df['score_base'], df['rank_change'])
plt.xlabel('Base Score')
plt.ylabel('Rank Change')
plt.title('LLM Reranking Impact')
plt.show()
```

## 주의사항

1. **콘솔 로그는 항상 출력됨**: 모든 추천 요청에서 자동 출력
2. **CSV 로그는 선택적**: 환경변수로 활성화 필요
3. **로그 파일 크기**: CSV는 계속 누적되므로 주기적으로 정리 필요
4. **성능 영향**: CSV 로깅은 I/O 오버헤드가 있으므로 프로덕션에서는 신중히 사용

## 비활성화 방법

**콘솔 로그 비활성화:** (코드 수정 필요)
- `recommend_service.py`의 `print()` 문을 주석 처리

**CSV 로그 비활성화:**
```bash
RERANK_CSV_LOG=false
```
또는 `.env`에서 해당 라인 삭제
