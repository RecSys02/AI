# 평가용 파이프라인 요약

내일 다시 봐도 이해될 수 있게, 오늘까지 진행한 eval 파이프라인을 단계별로 정리했습니다.

## 1) 유저 데이터 생성

- 스크립트: `scripts/generate_users.py`
- 출력: `data/eval/users_5000.jsonl`

```bash
python scripts/generate_users.py --count 5000 --output data/eval/users_5000.jsonl
```

## 2) 후보 풀 생성 (3km, 카테고리별 최대 50)

- 스크립트: `scripts/generate_candidates.py`
- 출력: `data/eval/candidates_3km.jsonl`

```bash
python scripts/generate_candidates.py --radius-km 3.0 --max-per-mode 50
```

후보 필드:
- restaurant: `food_type`
- cafe: `cafe_type`
- tourspot: `themes`

## 3) 후보 수 기준 유저 필터링

카테고리별 후보 20개 이상인 유저만 유지.

- 스크립트: `scripts/filter_users_by_candidates.py`
- 출력:
  - `data/eval/filtered/users.jsonl`
  - `data/eval/filtered/candidates.jsonl`
  - `data/eval/filtered/recommendations.jsonl`

```bash
python scripts/filter_users_by_candidates.py --k 20
```

## 4) 추천 결과 생성 (retrieval only)

- 스크립트: `scripts/run_recommend_sample.py`
- 출력: `data/eval/recommend_top10_retrieval.jsonl`

```bash
cd fastapi_app
uv run python ../scripts/run_recommend_sample.py \
  --users ../data/eval/filtered/users.jsonl \
  --output ../data/eval/recommend_top10_retrieval.jsonl \
  --limit 5000
```

## 5) 라벨 생성 (Top5)

### OpenAI 라벨링
- 스크립트: `scripts/label_candidates_llm.py`

```bash
uv run ../scripts/label_candidates_llm.py \
  --input ../data/eval/filtered/candidates.jsonl \
  --output ../data/eval/labels_top5_0001.jsonl \
  --top-k 5 --limit 1000 --offset 0
```

### Gemini 라벨링
- 스크립트: `scripts/eval/gemini/label_candidates_gemini.py`
- `.env`에 `GEMINI_API_KEY` 필요

```bash
uv run ../scripts/eval/gemini/label_candidates_gemini.py \
  --input ../data/eval/filtered/candidates.jsonl \
  --output ../data/eval/labels_top5_gemini_0001.jsonl \
  --top-k 5 --limit 1000 --offset 0 --sleep 0.1
```

## 6) 라벨 병합 (중복 제거)

```bash
python scripts/merge_labels.py \
  --inputs data/eval/labels_top5_*.jsonl \
  --output data/eval/labels_top5_all.jsonl
```

## 7) 라벨 누락 유저 추출 (429 대응)

```bash
python scripts/find_missing_labels.py \
  --candidates data/eval/filtered/candidates.jsonl \
  --labels data/eval/labels_top5_all.jsonl \
  --output data/eval/missing_candidates.jsonl
```

## 8) 누락 라벨 재생성

```bash
uv run ../scripts/eval/gemini/label_candidates_gemini.py \
  --input ../data/eval/missing_candidates.jsonl \
  --output ../data/eval/labels_top5_missing_gemini.jsonl \
  --top-k 5 --sleep 0.1
```

## 9) 라벨 최종 병합

```bash
python scripts/merge_labels.py \
  --inputs data/eval/labels_top5_all.jsonl data/eval/labels_top5_missing_gemini.jsonl \
  --output data/eval/labels_top5_all.jsonl \
  --prefer last
```

## 10) 평가 (Recall/NDCG)

- 스크립트: `scripts/eval/compute_metrics.py`

```bash
python scripts/eval/compute_metrics.py \
  --labels data/eval/labels_top5_all.jsonl \
  --pred data/eval/filtered/recommendations.jsonl \
  --k 10
```

## 11) Gemini rerank 비교 (옵션)

- `.env` 설정:
```
RERANK_PROVIDER=gemini
GEMINI_API_KEY=...
```

```bash
uv run python ../scripts/run_recommend_sample.py \
  --users ../data/eval/filtered/users.jsonl \
  --output ../data/eval/recommend_top10_rerank_gemini.jsonl \
  --limit 5000 \
  --rerank
```

```bash
python scripts/eval/compute_metrics.py \
  --labels data/eval/labels_top5_all.jsonl \
  --pred data/eval/recommend_top10_retrieval.jsonl \
  --pred data/eval/recommend_top10_rerank_gemini.jsonl \
  --k 10
```

---

### 중요한 수정사항
- 거리 계산은 **마지막 선택 장소의 좌표** 기준
- 카테고리 충돌 fallback 제거
- cafe는 `cafe_type`, tourspot은 `themes`
