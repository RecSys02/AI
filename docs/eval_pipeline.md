# 평가(Eval) 파이프라인 문서

추천 결과의 품질(Recall@K, NDCG@K)을 측정하기 위한 데이터 생성/라벨링/평가 흐름을 정리합니다.

## 0) 준비 사항
- **입력 데이터**: `data/embedding_json/embedding_{mode}.json` (tourspot/cafe/restaurant)
- **환경 변수**
  - OpenAI 라벨링: `OPENAI_API_KEY`
  - Gemini 라벨링: `GEMINI_API_KEY` 또는 `GOOGLE_API_KEY`
  - Rerank 평가: `RERANK_PROVIDER=gemini` (옵션), `GEMINI_RERANK_MODEL`

## 1) 유저 샘플 생성
- 스크립트: `scripts/eval/generate_users.py`
- 출력: `data/eval/users_5000.jsonl`

```bash
python scripts/eval/generate_users.py \
  --count 5000 \
  --output data/eval/users_5000.jsonl
```

## 2) 후보 풀 생성 (거리 기반)
- 스크립트: `scripts/eval/generate_candidates.py`
- 출력: `data/eval/candidates_3km.jsonl`
- 기준: 유저의 마지막 선택 장소 좌표(anchor)에서 반경 `radius_km` 이내 후보

```bash
python scripts/eval/generate_candidates.py \
  --users data/eval/users_5000.jsonl \
  --output data/eval/candidates_3km.jsonl \
  --radius-km 3.0 \
  --max-per-mode 50
```

후보 필드(카테고리별):
- restaurant: `food_type`
- cafe: `cafe_type`
- tourspot: `themes`

## 3) 추천 결과 생성 (retrieval / rerank)
- 스크립트: `scripts/eval/run_recommend_sample.py`
- 출력: `data/eval/recommend_top10_retrieval.jsonl`

```bash
python scripts/eval/run_recommend_sample.py \
  --users data/eval/users_5000.jsonl \
  --output data/eval/recommend_top10_retrieval.jsonl \
  --limit 5000
```

LLM rerank 비교(옵션):
```bash
export RERANK_PROVIDER=gemini
export GEMINI_API_KEY=...
python scripts/eval/run_recommend_sample.py \
  --users data/eval/users_5000.jsonl \
  --output data/eval/recommend_top10_rerank_gemini.jsonl \
  --limit 5000 \
  --rerank
```

## 4) 후보 수 기준 유저 필터링
- 스크립트: `scripts/eval/filter_users_by_candidates.py`
- 목적: 카테고리별 최소 후보 수 보장
- 출력: `data/eval/filtered_*/*.jsonl`

```bash
python scripts/eval/filter_users_by_candidates.py \
  --candidates data/eval/candidates_3km.jsonl \
  --users data/eval/users_5000.jsonl \
  --recommendations data/eval/recommend_top10_retrieval.jsonl \
  --output-dir data/eval/filtered_retrieval \
  --k 20
```

## 5) 라벨 생성 (Top-K)
### 5-1. OpenAI 라벨링
- 스크립트: `scripts/eval/label_candidates_llm.py`

```bash
python scripts/eval/label_candidates_llm.py \
  --input data/eval/filtered_retrieval/candidates.jsonl \
  --output data/eval/labels_top5_0001.jsonl \
  --top-k 5 --limit 1000 --offset 0
```

### 5-2. Gemini 라벨링(대체/추가)
- 스크립트: `scripts/eval/gemini/label_candidates_gemini.py`

```bash
python scripts/eval/gemini/label_candidates_gemini.py \
  --input data/eval/filtered_retrieval/candidates.jsonl \
  --output data/eval/labels_top5_gemini_0001.jsonl \
  --top-k 5 --limit 1000 --offset 0 --sleep 0.1
```

## 6) 라벨 병합 및 누락 대응
```bash
python scripts/merge_labels.py \
  --inputs data/eval/labels_top5_*.jsonl \
  --output data/eval/labels_top5_all.jsonl
```

누락 라벨 추출:
```bash
python scripts/find_missing_labels.py \
  --candidates data/eval/filtered_retrieval/candidates.jsonl \
  --labels data/eval/labels_top5_all.jsonl \
  --output data/eval/missing_candidates.jsonl
```

누락 라벨 재생성(예: Gemini):
```bash
python scripts/eval/gemini/label_candidates_gemini.py \
  --input data/eval/missing_candidates.jsonl \
  --output data/eval/labels_top5_missing_gemini.jsonl \
  --top-k 5 --sleep 0.1
```

최종 병합:
```bash
python scripts/merge_labels.py \
  --inputs data/eval/labels_top5_all.jsonl data/eval/labels_top5_missing_gemini.jsonl \
  --output data/eval/labels_top5_all.jsonl \
  --prefer last
```

## 7) 평가 지표 계산
- 스크립트: `scripts/eval/compute_metrics.py`
- 지표: Recall@K, NDCG@K (카테고리별)

```bash
python scripts/eval/compute_metrics.py \
  --labels data/eval/labels_top5_all.jsonl \
  --pred data/eval/filtered_retrieval/recommendations.jsonl \
  --k 10
```

여러 예측 비교:
```bash
python scripts/eval/compute_metrics.py \
  --labels data/eval/labels_top5_all.jsonl \
  --pred data/eval/filtered_retrieval/recommendations.jsonl \
  --pred data/eval/recommend_top10_rerank_gemini.jsonl \
  --k 10
```

## 8) 산출물 형식
- **labels JSONL**
  - `{ "userId": 1, "category": "restaurant", "relevant_ids": [101, 202] }`
- **recommendations JSONL**
  - `{ "userId": 1, "recommendations": [ { "category": "cafe", "items": [...] }, ... ] }`

## 9) 주의사항
- 라벨링은 API 호출 비용/속도 제약이 있으므로 `limit/offset/sleep`로 분할 실행 권장
- rerank 비교는 `RERANK_PROVIDER` 설정에 따라 결과가 달라질 수 있음
