# 추천 모델 성능 평가 보고서 (초안)

## 1. 목적
- Retrieval vs LLM rerank 정확도(Recall/NDCG) 비교
- Gemini 2.0 vs 3.0 rerank 속도/토큰 비교

## 2. 데이터셋 구성
- 사용자: `data/eval/users_5000.jsonl`
- 후보: `data/eval/candidates_3km.jsonl`
- 라벨: `data/eval/labels_top5_all.jsonl` (5천 유저 기준)

## 3. 파이프라인 요약
1) 유저 생성 → 2) 후보 생성(3km, 카테고리별 최대 50) → 3) 후보 부족 유저 필터링  
4) 추천 결과 생성(retrieval only) → 5) LLM 라벨링(Top5)  
6) 라벨 병합/누락 보완 → 7) Recall/NDCG 계산

## 4. 실험 설정
- top_k: 10
- distance_max_km: 3.0 (후보 부족 시 확장: `RECOMMEND_EXPAND_STEP_KM`, `RECOMMEND_EXPAND_MAX_KM`)
- rerank: 기본 on (후보가 top_k 초과 시 LLM rerank 수행)
- Gemini 비교는 비용/시간 이슈로 샘플 100명 기준 수행

## 5. 실행 명령어

### 5.1 추천 결과 생성 (retrieval only)
```bash
uv run python scripts/eval/run_recommend_sample.py \
  --users data/eval/filtered/users.jsonl \
  --output data/eval/recommend_top10_retrieval.jsonl \
  --limit 5000
```

### 5.2 Gemini rerank 샘플(100명) 비교
```bash
RERANK_PROVIDER=gemini GEMINI_RERANK_MODEL=gemini-2.0-flash \
uv run python scripts/eval/run_recommend_sample.py \
  --users data/eval/samples/users_100_from_labels.jsonl \
  --output data/eval/recommend_top10_rerank_gemini2_sample100.jsonl \
  --limit 100 --rerank

RERANK_PROVIDER=gemini GEMINI_RERANK_MODEL=gemini-3-flash-preview \
uv run python scripts/eval/run_recommend_sample.py \
  --users data/eval/samples/users_100_from_labels.jsonl \
  --output data/eval/recommend_top10_rerank_gemini3_sample100.jsonl \
  --limit 100 --rerank
```

### 5.3 평가 지표 계산 (Recall/NDCG)
```bash
python scripts/eval/compute_metrics.py \
  --labels data/eval/labels_top5_all.jsonl \
  --pred data/eval/recommend_top10_retrieval.jsonl \
  --k 10
```

## 6. 결과 기록 (채워 넣기)

### 6.1 정확도 (전체 라벨 교집합 기준)
- restaurant: Recall@10 = [ ], NDCG@10 = [ ]
- cafe: Recall@10 = [ ], NDCG@10 = [ ]
- tourspot: Recall@10 = [ ], NDCG@10 = [ ]

### 6.2 Rerank 속도/토큰 (샘플 100명 × 10회 평균)
- Gemini 2.0: latency_ms 평균 [ ], total_tokens 평균 [ ]
- Gemini 3.0: latency_ms 평균 [ ], total_tokens 평균 [ ]

## 7. 로그 위치
- rerank 비교 로그: `logs/rerank_comparison.csv` (옵션: `RERANK_CSV_LOG=true`)
- rerank 메트릭 로그: `logs/rerank_metrics.csv` (옵션: `RERANK_METRICS_LOG=true`)

## 8. 해석/결론 (초안)
- [ ] 2.0과 3.0의 속도/비용 트레이드오프 요약
- [ ] rerank가 retrieval 대비 정확도 개선에 미치는 영향
- [ ] 운영 환경에서의 권장 모델/전략

