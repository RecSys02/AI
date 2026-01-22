# 평가 데이터셋 파이프라인

이 문서는 추천 평가용 데이터셋 생성 흐름을 정리합니다.

## 1) 유저 데이터 생성

- 스크립트: `scripts/generate_users.py`
- 출력: `data/eval/users_5000.jsonl`

예시:
```bash
python scripts/generate_users.py --count 5000 --output data/eval/users_5000.jsonl
```

## 2) 후보 풀 생성 (반경 3km, 카테고리별 최대 50개)

- 스크립트: `scripts/generate_candidates.py`
- 출력: `data/eval/candidates_3km.jsonl`

예시:
```bash
python scripts/generate_candidates.py --radius-km 3.0 --max-per-mode 50
```

후보 필드(아이템 단위):
- `id`, `category`, `name`, `distance_km`, `keywords`
- 음식점: `food_type`
- 카페: `cafe_type`
- 관광지: `themes`

## 3) 후보 수 기준 유저 필터링

카테고리별 후보 개수가 충분한 유저만 남깁니다.

- 스크립트: `scripts/filter_users_by_candidates.py`
- 출력 디렉터리: `data/eval/filtered/`
  - `users.jsonl`
  - `candidates.jsonl`
  - `recommendations.jsonl`

예시 (카테고리별 최소 20개):
```bash
python scripts/filter_users_by_candidates.py --k 20
```

## 4) 추천 결과 생성 (Top10, LLM rerank 제외)

- 스크립트: `scripts/run_recommend_sample.py`
- 출력: `data/eval/recommend_top10_5000.jsonl`

예시:
```bash
cd fastapi_app
uv run python ../scripts/run_recommend_sample.py \
  --limit 5000 \
  --output ../data/eval/recommend_top10_5000.jsonl
```

## 5) LLM 라벨링 (Top5)

필터링된 후보를 대상으로 LLM이 Top5 정답을 선택합니다.

- 스크립트: `scripts/label_candidates_llm.py`
- 출력: `data/eval/labels_top5.jsonl`

예시:
```bash
uv run ../scripts/label_candidates_llm.py \
  --input ../data/eval/filtered/candidates.jsonl \
  --output ../data/eval/labels_top5.jsonl \
  --top-k 5 \
  --model gpt-4o-mini
```

Gemini 사용 (별도 스크립트):
- 스크립트: `scripts/eval/gemini/label_candidates_gemini.py`
- 입력/출력은 파일 내부 변수(`INPUT_FILE`, `OUTPUT_FILE`)로 지정

## 6) 평가 (다음 단계)

예측 결과와 라벨을 비교해 지표를 계산합니다.

- 예측: `data/eval/filtered/recommendations.jsonl`
- 라벨: `data/eval/labels_top5.jsonl`
- 지표: `Recall@10`, `NDCG@10`

---

참고:
- 라벨링 비용 절감을 위해 `--offset`/`--limit`로 배치 실행을 권장합니다.
- `fastapi_app` 폴더에서 실행 시 경로는 `../`를 사용하세요.
