# Postgres FTS 운영 정리

## 구성
- 대상 DB: Cloud SQL Postgres 15
- 테이블: `poi_meta`
- 컬럼: `search_text` (TEXT), `search_vector` (tsvector)
- 인덱스: `GIN(search_vector)`
- 토크나이저: Kiwi(외부 형태소 분석)
- FTS 설정: `simple` (외부 토큰화 결과를 그대로 사용)

## 인덱싱 흐름
1. `scripts/db/ingest_poi_meta.py`가 embedding JSON을 로드한다.
2. 카테고리별 텍스트를 `build_embedding_text_*`로 만든다.
3. `tokenize_ko()`로 Kiwi 토큰화 → `search_text`에 저장한다.
   - 토큰이 비면 원문 텍스트로 fallback 한다.
4. `search_vector = to_tsvector('simple', search_text)`로 생성한다.
5. `poi_meta`에 upsert한다.

## 조회 흐름
1. `fastapi_app/services/stores/postgres_store.py:fts_scores()`에서 검색을 수행한다.
2. 사용자 쿼리를 `tokenize_ko()`로 토큰화한다.
3. 동의어 테이블을 조회해 토큰을 확장한다(해당 토큰이 있을 때만).
4. `to_tsquery('simple', ...)`로 tsquery를 만든다.
5. `search_vector @@ query.q`로 매칭 후 `ts_rank_cd`로 스코어링한다.

## 검색/랭킹 결합
- `fastapi_app/services/retriever.py`에서
  - Milvus dense 결과
  - Postgres FTS(BM25 유사 스코어) 결과
  - 둘을 `ALPHA_DENSE` 비율로 결합한다.

## 운영 명령
Cloud SQL 프록시가 켜진 상태에서:

```bash
export DATABASE_URL="postgresql://poi_user:<PASSWORD>@127.0.0.1:5432/poi_meta"
python scripts/db/ingest_poi_meta.py --mode tourspot
python scripts/db/ingest_poi_meta.py --mode cafe
python scripts/db/ingest_poi_meta.py --mode restaurant
```

## 튜닝 포인트
- Kiwi 사용자 사전 등록으로 도메인 단어 분절 개선 가능.
- `FTS_POS_TAGS` 목록 조정으로 검색 품질/잡음 균형 조정 가능.
- `simple` 설정은 `to_tsvector`와 `to_tsquery`에서 동일하게 유지해야 한다.
- 동의어 확장은 최대 개수/캐시 TTL로 과확장과 성능 저하를 방지한다.

## 동의어 테이블(쿼리 확장)
`fts_synonym` 테이블을 통해 쿼리 확장을 관리한다.

```sql
create table if not exists fts_synonym (
  id bigserial primary key,
  mode text,
  group_key text not null,
  term text not null,
  enabled boolean not null default true,
  updated_at timestamptz not null default now(),
  unique (mode, group_key, term)
);
create index if not exists fts_synonym_mode_group_idx on fts_synonym (mode, group_key);
create index if not exists fts_synonym_term_idx on fts_synonym (term);
```

예시 입력:

```sql
insert into fts_synonym (mode, group_key, term) values
  ('restaurant', 'sushi', '스시'),
  ('restaurant', 'sushi', '초밥'),
  ('restaurant', 'sushi', '오마카세');
```

적재 스크립트(CSV 기준):

```bash
python scripts/db/seed_fts_synonyms.py --source csv
```

CSV 경로 기본값: `data/fts_synonyms.csv`

## 비고
- `fastapi_app/services/retriever.py`의 `SYNONYMS`는 FTS가 아니라
  키워드 필터링에만 사용한다.
