# 작업 로그

## 2026-01-28
### 목표
- 문맥 기반 쿼리 리라이팅과 의도 라우팅 개선
- admin-term fallback 제거 및 앵커 실패 시 위치 재질문
- avoid 옵션화
- 채팅 LLM을 Gemini로 전환

### 변경 사항
- 그래프 흐름: `route` 앞에 `rewrite_query` 추가, intent는 `normalized_query` 기준 판단.
- 리라이팅: 최근 3개 메시지 반영, `last_normalized_query`를 context에 저장.
- 앵커 처리: admin-term 경로 제거, geocode 실패 시 `anchor_failed` 응답, retrieve/apply에서 anchor만 필터로 사용.
- 모델/컨텍스트: `avoid` optional, `last_normalized_query` 추가, `last_admin_term` 제거.
- 추천 서비스: 기본 anchor 좌표를 env로 설정, 좌표 실패 시 기본 anchor로 fallback.
- Gemini 적용: `llm_clients`가 GEMINI 키 감지 시 Gemini 사용, 기본 모델 `gemini-2.0-flash`.
- 문서: `docs/chat_graph.mmd`, `docs/chat_stream_api.md` 업데이트.
- Langfuse: state 입력 슬림화, query/final 중심 출력, retrievals 요약 기록 추가.

### 추가 작업
- embedding_json 위경도 누락 점검/보정 워크플로우 정리 및 스크립트 운영.
- `scripts/poi/update_backend_db.py` 추가: `data/embedding_json` → `data/backend_db/merged_poi.json` 변환 병합.

### 설정/환경변수
- `GEMINI_API_KEY`, `RERANK_PROVIDER=gemini`, `GEMINI_RERANK_MODEL=gemini-2.0-flash`
- 선택: `GEMINI_CHAT_MODEL`, `DEFAULT_ANCHOR_LAT`, `DEFAULT_ANCHOR_LNG`

### 후속 작업
- Gemini 의존성 설치 필요 시: `langchain-google-genai`, `google-generativeai`
- 선택: `uv run ../scripts/render_graph.sh`로 `data/langgraph.mmd` 재생성
- 테스트: `pytest fastapi_app/tests/test_recommend_service.py`
