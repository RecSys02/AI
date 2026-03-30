# Async Retrieval Refactor Interview Notes

## 1. 한 줄 요약

이 챗봇은 원래 FastAPI 라우트와 SSE 스트리밍 자체는 `async`였지만, 내부 retrieval 파이프라인이 동기 DB 조회, 동기 Milvus 검색, 동기 임베딩 인코딩으로 구성돼 있어서 실제로는 이벤트 루프를 막고 있었습니다. 이를 해결하기 위해 retrieval 경로를 다시 설계해서, I/O는 async로 바꾸고 CPU 바운드 작업은 별도 executor로 분리했습니다.

## 2. 원래 어떤 문제가 있었나

겉으로 보면 API는 비동기처럼 보였습니다.

- FastAPI 엔드포인트가 `async def`
- SSE 스트리밍으로 토큰 응답
- LangGraph도 async 이벤트 스트림 사용

하지만 내부 검색 흐름은 실제로는 블로킹 구조였습니다.

- Postgres 조회를 동기로 실행
- Milvus 검색을 동기로 실행
- `SentenceTransformer.encode()`를 요청 처리 쓰레드에서 직접 실행

즉, `겉은 async인데 속은 blocking`인 상태였습니다.

## 3. Before 구조

기존 구조는 대략 아래와 같았습니다.

```text
request
-> async router
-> retrieve()
-> sync Postgres query
-> sync Milvus search
-> sync embedding encode
-> event loop blocked
```

코드 레벨로는 이런 식이었습니다.

```python
async def retrieve_node(...):
    hits = retrieve(...)  # 내부가 동기 호출
```

문제는 `retrieve()`가 async 함수가 아니었고, 그 안에서 외부 I/O와 무거운 연산을 모두 동기로 처리했다는 점입니다.

## 4. 여러 사용자가 동시에 접근하면 무슨 일이 생기나

예를 들어 A 사용자가 검색 요청을 보냈다고 가정합니다.

1. A 요청이 retrieval 단계에 들어감
2. Postgres 조회, Milvus 검색, encode가 순차적으로 동기로 실행됨
3. 그동안 같은 worker의 이벤트 루프가 해당 작업에 묶일 수 있음
4. 동시에 들어온 B, C 요청의 진행이 늦어짐
5. 스트리밍 응답이라도 첫 토큰이 늦거나 중간 체감이 버벅일 수 있음

정리하면 이런 문제가 있었습니다.

- 동시성 저하
- 응답 지연 증가
- tail latency 증가
- SSE 스트리밍 체감 품질 저하

중요한 포인트는 `async def`라고 자동으로 비동기 아키텍처가 되는 게 아니라는 점입니다. 내부에서 동기 블로킹 함수를 직접 호출하면 async의 장점이 거의 사라집니다.

## 5. 어떻게 고쳤나

핵심 원칙은 아래 두 가지였습니다.

- I/O는 async로 처리
- CPU 바운드 작업은 executor로 분리

### 5.1 Postgres를 async로 변경

기존 동기 psycopg 호출과 별개로 async connection 기반 메서드를 추가했습니다.

- async connection 생성
- async FTS 조회
- async 이름 조회
- async meta 조회

즉, DB I/O를 `await` 가능한 형태로 바꿨습니다.

### 5.2 Milvus 검색을 async로 변경

기존에는 sync `Collection.search()` 경로를 사용했지만, retrieval 경로에서는 `AsyncMilvusClient` 기반의 async search를 사용하도록 바꿨습니다.

또한 sync connect를 store 생성 시점에 바로 하지 않고 lazy init으로 바꿔서, async 경로에서 불필요한 sync connect가 끼어들지 않게 했습니다.

### 5.3 Retrieval 자체를 async orchestration으로 변경

이제 `retrieve()` 자체가 async 함수가 되었고, 내부에서 async DB 조회와 async Milvus 검색을 조합합니다.

Before:

```python
hits = retrieve(...)
```

After:

```python
hits = await retrieve(...)
```

또한 가능한 부분은 병렬화했습니다.

- BM25 조회
- 임베딩 준비

이 둘은 서로 독립적이어서 동시에 시작할 수 있게 정리했습니다.

### 5.4 임베딩 인코딩은 executor로 분리

임베딩 인코딩은 CPU/GPU 바운드 작업이기 때문에, 이걸 억지로 async 함수로 바꾼다고 해결되지는 않습니다.

그래서 이벤트 루프에서 직접 실행하지 않고, 전용 executor로 분리했습니다.

즉, 이 부분은 `진짜 async`가 아니라 `event loop를 막지 않도록 격리`한 것입니다.

### 5.5 위치 필터는 Geo-hash prefilter + exact distance로 최적화

위치 기반 추천에서는 anchor 좌표와 반경이 주어졌을 때, 후보 전체에 바로 exact distance 계산을 거는 대신 Geo-hash를 사전 필터로 추가했습니다.

흐름은 아래와 같습니다.

- anchor 좌표와 반경에 맞는 Geo-hash precision 선택
- anchor 주변 셀과 경계 인접 셀의 prefix 집합 생성
- 후보 장소 중 같은 Geo-hash prefix를 가진 항목만 먼저 남김
- 마지막에는 Haversine distance로 exact distance를 다시 계산해서 최종 판정

즉, Geo-hash는 `정확한 거리 계산을 대체하는 것`이 아니라, `후보 수를 먼저 줄이는 공간 prefilter` 역할입니다.

이렇게 한 이유는 두 가지입니다.

- 반경 필터가 자주 걸리는 추천형 질의에서 불필요한 거리 계산 대상을 줄이기 위해
- 공간 인덱싱 개념은 활용하되, 경계 오차는 exact distance 재검증으로 보완하기 위해

그래서 면접에서는 `Geo-hash prefix로 지역 후보를 먼저 줄이고, 최종 판정은 exact distance로 다시 검증했다`고 말하면 됩니다.

### 5.6 LangGraph fallback도 async로 통일

라우터 fallback에서 사용하던 sync `invoke()`를 async `ainvoke()`로 바꿨습니다.

그래서 라우터 레벨에서도 남아 있던 sync fallback 블로킹 포인트를 제거했습니다.

## 6. After 구조

리팩토링 후 구조는 아래와 같습니다.

```text
request
-> async router
-> await async retrieve
-> await async Postgres
-> await async Milvus
-> encode in dedicated executor
-> event loop remains responsive
```

이제 여러 명이 동시에 들어와도 최소한 API 이벤트 루프가 검색 로직 때문에 직접 막히지는 않게 됐습니다.

## 7. 무엇이 좋아졌나

- 여러 요청이 동시에 들어와도 이벤트 루프가 덜 막힘
- 스트리밍 응답성이 개선될 수 있는 구조가 됨
- retrieval 경로가 겉보기 async가 아니라 실제 non-blocking 흐름으로 정리됨
- I/O와 CPU 작업의 책임이 분리됨
- 위치 기반 추천에서는 Geo-hash prefilter로 지역 후보를 먼저 줄이고, exact distance는 최종 검증에만 쓰게 됨

## 8. 아직 남아 있는 점

면접에서는 이 부분도 같이 말하는 게 좋습니다.

- Postgres는 async이지만 아직 connection pool까지 붙인 건 아님
- 임베딩 인코딩은 executor로 분리했지만, 별도 inference service까지는 아님
- 챗봇 retrieval 경로 중심으로 정리했고, 추천 시스템의 다른 sync scorer 경로는 별도 범위

즉, 구조적으로는 크게 개선했지만 운영 최적화 포인트는 더 남아 있다고 설명하면 됩니다.

## 9. 비동기의 장점과 단점

### 장점

- I/O 대기 중에도 다른 요청 처리 가능
- 동시 사용자 대응에 유리
- SSE, websocket, streaming API와 잘 맞음

### 단점

- 코드 구조가 더 복잡해짐
- 디버깅이 어려워짐
- async/sync 혼합 시 실수하기 쉬움
- CPU 바운드 작업은 async만으로 해결되지 않음

실무적으로는 `무조건 async`가 아니라, I/O 중심 경로만 async로 만들고 CPU 작업은 executor나 worker로 분리하는 게 현실적입니다.

## 10. 면접에서 이렇게 말하면 좋다

### 10.1 30초 버전

처음에는 챗봇 API가 async로 작성돼 있어서 비동기 구조라고 생각했지만, 실제 retrieval 내부는 동기 Postgres 조회, 동기 Milvus 검색, 동기 임베딩 인코딩으로 구성돼 있어서 이벤트 루프를 막고 있었습니다. 그래서 retrieval 경로를 다시 설계해서 Postgres와 Milvus는 async로 바꾸고, CPU 바운드인 임베딩 인코딩은 executor로 분리했습니다. 또 sync fallback도 `ainvoke()`로 통일해서 실제 non-blocking 구조로 리팩토링했습니다.

### 10.2 1분 버전

이 프로젝트는 원래 FastAPI 라우트와 SSE 스트리밍 자체는 async였지만, 실제 병목은 retrieval 단계였습니다. 내부에서 동기  동기 Milvus 검색, 그리고 로컬 임베딩 인코딩을 그대로 실행하고 있어서, 여러 사용자가 동시에 들어오면 한 요청의 검색 작업이 같은 worker의 다른 요청 응답성까지 떨어뜨릴 수 있는 구조였습니다. 그래서 저는 단순히 함수에 async를 붙이는 방식이 아니라, 작업 성격에 맞게 분리하는 쪽으로 접근했습니다.ilvus는 async client로 전환했습니다. retrieval 함수 자체도 async orchestration으로 바꿔서 `await` 기반 흐름으로 만들었고, CPU 바운드인 임베딩 인코딩은 전용 executor로 분리했습니다. 또 라우터 fallback도 sync invoke 대신 `ainvoke()`로 변경했습니다. 결과적으로 겉보기 async가 아니라, 실제로 이벤트 루프를 덜 막는 구조로 개선했다고 설명할 수 있습니다.

## 11. 예상 질문과 답변

### Q. 원래도 async였는데 뭐가 문제였나요?

A. 라우트 시그니처만 async였고, 내부 retrieval이 동기 I/O와 동기 연산을 직접 실행하고 있었습니다. 그래서 실제로는 이벤트 루프를 막을 수 있었습니다.

### Q. 왜 모든 걸 async로 바꾸지 않고 executor를 썼나요?

A. 임베딩 인코딩은 I/O가 아니라 CPU/GPU 바운드 작업이라서, async 함수로 바꿔도 본질적으로 해결되지 않습니다. 이런 작업은 이벤트 루프 밖의 executor나 worker로 분리하는 게 맞습니다.

### Q. 비동기의 단점은 없나요?

A. 있습니다. 구조가 복잡해지고 디버깅이 어려워집니다. 그래서 무조건 async로 가기보다는, I/O 중심 경로에만 적용하고 CPU 작업은 별도 실행 환경으로 빼는 게 현실적입니다.


### Q. Geo-hash를 왜 넣었나요?

A. 위치 기반 추천에서는 반경 필터가 자주 걸리기 때문에, 모든 후보에 바로 exact distance를 계산하기보다 Geo-hash prefix로 같은 영역 후보를 먼저 줄이는 게 효율적입니다. 다만 Geo-hash만으로는 경계 오차가 있을 수 있어서, 랭킹단계에서의 거리 점수 계산은 Haversine distance로 다시 검증했습니다.

### Q. Geo-hash가 행정구역 필터와 같은 건가요?

A. 역할은 비슷하게 사전 필터지만 기준이 다릅니다. 행정구역은 `종로구`처럼 주소 체계 기반이고, Geo-hash는 위경도를 격자 문자열로 인코딩한 공간 인덱싱입니다. 그래서 `성수 근처 2km` 같은 좌표 중심 질의에는 Geo-hash가 더 직접적으로 맞습니다.

## 12. 429 / Rate Limit 대응

비동기화 이후에는 앱 내부 병목이 줄어들기 때문에, 오히려 외부 LLM provider의 rate limit이 더 빨리 드러날 수 있습니다.

즉, 이전에는 서버가 느려서 `우연히 self-throttling` 되고 있었을 수 있고, 비동기화 이후에는 더 많은 동시 요청이 실제로 provider까지 도달하면서 429가 더 잘 보일 수 있습니다.

이건 비동기화의 실패가 아니라, 원래 숨어 있던 upstream quota bottleneck이 드러난 것입니다.

### 12.1 왜 429가 더 잘 보일 수 있나

- Before: 서버 내부 blocking 때문에 outbound LLM 호출량이 낮았음
- After: 이벤트 루프가 덜 막히면서 동시에 더 많은 LLM 호출이 나갈 수 있음
- 결과: 순간 burst가 커지면 provider에서 429를 반환할 가능성이 올라감

중요한 포인트는 `단일 요청이 quota를 더 많이 먹는 것`이 아니라, `동시 요청 처리량이 올라가면서 순간 호출량이 커지는 것`입니다.

### 12.2 어떻게 대응했나

LLM 공통 클라이언트 레이어에 rate limit 보호 장치를 넣었습니다.

- `Semaphore`로 LLM의 동시 호출 수 제한
- 429 계열 예외를 공통 감지
- `Retry-After`가 있으면 우선 존중
- 없으면 exponential backoff + jitter로 재시도
- 재시도 끝까지 실패하면 사용자에게 fallback 메시지 반환

즉, `비동기화로 처리량은 올리되`, outbound 호출은 무제한으로 터뜨리지 않도록 backpressure를 건 구조입니다.

### 12.3 이 설계의 의미

이 대응의 목적은 세 가지입니다.

- provider에 순간 burst를 줄인다
- 일시적인 429는 자동 복구한다
- 반복 실패 시 사용자 경험이 깨지지 않게 한다

즉, `비동기화`와 `rate limit 제어`는 같이 가야 하는 설계입니다.

### 12.4 면접에서 이렇게 말하면 좋다

“비동기화 이후에는 내부 병목이 줄어들기 때문에 외부 LLM provider의 429가 더 빨리 드러날 수 있습니다. 그래서 공통 LLM 래퍼에 semaphore 기반 동시성 제한과 429 retry/backoff를 넣었고, 최종 실패 시에는 사용자에게 fallback 메시지를 반환하도록 했습니다. 즉, throughput을 높이는 동시에 outbound rate도 제어하는 구조로 설계했습니다.”

### 12.5 예상 질문

#### Q. API 키가 하나면 429는 어쩔 수 없는 것 아닌가요?

A. 완전히는 아닙니다. 순간 burst로 나는 429는 semaphore, rate limiting, retry/backoff로 많이 줄일 수 있습니다. 다만 지속적으로 quota를 넘는 수준이면 호출 수를 줄이거나 provider quota를 올려야 합니다.

#### Q. 여러 키로 나누면 해결되나요?

A. 보통은 근본 해결책이 아닙니다. 많은 provider는 계정/프로젝트 단위로 limit를 걸기 때문에, 먼저 outbound 호출량을 제어하는 게 실무적으로 맞습니다.

#### Q. fallback 메시지를 넣은 이유는 뭔가요?

A. 스트리밍 중 또는 최종 생성 단계에서 provider가 계속 429를 반환하면 사용자 경험이 바로 깨집니다. 그래서 일시적 실패는 자동 재시도하고, 최종 실패는 서비스 메시지로 부드럽게 마무리하도록 했습니다.

## 13. 마지막 정리

이 리팩토링의 핵심은 아래 문장으로 정리할 수 있습니다.

`겉보기 async였던 챗봇을, 실제로 이벤트 루프를 덜 막는 non-blocking 구조로 바꿨다.`

그리고 그 방법은 아래 두 줄로 설명할 수 있습니다.

- I/O는 async client로 전환
- CPU 바운드 작업은 executor로 분리

추가로 운영 안정성 측면에서는 아래 한 줄도 같이 말하면 좋습니다.

`비동기화로 늘어난 처리량이 외부 provider 429로 이어지지 않도록, semaphore와 retry/backoff로 outbound rate를 제어했다.`

위치 기반 추천 최적화까지 한 줄로 붙이면 아래처럼 정리할 수 있습니다.

`위치 필터는 Geo-hash prefix로 후보를 먼저 줄이고, 최종 정확도는 exact distance 계산으로 보장했다.`
