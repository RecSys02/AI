# Chat Stream API 문서 (한국어)

## 1. 개요
이 문서는 챗봇 스트리밍 API의 요청/응답 형식을 정의합니다. 프론트/백엔드가 동일한 규격으로 `context`를 주고받아, 후속 확장 기능(범위 확대, 이전 앵커 재사용 등)을 안정적으로 처리하는 것이 목적입니다.

- 서비스 범위: 서울
- 기본 통신 방식: **SSE(Server-Sent Events)**
- 권장 엔드포인트: **POST /chat/stream**
---
## 2. 엔드포인트

### 2.1 POST `/chat/stream`
- JSON Body로 요청
- `context` 수신/반환 포함
- 프로덕션 사용 권장

## 3. 요청 스키마 (POST)

### 3.1 ChatRequest
```json
{
  "query": "강남 맛집 추천해줘",
  "messages": [
    {"role": "user", "content": "홍대 맛집 추천해줘"},
    {"role": "assistant", "content": "홍대 근처 이런 곳 어때요?"}
  ],
  "historyPlaces": [
    {"placeId": 10, "category": "restaurant", "province": "seoul"}
  ],
  "top_k": 10,
  "debug": false,
  "preferred_themes": ["야경", "데이트"],
  "preferred_moods": ["조용한"],
  "preferred_restaurant_types": ["한식"],
  "preferred_cafe_types": ["로스터리"],
  "avoid": ["붐비는"],
  "activity_level": "적당히",
  "context": {
    "last_anchor": {
      "centers": [[37.4979052, 127.0275777]],
      "radius_by_intent": {"restaurant": 2.0, "cafe": 2.0, "tourspot": 3.0},
      "source": "anchor_cache"
    },
    "last_radius_km": 2.0,
    "last_mode": "restaurant",
    "last_query": "강남 맛집 추천해줘",
    "last_normalized_query": "강남역 근처 맛집 추천해줘",
    "last_resolved_name": "서울특별시 강남역사거리",
    "last_place": {"place": "강남"},
    "last_filter_applied": true
  }
}
```
#### 필드 설명
- **query** (필수, string)
  - 사용자 입력 문장
- **messages** (선택, ChatMessage[])
  - 직전 대화 메시지 목록
- **historyPlaces** (선택, HistoryPlace[])
  - 사용자의 히스토리 장소 목록
- **top_k** (선택, int)
  - 현재는 **항상 10으로 고정**되며, 요청 값은 무시됩니다.
- **debug** (선택, bool)
  - 디버그 출력 포함 여부
- **preferred_* / avoid / activity_level** (선택)
  - 사용자 취향/선호 정보
- **context** (선택, ChatContext)
  - 이전 응답에서 받은 상태값

> **주의**: `historyPlaces`, `placeId`는 camelCase를 사용합니다. 나머지 필드와 `context` 내부는 snake_case를 사용합니다.

---

### 3.2 ChatMessage
```json
{"role": "user", "content": "홍대 맛집 추천해줘"}
```
- **role**: "user" | "assistant"
- **content**: 메시지 텍스트
---
### 3.3 HistoryPlace
```json
{"placeId": 10, "category": "restaurant", "province": "seoul"}
```
- **placeId**: 내부 장소 ID
- **category**: "tourspot" | "restaurant" | "cafe"
- **province**: "seoul" | "busan" 등
---
### 3.4 ChatContext
```json
{
  "last_anchor": {"centers": [[37.4979052, 127.0275777]], "radius_by_intent": {}, "source": "anchor_cache"},
  "last_radius_km": 2.0,
  "last_mode": "restaurant",
  "last_query": "강남 맛집 추천해줘",
  "last_normalized_query": "강남역 근처 맛집 추천해줘",
  "last_resolved_name": "서울특별시 강남역사거리",
  "last_place": {"place": "강남"},
  "last_filter_applied": true
}
```

#### ChatContext 필드
- **last_anchor** (AnchorRef | null)
  - 앵커 좌표(centers), 출처(source), 반경 힌트(radius_by_intent)
- **last_radius_km** (float | null)
  - 마지막으로 사용한 반경(km)
- **last_mode** (string | null)
  - 마지막 모드("restaurant"|"cafe"|"tourspot")
- **last_query** (string | null)
  - 마지막 사용자 질의
- **last_normalized_query** (string | null)
  - 마지막 정규화된 질의
- **last_resolved_name** (string | null)
  - 실제 필터가 적용된 기준 지명
- **last_place** (PlaceRef | null)
  - 마지막으로 해석된 장소 (구조는 3.5 PlaceRef 참고)
- **last_filter_applied** (bool | null)
  - 앵커 필터가 실제 적용되었는지 여부

---

### 3.5 PlaceRef
PlaceRef는 **장소를 표현하는 최소 구조체**입니다. `context.last_place`가 이 형태를 따릅니다.

#### 필드
- **place** (string | null)
  - 단일 문자열 형태의 장소명 (예: \"강남\")
- **area** (string | null)
  - 넓은 지역명 (예: \"여의도\", \"신림동\")
- **point** (string | null)
  - 구체 지점/랜드마크 (예: \"더현대\", \"코엑스 아쿠아리움\")

#### 사용 규칙
- **다음 중 하나만 사용**해도 됩니다.
  - `{"place": "강남"}`
  - `{"area": "여의도", "point": null}`
  - `{"area": "여의도", "point": "더현대"}`
- 두 형태가 동시에 와도 파싱은 가능하지만, **권장되는 형식은 하나만 사용**입니다.

#### 예시
```json
{"place": "강남"}
```

```json
{"area": "여의도", "point": null}
```

```json
{"area": "여의도", "point": "더현대"}
```

---

## 4. 응답 스키마 (SSE)
스트리밍은 다음 이벤트들을 포함합니다.

### 4.1 이벤트 목록
- **event: token**
  - 토큰 단위 스트리밍
- **event: final**
  - 최종 답변 1회 전송
- **event: context**
  - 최종 상태값 전송 (백엔드 저장용)
- **event: debug**
  - 디버그 정보(선택)
- **event: node**
  - LangGraph 노드 이름(디버그용)
- **event: done**
  - 스트림 종료 신호

### 4.2 응답 예시
```
event: token
data: 서울특별시 강남역사거리 기준으로 추천드립니다.

event: final
data: 서울특별시 강남역사거리 기준으로 추천드립니다. ...

event: context
data: {"last_anchor": {"centers": [[37.4979052,127.0275777]], "radius_by_intent": {"restaurant":2.0,"cafe":2.0,"tourspot":3.0}, "source":"anchor_cache"}, "last_radius_km": 2.0, "last_mode": "restaurant", "last_query": "강남 맛집 추천해줘", "last_normalized_query": "강남역 근처 맛집 추천해줘", "last_resolved_name": "서울특별시 강남역사거리", "last_place": {"place": "강남"}, "last_filter_applied": true}

event: done
data: ok
```

> 백엔드 저장 규칙: `event: context`가 도착하면 그대로 저장해두었다가 다음 요청의 `context` 필드로 다시 전달합니다.

---
