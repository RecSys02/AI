# fastapi_app/main.py
# Run (dev): uvicorn main:app --reload --port 8000
import logging
import time
from fastapi import Depends, FastAPI, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from models.user_input import UserInput
from services.recommend_service import RecommendService

app = FastAPI(title="POI Recommendation API")
service = RecommendService()
logger = logging.getLogger("uvicorn.error")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    body = await request.body()
    # 과도한 본문은 자르되 내용과 오류를 남김
    body_preview = body[:2000] + (b"...(truncated)" if len(body) > 2000 else b"")
    logger.warning(
        "422 validation error method=%s path=%s query=%s body=%s errors=%s",
        request.method,
        request.url.path,
        request.url.query,
        body_preview,
        exc.errors(),
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.middleware("http")
async def add_process_time_header(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response


@app.post("/recommend")
def recommend(
    user: UserInput,
    top_k_per_category: int = Query(10, ge=1),
    distance_max_km: float = Query(3.0, ge=0.0, description="최근 선택 좌표 기준 최대 허용 거리(km). 0이면 필터 비활성."),
    debug: bool = Query(False, description="점수 구성요소(debug) 포함 여부"),
    svc: RecommendService = Depends(lambda: service),
):
    # distance_max_km=0 은 필터 비활성화
    max_km = None if distance_max_km == 0 else distance_max_km
    recommendations = svc.recommend(
        user,
        top_k_per_category=top_k_per_category,
        distance_max_km=max_km,
        debug=debug,
    )
    return {"recommendations": recommendations}
