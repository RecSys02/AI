# fastapi_app/main.py
from fastapi import Depends, FastAPI, Query
import time
from models.user_input import UserInput
from services.recommend_service import RecommendService

app = FastAPI(title="POI Recommendation API")
service = RecommendService()

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
    svc: RecommendService = Depends(lambda: service),
):
    # distance_max_km=0 은 필터 비활성화
    max_km = None if distance_max_km == 0 else distance_max_km
    recommendations = svc.recommend(
        user,
        top_k_per_category=top_k_per_category,
        distance_max_km=max_km,
    )
    return {"recommendations": recommendations}
