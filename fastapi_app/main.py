# fastapi_app/main.py
from fastapi import Depends, FastAPI, Query
from models.user_input import UserInput
from services.recommend_service import RecommendService

app = FastAPI(title="POI Recommendation API")
service = RecommendService()

@app.post("/recommend")
def recommend(
    user: UserInput,
    top_k_per_category: int = Query(10, ge=1),
    svc: RecommendService = Depends(lambda: service),
):
    recommendations = svc.recommend(user, top_k_per_category=top_k_per_category)
    return {"recommendations": recommendations}
