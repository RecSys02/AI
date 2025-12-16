# main.py
from fastapi import FastAPI
from models.user_input import UserInput
from recommend import recommend_pois

app = FastAPI(title="POI Recommendation API")


@app.post("/recommend")
def recommend(user: UserInput):
    results, latency = recommend_pois(user, top_k=10, return_latency=True)

    return {
        "recommendations": results,
        "latency": latency
    }
