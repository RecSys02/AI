# fastapi_app/services/scorers/restaurant.py
from .base import EMBEDDING_JSON_DIR, EMBEDDINGS_DIR, EmbeddingScorer


def build_restaurant_scorer():
    json_file = EMBEDDING_JSON_DIR / "embedding_restaurant.json"
    return EmbeddingScorer(
        name="restaurant",
        embedding_path=EMBEDDINGS_DIR / "embeddings_restaurant.npy",
        keys_path=EMBEDDINGS_DIR / "keys_restaurant.npy",
        json_path=json_file,
        coords_path=json_file,
    )
