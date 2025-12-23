# fastapi_app/services/scorers/restaurant.py
from .base import EMBEDDINGS_DIR, EmbeddingScorer


def build_restaurant_scorer():
    return EmbeddingScorer(
        name="restaurant",
        embedding_path=EMBEDDINGS_DIR / "poi_embeddings_rest.npy",
        keys_path=EMBEDDINGS_DIR / "poi_keys_rest.npy",
    )
