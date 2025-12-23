# fastapi_app/services/scorers/restaurant.py
from .base import EMBEDDING_JSON_DIR, EMBEDDINGS_DIR, EmbeddingScorer


def build_restaurant_scorer():
    return EmbeddingScorer(
        name="restaurant",
        embedding_path=EMBEDDINGS_DIR / "poi_embeddings_rest.npy",
        keys_path=EMBEDDINGS_DIR / "poi_keys_rest.npy",
        coords_path=EMBEDDING_JSON_DIR / "embedding_restaurant.json",
    )
