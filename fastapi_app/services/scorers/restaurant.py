# fastapi_app/services/scorers/restaurant.py
from .base import EMBEDDING_JSON_DIR, EMBEDDINGS_DIR, EmbeddingScorer
from .milvus import MilvusScorer
from .utils import use_milvus


def build_restaurant_scorer():
    if use_milvus():
        return MilvusScorer(name="restaurant")
    return EmbeddingScorer(
        name="restaurant",
        embedding_path=EMBEDDINGS_DIR / "embeddings_restaurant.npy",
        keys_path=EMBEDDINGS_DIR / "keys_restaurant.npy",
        coords_path=EMBEDDING_JSON_DIR / "embedding_restaurant.json",
    )
