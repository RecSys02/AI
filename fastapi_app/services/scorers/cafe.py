# fastapi_app/services/scorers/cafe.py
from .base import EMBEDDINGS_DIR, EmbeddingScorer


def build_cafe_scorer():
    return EmbeddingScorer(
        name="cafe",
        embedding_path=EMBEDDINGS_DIR / "poi_embeddings_cafe.npy",
        keys_path=EMBEDDINGS_DIR / "poi_keys_cafe.npy",
    )
