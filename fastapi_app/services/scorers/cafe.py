# fastapi_app/services/scorers/cafe.py
from .base import EMBEDDING_JSON_DIR, EMBEDDINGS_DIR, EmbeddingScorer


def build_cafe_scorer():
    return EmbeddingScorer(
        name="cafe",
        embedding_path=EMBEDDINGS_DIR / "embeddings_cafe.npy",
        keys_path=EMBEDDINGS_DIR / "keys_cafe.npy",
        coords_path=EMBEDDING_JSON_DIR / "embedding_cafe.json",
    )
