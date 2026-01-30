# fastapi_app/services/scorers/cafe.py
from .base import EMBEDDING_JSON_DIR, EMBEDDINGS_DIR, EmbeddingScorer
from .milvus import MilvusScorer
from .utils import use_milvus


def build_cafe_scorer():
    if use_milvus():
        return MilvusScorer(name="cafe")
    return EmbeddingScorer(
        name="cafe",
        embedding_path=EMBEDDINGS_DIR / "embeddings_cafe.npy",
        keys_path=EMBEDDINGS_DIR / "keys_cafe.npy",
        coords_path=EMBEDDING_JSON_DIR / "embedding_cafe.json",
    )
