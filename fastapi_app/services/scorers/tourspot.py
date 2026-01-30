# services/scorers/tourspot.py
from .base import EMBEDDING_JSON_DIR, EMBEDDINGS_DIR, EmbeddingScorer
from .milvus import MilvusScorer
from .utils import use_milvus

def build_tourspot_scorer():
    if use_milvus():
        return MilvusScorer(name="tourspot")
    return EmbeddingScorer(
        name="tourspot",
        embedding_path=EMBEDDINGS_DIR / "embeddings_tourspot.npy",
        keys_path=EMBEDDINGS_DIR / "keys_tourspot.npy",
        coords_path=EMBEDDING_JSON_DIR / "embedding_tourspot.json",
    )
3
