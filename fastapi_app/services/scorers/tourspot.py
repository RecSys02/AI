# services/scorers/tourspot.py
from .base import EMBEDDING_JSON_DIR, EMBEDDINGS_DIR, EmbeddingScorer

def build_tourspot_scorer():
    return EmbeddingScorer(
        name="tourspot",
        embedding_path=EMBEDDINGS_DIR / "embeddings_tourspot.npy",
        keys_path=EMBEDDINGS_DIR / "keys_tourspot.npy",
        coords_path=EMBEDDING_JSON_DIR / "embedding_tourspot.json",
    )
3