# services/scorers/tourspot.py
from .base import EMBEDDINGS_DIR, EmbeddingScorer

def build_tourspot_scorer():
    return EmbeddingScorer(
        name="tourspot",
        embedding_path=EMBEDDINGS_DIR / "poi_embeddings_tourspot.npy",
        keys_path=EMBEDDINGS_DIR / "poi_keys_tourspot.npy",
    )
