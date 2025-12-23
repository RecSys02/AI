# services/scorers/base.py
import numpy as np
from pathlib import Path

# Project root: .../AI
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

class EmbeddingScorer:
    def __init__(self, name: str, embedding_path: str, keys_path: str):
        self.name = name
        self.embedding_path = Path(embedding_path)
        self.keys_path = Path(keys_path)
        self._embeddings = None
        self._keys = None

    def _load(self):
        if self._embeddings is None:
            self._embeddings = np.load(self.embedding_path)
            self._keys = np.load(self.keys_path, allow_pickle=True)

    def topk(self, user_vec: np.ndarray, top_k: int = 10):
        self._load()
        scores = np.dot(self._embeddings, user_vec)
        idxs = scores.argsort()[::-1][:top_k]
        return [
            {
                "category": self.name,
                "region": self._keys[i][0],
                "place_id": int(self._keys[i][2]),
                "score": float(scores[i]),
            }
            for i in idxs
        ]
