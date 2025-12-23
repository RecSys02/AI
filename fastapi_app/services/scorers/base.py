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
        self._idx_by_place_id = None

    def _load(self):
        if self._embeddings is None:
            self._embeddings = np.load(self.embedding_path)
            self._keys = np.load(self.keys_path, allow_pickle=True)
            self._idx_by_place_id = {int(k[2]): i for i, k in enumerate(self._keys)}

    def _recent_vector(self, place_ids: list[int]) -> np.ndarray | None:
        if not place_ids:
            return None
        self._load()
        idxs = [self._idx_by_place_id.get(int(pid)) for pid in place_ids]
        idxs = [i for i in idxs if i is not None]
        if not idxs:
            return None
        vecs = self._embeddings[idxs]
        avg = vecs.mean(axis=0)
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 0 else avg

    def topk(
        self,
        user_vec: np.ndarray,
        top_k: int = 10,
        recent_place_ids: list[int] | None = None,
        recent_weight: float = 0.3,
    ):
        self._load()
        scores = np.dot(self._embeddings, user_vec)
        recent_vec = self._recent_vector(recent_place_ids or [])
        if recent_vec is not None and recent_weight != 0:
            scores += recent_weight * np.dot(self._embeddings, recent_vec)

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
