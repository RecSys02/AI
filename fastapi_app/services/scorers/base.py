import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

# Project root: .../AI
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDING_JSON_DIR = PROJECT_ROOT / "data" / "embedding_json"


def _haversine_km(lat1, lon1, lat2, lon2):
    # Vectorized haversine (lat/lon in radians)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


class EmbeddingScorer:
    def __init__(
        self,
        name: str,
        embedding_path: str,
        keys_path: str,
        coords_path: Optional[str] = None,
    ):
        self.name = name
        self.embedding_path = Path(embedding_path)
        self.keys_path = Path(keys_path)
        self.coords_path = Path(coords_path) if coords_path else None
        self._embeddings = None
        self._keys = None
        self._idx_by_place_id = None
        self._lat = None
        self._lng = None
        self._coords_by_place_id = None

    def _load(self):
        if self._embeddings is None:
            self._embeddings = np.load(self.embedding_path)
            self._keys = np.load(self.keys_path, allow_pickle=True)
            self._idx_by_place_id = {int(k[2]): i for i, k in enumerate(self._keys)}
            self._load_coords()

    def _load_coords(self):
        if not self.coords_path:
            return
        with self.coords_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self._coords_by_place_id = {
            int(obj["place_id"]): (float(obj["lat"]), float(obj.get("lng") or obj.get("lon")))
            for obj in data
            if "place_id" in obj and "lat" in obj and ("lng" in obj or "lon" in obj)
        }
        lats = []
        lngs = []
        for k in self._keys:
            pid = int(k[2])
            coord = self._coords_by_place_id.get(pid)
            if coord:
                lats.append(coord[0])
                lngs.append(coord[1])
            else:
                lats.append(math.nan)
                lngs.append(math.nan)
        self._lat = np.array(lats)
        self._lng = np.array(lngs)

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

    def _distance_from_recent_centroid(self, recent_place_ids: list[int]) -> Optional[np.ndarray]:
        if not recent_place_ids or self._lat is None or self._lng is None:
            return None
        rec_coords = []
        for pid in recent_place_ids:
            coord = self._coords_by_place_id.get(int(pid)) if self._coords_by_place_id else None
            if coord:
                rec_coords.append(coord)
        if not rec_coords:
            return None

        # centroid(평균) 좌표와의 거리만 사용해 O(N)으로 계산
        lats = np.array([c[0] for c in rec_coords], dtype=float)
        lngs = np.array([c[1] for c in rec_coords], dtype=float)
        centroid_lat = float(np.nanmean(lats))
        centroid_lng = float(np.nanmean(lngs))
        if math.isnan(centroid_lat) or math.isnan(centroid_lng):
            return None

        lat_rad = np.deg2rad(self._lat)
        lng_rad = np.deg2rad(self._lng)
        dist = _haversine_km(lat_rad, lng_rad, math.radians(centroid_lat), math.radians(centroid_lng))
        return dist

    def topk(
        self,
        user_vec: np.ndarray,
        top_k: int = 10,
        recent_place_ids: list[int] | None = None,
        recent_weight: float = 0.3,
        distance_weight: float = 0.2,
        distance_scale_km: float = 5.0,
        distance_max_km: float | None = None,
    ):
        self._load()
        scores = np.dot(self._embeddings, user_vec)

        # recency by embedding
        recent_vec = self._recent_vector(recent_place_ids or [])
        if recent_vec is not None and recent_weight != 0:
            scores += recent_weight * np.dot(self._embeddings, recent_vec)

        # distance bonus
        dist = self._distance_from_recent_centroid(recent_place_ids or [])
        if dist is not None:
            if distance_max_km is not None:
                # 너무 먼 곳은 제외
                far_mask = dist > distance_max_km
                scores[far_mask] = -np.inf
            if distance_weight != 0:
                dist_bonus = np.exp(-dist / distance_scale_km)
                dist_bonus = np.where(np.isnan(dist_bonus), 0.0, dist_bonus)
                scores += distance_weight * dist_bonus

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
