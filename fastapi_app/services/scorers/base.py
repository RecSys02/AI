import json
import math
from pathlib import Path
from typing import Optional, List

import numpy as np

# Project root: .../AI
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDING_JSON_DIR = PROJECT_ROOT / "data" / "embedding_json"


def _extract_cuisine_from_user_text(user_text: str) -> Optional[List[str]]:
    """사용자 텍스트에서 음식 카테고리를 추출합니다."""
    text_lower = user_text.lower()

    # 음식 카테고리 매핑
    cuisine_map = {
        "한식": ["한국음식", "한식", "korean"],
        "중식": ["중국음식", "중식", "chinese"],
        "일식": ["일본음식", "일식", "japanese"],
        "양식": ["서양음식", "양식", "western"],
        "이탈리안": ["이탈리아", "italian"],
        "프렌치": ["프랑스", "french"],
        "멕시칸": ["멕시코", "mexican"],
        "태국": ["태국", "thai"],
        "베트남": ["베트남", "vietnam"],
        "인도": ["인도", "india"],
    }

    for key, patterns in cuisine_map.items():
        if key in text_lower:
            return patterns

    return None


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
        json_path: str,
        coords_path: Optional[str] = None,
    ):
        self.name = name
        self.embedding_path = Path(embedding_path)
        self.keys_path = Path(keys_path)
        self.json_path = Path(json_path)
        self.coords_path = Path(coords_path) if coords_path else None
        self._embeddings = None
        self._keys = None
        self._idx_by_place_id = None
        self._poi_by_place_id = None
        self._lat = None
        self._lng = None
        self._coords_by_place_id = None

    def _load(self):
        if self._embeddings is None:
            self._embeddings = np.load(self.embedding_path)
            self._keys = np.load(self.keys_path, allow_pickle=True)
            self._idx_by_place_id = {int(k[2]): i for i, k in enumerate(self._keys)}
            self._load_poi_data()
            self._load_coords()

    def _load_poi_data(self):
        if not self.json_path.exists():
            print(f"[SCORER] POI json not found at {self.json_path}")
            return
        with self.json_path.open("r", encoding="utf-8") as f:
            pois = json.load(f)
            self._poi_by_place_id = {p["place_id"]: p for p in pois}
        print(
            f"[SCORER] loaded POI data from {self.json_path}: {len(self._poi_by_place_id)} items"
        )

    def _load_coords(self):
        if not self.coords_path:
            return
        print(f"[SCORER] loading coords from {self.coords_path}")
        with self.coords_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            self._coords_by_place_id = {}

            def _get_lat_lng(obj: dict):
                # support top-level lat/lng and nested location.lat/lng
                lat = obj.get("lat") or obj.get("latitude")
                lng = obj.get("lng") or obj.get("lon") or obj.get("longitude")
                if (lat is None or lng is None) and isinstance(obj.get("location"), dict):
                    loc = obj["location"]
                    lat = loc.get("lat") or loc.get("latitude") or lat
                    lng = loc.get("lng") or loc.get("lon") or loc.get("longitude") or lng
                return lat, lng

            for obj in data:
                if "place_id" not in obj:
                    continue
                lat, lng = _get_lat_lng(obj)
                if lat is None or lng is None:
                    if __debug__:
                        print(f"[SCORER] skip no coords place_id={obj.get('place_id')}")
                    continue
                try:
                    lat_f = float(lat)
                    lng_f = float(lng)
                except Exception:
                    if __debug__:
                        print(f"[SCORER] skip invalid coords place_id={obj.get('place_id')} lat={lat} lng={lng}")
                    continue
                self._coords_by_place_id[int(obj["place_id"])] = (lat_f, lng_f)
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
            print(f"[SCORER] loaded coords: {len(self._coords_by_place_id)} / keys={len(self._keys)}")

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
        distance_place_ids: list[int] | None = None,
        recent_weight: float = 0.3,
        distance_weight: float = 0.2,
        distance_scale_km: float = 5.0,
        distance_max_km: float | None = None,
        debug: bool = False,
        user_text: str | None = None,  # 음식 카테고리 필터링을 위한 사용자 텍스트
    ):
        self._load()
        base_scores = np.dot(self._embeddings, user_vec)
        scores = base_scores.copy()
        recent_component = np.zeros_like(scores)
        distance_component = np.zeros_like(scores)
        distance_km = None

        # recency by embedding
        recent_vec = self._recent_vector(recent_place_ids or [])
        if recent_vec is not None and recent_weight != 0:
            recent_component = recent_weight * np.dot(self._embeddings, recent_vec)
            scores += recent_component

        # distance bonus
        dist_ids = distance_place_ids if distance_place_ids is not None else recent_place_ids
        dist = self._distance_from_recent_centroid(dist_ids or [])
        if dist is not None:
            distance_km = dist
            if distance_max_km is not None:
                # 너무 먼 곳은 제외
                far_mask = (dist > distance_max_km) | np.isnan(dist)
                scores[far_mask] = -np.inf
            if distance_weight != 0:
                dist_bonus = np.exp(-dist / distance_scale_km)
                dist_bonus = np.where(np.isnan(dist_bonus), 0.0, dist_bonus)
                distance_component = distance_weight * dist_bonus
                scores += distance_component

        # 음식 카테고리 필터링 (restaurant/cafe만 해당)
        cuisine_patterns = None
        if self.name in ["restaurant", "cafe"] and user_text:
            cuisine_patterns = _extract_cuisine_from_user_text(user_text)

        # 점수 순으로 정렬
        idxs_all = scores.argsort()[::-1]

        # 음식 카테고리 필터가 있으면 적용
        if cuisine_patterns:
            filtered_idxs = []
            for i in idxs_all:
                place_id = int(self._keys[i][2])
                poi_data = self._poi_by_place_id.get(place_id, {}) if self._poi_by_place_id else {}
                content = poi_data.get("content", "").lower()
                # content 필드에 cuisine_patterns 중 하나라도 포함되면 통과
                if any(pattern.lower() in content for pattern in cuisine_patterns):
                    filtered_idxs.append(i)
                if len(filtered_idxs) >= top_k * 2:  # 넉넉하게 가져옴 (reranking 대비)
                    break
            idxs = filtered_idxs[:top_k] if filtered_idxs else idxs_all[:top_k]
        else:
            idxs = idxs_all[:top_k]

        results = []
        for i in idxs:
            place_id = int(self._keys[i][2])
            # self._poi_by_place_id가 로드되었는지 확인하고 데이터를 가져옵니다.
            poi_data = self._poi_by_place_id.get(place_id, {}) if self._poi_by_place_id else {}

            item = {
                # POI의 모든 정보를 결과에 포함시킵니다.
                **poi_data,
                "score": float(scores[i]),
                "place_id": place_id, # place_id를 명시적으로 보장합니다.
                "category": self.name, # 카테고리를 명시적으로 보장합니다.
                # [수정] Reranker를 위해 항상 debug 정보 포함
                "debug": {
                    "score_base": float(base_scores[i]),
                    "score_recent": float(recent_component[i]),
                    "score_distance": float(distance_component[i]),
                }
            }

            # [수정] 거리 정보도 항상 포함 (Reranker에서 사용)
            item["distance_km"] = float(distance_km[i]) if distance_km is not None and not np.isnan(distance_km[i]) else None

            results.append(item)

        return results
