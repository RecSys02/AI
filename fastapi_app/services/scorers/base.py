import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

# Project root: .../AI
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDING_JSON_DIR = PROJECT_ROOT / "data" / "embedding_json"


def _is_valid_poi_meta(meta: dict, category: str) -> bool:
    """
    POI 메타데이터 유효성 검증 (런타임 필터링용)

    Args:
        meta: POI 메타데이터
        category: POI 카테고리 (tourspot/cafe/restaurant)

    Returns:
        유효하면 True, 아니면 False
    """
    if not meta:
        return True  # 메타데이터가 없으면 필터링 안함

    name = meta.get("name") or meta.get("title")
    if not name:
        return False

    # 금칙어 체크
    forbidden_names = [
        "내 위치",
        "현재 위치",
        "unknown",
        "test",
        "테스트",
        "샘플",
    ]
    name_lower = name.lower().strip()
    for forbidden in forbidden_names:
        if forbidden in name_lower:
            return False

    # 카페/레스토랑 전용 검증
    if category in ["cafe", "restaurant"]:
        content = meta.get("content", "").strip()

        if not content:
            return False

        # 유효하지 않은 content 값
        invalid_contents = [
            "평가중",
            "정보없음",
            "unknown",
            "n/a",
            ".",
            "-",
        ]
        content_lower = content.lower()
        if content_lower in invalid_contents:
            return False

        if len(content) < 2:
            return False

        # 카테고리 키워드 체크
        expected_keywords = {
            "cafe": ["카페", "커피", "디저트", "베이커리", "음료"],
            "restaurant": ["음식", "식당", "요리", "맛집", "한식", "중식", "일식", "양식", "세계음식"],
        }

        keywords = expected_keywords.get(category, [])
        has_keyword = any(keyword in content for keyword in keywords)

        if not has_keyword:
            return False

    return True


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
        self._meta_by_place_id = None

    def _load(self):
        if self._embeddings is None:
            self._embeddings = np.load(self.embedding_path)
            self._keys = np.load(self.keys_path, allow_pickle=True)
            self._idx_by_place_id = {int(k[2]): i for i, k in enumerate(self._keys)}
            self._load_coords()

    def _load_coords(self):
        if not self.coords_path:
            return
        print(f"[SCORER] loading coords from {self.coords_path}")
        with self.coords_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            self._coords_by_place_id = {}
            self._meta_by_place_id = {}

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
                place_id = int(obj["place_id"])
                # 메타데이터 저장
                self._meta_by_place_id[place_id] = obj

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
                self._coords_by_place_id[place_id] = (lat_f, lng_f)
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

    def get_coords(self, place_id: int) -> Optional[tuple[float, float]]:
        self._load()
        if not self._coords_by_place_id:
            return None
        return self._coords_by_place_id.get(int(place_id))

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

    def _distance_from_anchor(self, lat: float, lng: float) -> Optional[np.ndarray]:
        if self._lat is None or self._lng is None:
            return None
        lat_rad = np.deg2rad(self._lat)
        lng_rad = np.deg2rad(self._lng)
        return _haversine_km(lat_rad, lng_rad, math.radians(lat), math.radians(lng))

    def topk(
        self,
        user_vec: np.ndarray,
        top_k: int = 10,
        recent_place_ids: list[int] | None = None,
        distance_place_ids: list[int] | None = None,
        anchor_coords: tuple[float, float] | None = None,
        recent_weight: float = 0.3,
        distance_weight: float = 0.2,
        distance_scale_km: float = 5.0,
        distance_max_km: float | None = None,
        debug: bool = False,
        include_meta: bool = False,
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
        dist = None
        if anchor_coords is not None:
            dist = self._distance_from_anchor(anchor_coords[0], anchor_coords[1])
        if dist is None:
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

        # 정렬된 인덱스 (높은 점수부터)
        sorted_idxs = scores.argsort()[::-1]

        results = []
        filtered_count = 0
        checked_count = 0
        max_check = min(len(sorted_idxs), top_k * 3)  # 최대 top_k의 3배까지 확인

        for i in sorted_idxs[:max_check]:
            checked_count += 1
            place_id = int(self._keys[i][2])

            # 메타데이터 유효성 검증
            meta = self._meta_by_place_id.get(place_id, {}) if self._meta_by_place_id else {}
            if not _is_valid_poi_meta(meta, self.name):
                filtered_count += 1
                name = meta.get("name", "N/A")
                content = meta.get("content", "N/A")
                print(f"[SCORER] Filtered out: place_id={place_id}, name='{name}', content='{content}'")
                continue

            item = {
                "category": self.name,
                "region": self._keys[i][0],
                "place_id": place_id,
                "score": float(scores[i]),
            }
            if debug:
                item.update({
                    "score_base": float(base_scores[i]),
                    "score_recent": float(recent_component[i]),
                    "score_distance": float(distance_component[i]),
                    "distance_km": float(distance_km[i]) if distance_km is not None else None,
                })
                # debug=True일 때는 메타데이터도 포함
                if meta:
                    item["meta"] = meta
            # include_meta=True일 때만 내부 메타데이터 추가 (LLM reranking용)
            if include_meta:
                item["_meta"] = meta
            results.append(item)

            # 충분한 개수를 모으면 종료
            if len(results) >= top_k:
                break

        if filtered_count > 0:
            print(f"[SCORER] {self.name}: Checked {checked_count} items, filtered out {filtered_count}, returned {len(results)}")

        return results
