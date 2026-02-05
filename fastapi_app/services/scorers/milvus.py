import math
from typing import Iterable, Optional

import numpy as np

from services.stores import get_milvus_store, get_postgres_store
from .base import _haversine_km, _is_valid_poi_meta


class MilvusScorer:
    def __init__(self, name: str):
        self.name = name
        self._milvus = get_milvus_store()
        self._pg = get_postgres_store()

    def _extract_lat_lng(self, meta_row: dict) -> tuple[Optional[float], Optional[float]]:
        if not meta_row:
            return None, None
        lat = meta_row.get("lat")
        lng = meta_row.get("lng")
        if lat is None or lng is None:
            meta = meta_row.get("meta") or {}
            lat = meta.get("lat") or meta.get("latitude")
            lng = meta.get("lng") or meta.get("lon") or meta.get("longitude")
            if (lat is None or lng is None) and isinstance(meta.get("location"), dict):
                loc = meta["location"]
                lat = loc.get("lat") or loc.get("latitude") or lat
                lng = loc.get("lng") or loc.get("lon") or loc.get("longitude") or lng
        try:
            return float(lat), float(lng)
        except Exception:
            return None, None

    def get_coords(self, place_id: int) -> Optional[tuple[float, float]]:
        meta = self._pg.fetch_meta([place_id], category=self.name).get(int(place_id))
        lat, lng = self._extract_lat_lng(meta or {})
        if lat is None or lng is None:
            return None
        return lat, lng

    def _recent_vector(self, place_ids: list[int]) -> np.ndarray | None:
        if not place_ids:
            return None
        vecs = self._milvus.get_vectors(self.name, place_ids)
        if not vecs:
            return None
        arr = np.stack(list(vecs.values()), axis=0)
        avg = arr.mean(axis=0)
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 0 else avg

    def _recent_vector_weighted(self, place_weights: dict[int, float]) -> np.ndarray | None:
        if not place_weights:
            return None
        vecs = self._milvus.get_vectors(self.name, list(place_weights.keys()))
        if not vecs:
            return None
        weighted = None
        for pid, vec in vecs.items():
            weight = place_weights.get(pid)
            if weight is None:
                try:
                    weight = place_weights.get(int(pid))
                except (TypeError, ValueError):
                    weight = None
            if weight is None or not math.isfinite(weight) or weight == 0.0:
                continue
            if weighted is None:
                weighted = vec * weight
            else:
                weighted = weighted + (vec * weight)
        if weighted is None:
            return None
        norm = np.linalg.norm(weighted)
        return weighted / norm if norm > 0 else weighted

    def _distance_from_recent_centroid(
        self, recent_place_ids: list[int], candidate_ids: list[int], meta_map: dict[int, dict]
    ) -> Optional[np.ndarray]:
        if not recent_place_ids:
            return None
        recent_meta = self._pg.fetch_meta(recent_place_ids, category=self.name)
        rec_coords = []
        for pid in recent_place_ids:
            row = recent_meta.get(int(pid))
            if not row:
                continue
            lat, lng = self._extract_lat_lng(row)
            if lat is not None and lng is not None:
                rec_coords.append((lat, lng))
        if not rec_coords:
            return None
        lats = np.array([c[0] for c in rec_coords], dtype=float)
        lngs = np.array([c[1] for c in rec_coords], dtype=float)
        centroid_lat = float(np.nanmean(lats))
        centroid_lng = float(np.nanmean(lngs))
        if math.isnan(centroid_lat) or math.isnan(centroid_lng):
            return None
        cand_lats = []
        cand_lngs = []
        for pid in candidate_ids:
            row = meta_map.get(int(pid))
            lat, lng = self._extract_lat_lng(row or {})
            cand_lats.append(lat if lat is not None else math.nan)
            cand_lngs.append(lng if lng is not None else math.nan)
        lat_rad = np.deg2rad(np.array(cand_lats, dtype=float))
        lng_rad = np.deg2rad(np.array(cand_lngs, dtype=float))
        dist = _haversine_km(lat_rad, lng_rad, math.radians(centroid_lat), math.radians(centroid_lng))
        return dist

    def _distance_from_anchor(
        self, lat: float, lng: float, candidate_ids: list[int], meta_map: dict[int, dict]
    ) -> Optional[np.ndarray]:
        cand_lats = []
        cand_lngs = []
        for pid in candidate_ids:
            row = meta_map.get(int(pid))
            lat_i, lng_i = self._extract_lat_lng(row or {})
            cand_lats.append(lat_i if lat_i is not None else math.nan)
            cand_lngs.append(lng_i if lng_i is not None else math.nan)
        lat_rad = np.deg2rad(np.array(cand_lats, dtype=float))
        lng_rad = np.deg2rad(np.array(cand_lngs, dtype=float))
        return _haversine_km(lat_rad, lng_rad, math.radians(lat), math.radians(lng))

    def topk(
        self,
        user_vec: np.ndarray,
        top_k: int = 10,
        recent_place_ids: list[int] | None = None,
        recent_place_weights: dict[int, float] | None = None,
        exclude_place_ids: Iterable[int] | None = None,
        distance_place_ids: list[int] | None = None,
        anchor_coords: tuple[float, float] | None = None,
        recent_weight: float = 0.3,
        distance_weight: float = 0.2,
        distance_scale_km: float = 5.0,
        distance_max_km: float | None = None,
        popularity_weight: float = 0.0,
        debug: bool = False,
        include_meta: bool = False,
    ):
        recent_place_ids = recent_place_ids or []
        # Reduce Milvus candidate pool size (default to 30, but never below top_k)
        candidate_k = max(top_k, 30)
        dense_hits = self._milvus.search(self.name, user_vec, top_k=candidate_k)
        if not dense_hits:
            return []
        exclude_set = {int(pid) for pid in exclude_place_ids} if exclude_place_ids else set()
        if exclude_set:
            dense_hits = [(pid, score) for pid, score in dense_hits if int(pid) not in exclude_set]
            if not dense_hits:
                return []

        candidate_ids = [pid for pid, _ in dense_hits]
        base_scores = np.array([score for _, score in dense_hits], dtype=float)
        scores = base_scores.copy()
        recent_component = np.zeros_like(scores)
        distance_component = np.zeros_like(scores)
        popularity_component = np.zeros_like(scores)
        distance_km = None

        meta_map = self._pg.fetch_meta(candidate_ids, category=self.name)

        dist = None
        if anchor_coords is not None:
            dist = self._distance_from_anchor(anchor_coords[0], anchor_coords[1], candidate_ids, meta_map)
        if dist is None:
            dist_ids = distance_place_ids if distance_place_ids is not None else recent_place_ids
            dist = self._distance_from_recent_centroid(dist_ids or [], candidate_ids, meta_map)
        if dist is not None:
            distance_km = dist

        recent_vec = None
        if recent_place_weights:
            recent_vec = self._recent_vector_weighted(recent_place_weights)
        if recent_vec is None:
            recent_vec = self._recent_vector(recent_place_ids)
        if recent_vec is not None and recent_weight != 0:
            vec_map = self._milvus.get_vectors(self.name, candidate_ids)
            recent_component = np.array(
                [
                    float(np.dot(vec_map.get(pid, np.zeros_like(recent_vec)), recent_vec))
                    for pid in candidate_ids
                ],
                dtype=float,
            )
            recent_component *= recent_weight
            scores += recent_component

        if distance_km is not None and distance_weight != 0:
            dist_bonus = np.exp(-distance_km / distance_scale_km)
            dist_bonus = np.where(np.isnan(dist_bonus), 0.0, dist_bonus)
            distance_component = distance_weight * dist_bonus
            scores += distance_component

        if popularity_weight != 0:
            popularity_component = np.array(
                [float(meta_map.get(pid, {}).get("popularity_score") or 0.0) for pid in candidate_ids],
                dtype=float,
            )
            popularity_component *= popularity_weight
            scores += popularity_component

        if distance_km is not None and distance_max_km is not None:
            valid_mask = (distance_km <= distance_max_km) & ~np.isnan(distance_km)
            valid_idxs = np.flatnonzero(valid_mask)
            if valid_idxs.size:
                sorted_local = scores[valid_idxs].argsort()[::-1]
                sorted_idxs = valid_idxs[sorted_local]
            else:
                sorted_idxs = np.array([], dtype=int)
        else:
            sorted_idxs = scores.argsort()[::-1]

        results = []
        filtered_count = 0
        max_check = min(len(sorted_idxs), top_k * 3)

        for i in sorted_idxs[:max_check]:
            pid = int(candidate_ids[i])
            meta_row = meta_map.get(pid, {})
            meta = meta_row.get("meta") if isinstance(meta_row, dict) else {}
            if not _is_valid_poi_meta(meta or {}, self.name):
                filtered_count += 1
                continue
            if len(results) >= top_k:
                break
            item = {
                "category": self.name,
                "province": meta_row.get("province") or (meta or {}).get("province"),
                "place_id": pid,
                "score": float(scores[i]),
            }
            if debug:
                item.update(
                    {
                        "score_base": float(base_scores[i]),
                        "score_recent": float(recent_component[i]),
                        "score_distance": float(distance_component[i]),
                        "score_popularity": float(popularity_component[i]),
                        "distance_km": float(distance_km[i]) if distance_km is not None else None,
                        "filtered_count": filtered_count,
                    }
                )
                if meta:
                    item["meta"] = meta
            if include_meta:
                item["_meta"] = meta or {}
            results.append(item)

        return results
