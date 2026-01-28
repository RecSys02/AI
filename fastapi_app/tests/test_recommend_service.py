import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import services.recommend_service as recommend_service
from models.poi_ref import PoiRef
from models.user_input import UserInput


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, normalize_embeddings=True):
        return [[1.0] for _ in texts]


class DummyScorer:
    def __init__(self, name, results):
        self.name = name
        self.results = results
        self.last_debug = None
        self.last_distance_max_km = None
        self.last_anchor_coords = None

    def topk(self, user_vec, top_k, **kwargs):
        self.last_debug = kwargs.get("debug")
        self.last_distance_max_km = kwargs.get("distance_max_km")
        self.last_anchor_coords = kwargs.get("anchor_coords")
        max_km = self.last_distance_max_km
        items = self.results
        if max_km is not None:
            items = [
                r
                for r in items
                if r.get("distance_km") is None or r["distance_km"] <= max_km
            ]
        return items[:top_k]


@pytest.fixture(autouse=True)
def patch_sentence_transformer(monkeypatch):
    monkeypatch.setattr(
        recommend_service, "SentenceTransformer", DummySentenceTransformer
    )


@pytest.fixture
def make_service(monkeypatch):
    def _make(scored_items):
        def build(name):
            return DummyScorer(name, scored_items[name])

        monkeypatch.setattr(
            recommend_service, "build_tourspot_scorer", lambda: build("tourspot")
        )
        monkeypatch.setattr(
            recommend_service, "build_cafe_scorer", lambda: build("cafe")
        )
        monkeypatch.setattr(
            recommend_service, "build_restaurant_scorer", lambda: build("restaurant")
        )
        return recommend_service.RecommendService()

    return _make


def _base_user(**overrides):
    base = dict(
        user_id=1,
        preferred_themes=[],
        preferred_moods=[],
        preferred_restaurant_types=[],
        preferred_cafe_types=[],
        avoid=[],
        activity_level="적당히",
        region="seoul",
        companion=[],
    )
    base.update(overrides)
    return UserInput(**base)


def test_history_filtered_topk_and_debug(make_service):
    scored = {
        "tourspot": [
            {"place_id": 1, "score": 1.0, "score_base": 1.0},
            {"place_id": 2, "score": 0.5, "score_base": 0.5},
        ],
        "cafe": [{"place_id": 3, "score": 1.0}],
        "restaurant": [{"place_id": 5, "score": 1.0}],
    }
    svc = make_service(scored)
    user = _base_user(
        history_places=[PoiRef(place_id=2, category="tourspot", province="seoul")]
    )

    result = svc.recommend(user, top_k_per_category=1, distance_max_km=5.0, debug=True)

    tour = next(r for r in result if r["category"] == "tourspot")
    assert [item["place_id"] for item in tour["items"]] == [1]
    # RecommendService should pass debug flag through to scorer
    assert svc.scorers[0].last_debug is True


def test_distance_max_filters_far(make_service):
    scored = {
        "tourspot": [
            {"place_id": 10, "distance_km": 2.0},
            {"place_id": 11, "distance_km": 5.1},
        ],
        "cafe": [],
        "restaurant": [],
    }
    svc = make_service(scored)
    user = _base_user()

    result = svc.recommend(user, top_k_per_category=5, distance_max_km=5.0)

    tour = next(r for r in result if r["category"] == "tourspot")
    assert [item["place_id"] for item in tour["items"]] == [10]
    assert svc.scorers[0].last_distance_max_km == 5.0


def test_region_anchor_used_when_no_selected_places(make_service, monkeypatch):
    scored = {"tourspot": [], "cafe": [], "restaurant": []}
    monkeypatch.setattr(
        recommend_service,
        "load_geo_centers",
        lambda: {"서울특별시 동작구": {"centers": [[37.5124, 126.9395]]}},
    )
    svc = make_service(scored)
    user = _base_user(region="서울특별시 동작구")

    svc.recommend(user, top_k_per_category=1, distance_max_km=5.0)

    assert svc.scorers[0].last_anchor_coords == (37.5124, 126.9395)


def test_accom_address_anchor_overrides_selected(make_service, monkeypatch):
    scored = {"tourspot": [], "cafe": [], "restaurant": []}
    monkeypatch.setattr(
        recommend_service,
        "geocode_address",
        lambda addr: {"lat": 37.5123, "lng": 126.9399, "address": addr},
    )
    svc = make_service(scored)
    user = _base_user(
        region="seoul",
        accom_address="서울특별시 동작구 상도로 123",
        selected_places=[PoiRef(place_id=123, category="tourspot", province="seoul")],
    )

    svc.recommend(user, top_k_per_category=1, distance_max_km=5.0)

    assert svc.scorers[0].last_anchor_coords == (37.5123, 126.9399)
