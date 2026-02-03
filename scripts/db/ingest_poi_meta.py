# -*- coding: utf-8 -*-
"""
Load embedding JSON into Cloud SQL (Postgres) with FTS.
"""
# 실행법 : python ingest_poi_meta.py --mode tourspot

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import psycopg

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOUR_PATH = ROOT / "data" / "embedding_json" / "embedding_tourspot.json"
DEFAULT_RESTAURANT_PATH = ROOT / "data" / "embedding_json" / "embedding_restaurant.json"
DEFAULT_CAFE_PATH = ROOT / "data" / "embedding_json" / "embedding_cafe.json"

FTS_POS_TAGS = {
    "NNG",
    "NNP",
    "NNB",
    "NR",
    "NP",
    "SL",
    "SH",
    "SN",
    "XR",
    "VV",
    "VA",
    "MAG",
}


@lru_cache(maxsize=1)
def _get_kiwi() -> "Kiwi":
    from kiwipiepy import Kiwi

    return Kiwi()


def tokenize_ko(text: str) -> str:
    if not text:
        return ""
    kiwi = _get_kiwi()
    tokens = [token.form for token in kiwi.tokenize(text) if token.tag in FTS_POS_TAGS]
    return " ".join(tokens)


def join_list(values: List[Any]) -> str:
    if not values:
        return ""
    return ", ".join([str(v).strip() for v in values if str(v).strip()])


def build_embedding_text_tourspot(poi: Dict[str, Any]) -> str:
    parts = []

    header = []
    if poi.get("name"):
        header.append(f"{poi['name']} (카테고리: 관광지)")
    if poi.get("summary_one_sentence"):
        header.append(poi["summary_one_sentence"])
    if header:
        parts.append(". ".join(header))

    themes = join_list(poi.get("themes", []))
    mood = join_list(poi.get("mood", []))
    indoor_outdoor = poi.get("indoor_outdoor")
    line = []
    if themes:
        line.append(f"테마: {themes}")
    if mood:
        line.append(f"분위기: {mood}")
    if indoor_outdoor:
        line.append(f"실내/실외: {indoor_outdoor}")
    if line:
        parts.append(". ".join(line))

    visitor_type = join_list(poi.get("visitor_type", []))
    keywords = join_list(poi.get("keywords", []))
    line = []
    if visitor_type:
        line.append(f"대상: {visitor_type}")
    if keywords:
        line.append(f"키워드: {keywords}")
    if line:
        parts.append(". ".join(line))

    activity = poi.get("activity") or {}
    activity_label = None
    if isinstance(activity, dict):
        activity_label = activity.get("label")
    if not activity_label and isinstance(activity, dict) and activity.get("level") is not None:
        activity_label = activity.get("level")
    best_time = join_list(poi.get("best_time", []))
    ideal_schedule = poi.get("ideal_schedule_position")
    line = []
    if activity_label:
        line.append(f"활동 강도: {activity_label}")
    if best_time:
        line.append(f"추천 시간: {best_time}")
    if ideal_schedule:
        line.append(str(ideal_schedule))
    if line:
        parts.append(". ".join(line))

    if poi.get("photospot") is True:
        parts.append("포토스팟: 있음")

    if not parts:
        pid = poi.get("poi_id", "unknown")
        return f"관광지 {pid}"
    return " ".join(parts)


def build_embedding_text_food(poi: Dict[str, Any]) -> str:
    parts = []
    title = poi.get("title") or poi.get("name")
    if title:
        parts.append(f"이름: {title}")
    if poi.get("category"):
        parts.append(f"카테고리: {poi['category']}")
    if poi.get("content"):
        parts.append(poi["content"])
    if poi.get("description"):
        parts.append(poi["description"])
    kws = join_list(poi.get("keywords", []))
    if kws:
        parts.append(f"키워드: {kws}")
    return " ".join(parts) or (title or "음식/카페")


def get_lat_lng(meta: Dict[str, Any]) -> Tuple[float | None, float | None]:
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


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("input JSON must be a list")
    return data


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS poi_meta (
                place_id BIGINT NOT NULL,
                category TEXT NOT NULL,
                province TEXT,
                name TEXT,
                title TEXT,
                content TEXT,
                description TEXT,
                lat DOUBLE PRECISION,
                lng DOUBLE PRECISION,
                popularity_score DOUBLE PRECISION,
                meta JSONB NOT NULL,
                search_text TEXT,
                search_vector tsvector,
                PRIMARY KEY (category, place_id)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS poi_meta_category_idx ON poi_meta (category);")
        cur.execute("CREATE INDEX IF NOT EXISTS poi_meta_province_idx ON poi_meta (province);")
        cur.execute("CREATE INDEX IF NOT EXISTS poi_meta_search_idx ON poi_meta USING GIN (search_vector);")
        cur.execute("CREATE INDEX IF NOT EXISTS poi_meta_lat_idx ON poi_meta (lat);")
        cur.execute("CREATE INDEX IF NOT EXISTS poi_meta_lng_idx ON poi_meta (lng);")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest POI metadata into Postgres.")
    parser.add_argument("--mode", choices=["tourspot", "cafe", "restaurant"], default="tourspot")
    parser.add_argument("--input", type=Path, help="input JSON path (default by mode)")
    parser.add_argument("--dsn", help="Postgres DSN (or use DATABASE_URL/POSTGRES_DSN)")
    parser.add_argument("--truncate", action="store_true", help="truncate poi_meta before insert")
    args = parser.parse_args()

    mode_config = {
        "tourspot": (DEFAULT_TOUR_PATH, build_embedding_text_tourspot),
        "restaurant": (DEFAULT_RESTAURANT_PATH, build_embedding_text_food),
        "cafe": (DEFAULT_CAFE_PATH, build_embedding_text_food),
    }
    default_input, builder = mode_config[args.mode]
    input_path = args.input or default_input
    dsn = args.dsn or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")
    if not dsn:
        raise ValueError("DATABASE_URL (or POSTGRES_DSN) is not set")

    pois = load_json(input_path)
    print(f"[INFO] mode={args.mode} input={input_path} count={len(pois)}")

    conn = psycopg.connect(dsn)
    conn.autocommit = True
    ensure_schema(conn)
    with conn.cursor() as cur:
        if args.truncate:
            cur.execute("TRUNCATE TABLE poi_meta;")

    rows = []
    for poi in pois:
        try:
            place_id = int(poi.get("place_id"))
        except (TypeError, ValueError):
            continue
        category = poi.get("category") or args.mode
        province = poi.get("province") or poi.get("city")
        name = poi.get("name")
        title = poi.get("title")
        content = poi.get("content")
        description = poi.get("description")
        lat, lng = get_lat_lng(poi)
        popularity = poi.get("popularity_score")
        search_text_raw = builder(poi)
        search_text = tokenize_ko(search_text_raw) or search_text_raw
        rows.append(
            (
                place_id,
                category,
                province,
                name,
                title,
                content,
                description,
                lat,
                lng,
                popularity,
                json.dumps(poi, ensure_ascii=False),
                search_text,
                search_text,
            )
        )

    sql = """
        INSERT INTO poi_meta (
            place_id, category, province, name, title, content, description,
            lat, lng, popularity_score, meta, search_text, search_vector
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s::jsonb, %s,
            to_tsvector('simple', %s)
        )
        ON CONFLICT (category, place_id) DO UPDATE SET
            category = EXCLUDED.category,
            province = EXCLUDED.province,
            name = EXCLUDED.name,
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            description = EXCLUDED.description,
            lat = EXCLUDED.lat,
            lng = EXCLUDED.lng,
            popularity_score = EXCLUDED.popularity_score,
            meta = EXCLUDED.meta,
            search_text = EXCLUDED.search_text,
            search_vector = EXCLUDED.search_vector;
    """

    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    print(f"✅ Upserted {len(rows)} rows into poi_meta")


if __name__ == "__main__":
    main()
