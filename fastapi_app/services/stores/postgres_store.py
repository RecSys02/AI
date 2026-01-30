import os
from functools import lru_cache
from typing import Iterable, Optional

import psycopg
from psycopg.rows import dict_row


class PostgresStore:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._conn: Optional[psycopg.Connection] = None

    def _connect(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._dsn, row_factory=dict_row)
            self._conn.autocommit = True
        return self._conn

    def fetch_meta(self, place_ids: Iterable[int], category: str) -> dict[int, dict]:
        ids = [int(pid) for pid in place_ids]
        if not ids:
            return {}
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT place_id, category, province, lat, lng, popularity_score, meta
                FROM poi_meta
                WHERE category = %s AND place_id = ANY(%s)
                """,
                (category, ids),
            )
            rows = cur.fetchall()
        return {int(row["place_id"]): row for row in rows}

    def fetch_names(self, place_ids: Iterable[int], category: str) -> list[str]:
        ids = [int(pid) for pid in place_ids]
        if not ids:
            return []
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT place_id, COALESCE(meta->>'name', meta->>'title', '') AS name
                FROM poi_meta
                WHERE category = %s AND place_id = ANY(%s)
                """,
                (category, ids),
            )
            rows = cur.fetchall()
        id_to_name = {int(row["place_id"]): (row.get("name") or "") for row in rows}
        return [id_to_name[pid] for pid in ids if id_to_name.get(pid)]

    def fts_scores(
        self,
        query_text: str,
        category: str,
        limit: int,
        place_ids: Optional[Iterable[int]] = None,
    ) -> dict[int, float]:
        if not query_text:
            return {}
        conn = self._connect()
        ids = [int(pid) for pid in place_ids] if place_ids else []
        params = [query_text, category]
        where = ["category = %s", "search_vector @@ query.q"]
        if ids:
            where.append("place_id = ANY(%s)")
            params.append(ids)
        sql = f"""
            WITH query AS (SELECT plainto_tsquery('simple', %s) AS q)
            SELECT place_id, ts_rank_cd(search_vector, query.q) AS score
            FROM poi_meta, query
            WHERE {" AND ".join(where)}
            ORDER BY score DESC
            LIMIT %s
        """
        params.append(limit)
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return {int(row["place_id"]): float(row["score"]) for row in rows}


@lru_cache(maxsize=1)
def get_postgres_store() -> PostgresStore:
    dsn = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")
    if not dsn:
        raise ValueError("DATABASE_URL (or POSTGRES_DSN) is not set")
    return PostgresStore(dsn=dsn)
