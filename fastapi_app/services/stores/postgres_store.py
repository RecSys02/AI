from contextlib import asynccontextmanager
import os
import re
import time
from functools import lru_cache
from typing import Iterable, Optional

import psycopg
from psycopg import errors as pg_errors
from psycopg.rows import dict_row

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

SYNONYM_CACHE_TTL_SEC = 300
MAX_SYNONYMS_PER_TOKEN = 8


@lru_cache(maxsize=1)
def _get_kiwi() -> "Kiwi":
    from kiwipiepy import Kiwi

    return Kiwi()


def _tokenize_ko(text: str) -> str:
    if not text:
        return ""
    kiwi = _get_kiwi()
    tokens = [token.form for token in kiwi.tokenize(text) if token.tag in FTS_POS_TAGS]
    return " ".join(tokens)


def _normalize_token(token: str) -> str:
    if not token:
        return ""
    cleaned = re.sub(r"[^0-9a-zA-Z가-힣_]", "", token.strip().lower())
    return cleaned


def _split_tokens(text: str) -> list[str]:
    tokenized = _tokenize_ko(text)
    raw_tokens = tokenized.split() if tokenized else re.split(r"\s+", text.strip())
    tokens = []
    for raw in raw_tokens:
        normalized = _normalize_token(raw)
        if normalized:
            tokens.append(normalized)
    return tokens


class PostgresStore:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._conn: Optional[psycopg.Connection] = None
        self._synonym_cache: dict[Optional[str], dict[str, list[str]]] = {}
        self._synonym_cache_at = 0.0

    def _connect(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._dsn, row_factory=dict_row)
            self._conn.autocommit = True
        return self._conn

    @asynccontextmanager
    async def async_connection(self):
        conn = await psycopg.AsyncConnection.connect(
            self._dsn,
            autocommit=True,
            row_factory=dict_row,
        )
        try:
            yield conn
        finally:
            await conn.close()

    def _cache_synonyms(self, rows: list[dict], now: float) -> dict[Optional[str], dict[str, list[str]]]:
        groups: dict[tuple[Optional[str], str], set[str]] = {}
        for row in rows:
            mode = row.get("mode")
            group_key = row.get("group_key")
            term = _normalize_token(row.get("term") or "")
            if not term or not group_key:
                continue
            key = (mode, group_key)
            groups.setdefault(key, set()).add(term)
        term_map_by_mode: dict[Optional[str], dict[str, list[str]]] = {}
        for (mode, _group_key), terms in groups.items():
            term_list = sorted(terms)
            mode_key = mode if mode else None
            term_map = term_map_by_mode.setdefault(mode_key, {})
            for term in term_list:
                term_map[term] = term_list
        self._synonym_cache = term_map_by_mode
        self._synonym_cache_at = now
        return term_map_by_mode

    def _load_synonyms(self) -> dict[Optional[str], dict[str, list[str]]]:
        now = time.monotonic()
        if self._synonym_cache and (now - self._synonym_cache_at) < SYNONYM_CACHE_TTL_SEC:
            return self._synonym_cache
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT mode, group_key, term
                    FROM fts_synonym
                    WHERE enabled = TRUE
                    """
                )
                rows = cur.fetchall()
        except pg_errors.UndefinedTable:
            self._synonym_cache = {}
            self._synonym_cache_at = now
            return self._synonym_cache
        return self._cache_synonyms(rows, now)

    async def _load_synonyms_async(
        self,
        conn: Optional[psycopg.AsyncConnection] = None,
    ) -> dict[Optional[str], dict[str, list[str]]]:
        now = time.monotonic()
        if self._synonym_cache and (now - self._synonym_cache_at) < SYNONYM_CACHE_TTL_SEC:
            return self._synonym_cache
        if conn is None:
            async with self.async_connection() as new_conn:
                return await self._load_synonyms_async(conn=new_conn)
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT mode, group_key, term
                    FROM fts_synonym
                    WHERE enabled = TRUE
                    """
                )
                rows = await cur.fetchall()
        except pg_errors.UndefinedTable:
            self._synonym_cache = {}
            self._synonym_cache_at = now
            return self._synonym_cache
        return self._cache_synonyms(rows, now)

    def _synonym_map_for_mode(self, mode: str) -> dict[str, list[str]]:
        term_map_by_mode = self._load_synonyms()
        merged: dict[str, list[str]] = {}
        global_map = term_map_by_mode.get(None) or {}
        merged.update(global_map)
        mode_map = term_map_by_mode.get(mode) or {}
        merged.update(mode_map)
        return merged

    async def _synonym_map_for_mode_async(
        self,
        mode: str,
        conn: Optional[psycopg.AsyncConnection] = None,
    ) -> dict[str, list[str]]:
        term_map_by_mode = await self._load_synonyms_async(conn=conn)
        merged: dict[str, list[str]] = {}
        global_map = term_map_by_mode.get(None) or {}
        merged.update(global_map)
        mode_map = term_map_by_mode.get(mode) or {}
        merged.update(mode_map)
        return merged

    def _build_tsquery(self, query_text: str, mode: str) -> str:
        tokens = _split_tokens(query_text)
        if not tokens:
            return ""
        synonym_map = self._synonym_map_for_mode(mode)
        groups = []
        for token in tokens:
            terms = synonym_map.get(token)
            if terms:
                clean_terms = []
                for term in terms:
                    normalized = _normalize_token(term)
                    if normalized and normalized not in clean_terms:
                        clean_terms.append(normalized)
                    if len(clean_terms) >= MAX_SYNONYMS_PER_TOKEN:
                        break
                if token not in clean_terms:
                    clean_terms.insert(0, token)
            else:
                clean_terms = [token]
            if len(clean_terms) == 1:
                groups.append(clean_terms[0])
            else:
                groups.append("(" + " | ".join(clean_terms) + ")")
        return " & ".join(groups)

    async def _build_tsquery_async(
        self,
        query_text: str,
        mode: str,
        conn: Optional[psycopg.AsyncConnection] = None,
    ) -> str:
        tokens = _split_tokens(query_text)
        if not tokens:
            return ""
        synonym_map = await self._synonym_map_for_mode_async(mode, conn=conn)
        groups = []
        for token in tokens:
            terms = synonym_map.get(token)
            if terms:
                clean_terms = []
                for term in terms:
                    normalized = _normalize_token(term)
                    if normalized and normalized not in clean_terms:
                        clean_terms.append(normalized)
                    if len(clean_terms) >= MAX_SYNONYMS_PER_TOKEN:
                        break
                if token not in clean_terms:
                    clean_terms.insert(0, token)
            else:
                clean_terms = [token]
            if len(clean_terms) == 1:
                groups.append(clean_terms[0])
            else:
                groups.append("(" + " | ".join(clean_terms) + ")")
        return " & ".join(groups)

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

    async def fetch_meta_async(
        self,
        place_ids: Iterable[int],
        category: str,
        conn: Optional[psycopg.AsyncConnection] = None,
    ) -> dict[int, dict]:
        ids = [int(pid) for pid in place_ids]
        if not ids:
            return {}
        if conn is None:
            async with self.async_connection() as new_conn:
                return await self.fetch_meta_async(ids, category=category, conn=new_conn)
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT place_id, category, province, lat, lng, popularity_score, meta
                FROM poi_meta
                WHERE category = %s AND place_id = ANY(%s)
                """,
                (category, ids),
            )
            rows = await cur.fetchall()
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

    async def fetch_names_async(
        self,
        place_ids: Iterable[int],
        category: str,
        conn: Optional[psycopg.AsyncConnection] = None,
    ) -> list[str]:
        ids = [int(pid) for pid in place_ids]
        if not ids:
            return []
        if conn is None:
            async with self.async_connection() as new_conn:
                return await self.fetch_names_async(ids, category=category, conn=new_conn)
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT place_id, COALESCE(meta->>'name', meta->>'title', '') AS name
                FROM poi_meta
                WHERE category = %s AND place_id = ANY(%s)
                """,
                (category, ids),
            )
            rows = await cur.fetchall()
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
        tsquery = self._build_tsquery(query_text, category)
        if not tsquery:
            return {}
        conn = self._connect()
        ids = [int(pid) for pid in place_ids] if place_ids else []
        params = [tsquery, category]
        where = ["category = %s", "search_vector @@ query.q"]
        if ids:
            where.append("place_id = ANY(%s)")
            params.append(ids)
        sql = f"""
            WITH query AS (SELECT to_tsquery('simple', %s) AS q)
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

    async def fts_scores_async(
        self,
        query_text: str,
        category: str,
        limit: int,
        place_ids: Optional[Iterable[int]] = None,
        conn: Optional[psycopg.AsyncConnection] = None,
    ) -> dict[int, float]:
        if not query_text:
            return {}
        if conn is None:
            async with self.async_connection() as new_conn:
                return await self.fts_scores_async(
                    query_text,
                    category,
                    limit,
                    place_ids=place_ids,
                    conn=new_conn,
                )
        tsquery = await self._build_tsquery_async(query_text, category, conn=conn)
        if not tsquery:
            return {}
        ids = [int(pid) for pid in place_ids] if place_ids else []
        params = [tsquery, category]
        where = ["category = %s", "search_vector @@ query.q"]
        if ids:
            where.append("place_id = ANY(%s)")
            params.append(ids)
        sql = f"""
            WITH query AS (SELECT to_tsquery('simple', %s) AS q)
            SELECT place_id, ts_rank_cd(search_vector, query.q) AS score
            FROM poi_meta, query
            WHERE {" AND ".join(where)}
            ORDER BY score DESC
            LIMIT %s
        """
        params.append(limit)
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return {int(row["place_id"]): float(row["score"]) for row in rows}


@lru_cache(maxsize=1)
def get_postgres_store() -> PostgresStore:
    dsn = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")
    if not dsn:
        raise ValueError("DATABASE_URL (or POSTGRES_DSN) is not set")
    return PostgresStore(dsn=dsn)
