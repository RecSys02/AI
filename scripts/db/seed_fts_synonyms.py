# -*- coding: utf-8 -*-
"""
Seed FTS synonyms into Postgres.
"""

import argparse
import ast
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import psycopg

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RETRIEVER_PATH = ROOT / "fastapi_app" / "services" / "retriever.py"
DEFAULT_CSV_PATH = ROOT / "data" / "fts_synonyms.csv"


def _load_from_retriever(path: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    synonyms = None
    groups = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SYNONYMS":
                    synonyms = ast.literal_eval(node.value)
                if isinstance(target, ast.Name) and target.id == "KEYWORD_GROUPS_BY_MODE":
                    groups = ast.literal_eval(node.value)
    if synonyms is None:
        raise ValueError("SYNONYMS not found in retriever.py")
    if groups is None:
        groups = {}
    return synonyms, groups


def _build_rows(
    synonyms: Dict[str, List[str]],
    groups: Dict[str, List[str]],
) -> List[Tuple[str | None, str, str]]:
    rows: List[Tuple[str | None, str, str]] = []
    if groups:
        for mode, group_keys in groups.items():
            for group_key in group_keys:
                for term in synonyms.get(group_key, []):
                    rows.append((mode, group_key, term))
    else:
        for group_key, terms in synonyms.items():
            for term in terms:
                rows.append((None, group_key, term))
    return rows


def _load_from_csv(path: Path) -> List[Tuple[str | None, str, str]]:
    if not path.exists():
        raise ValueError(f"CSV not found: {path}")
    rows: List[Tuple[str | None, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV must have a header row")
        for row in reader:
            mode = (row.get("mode") or "").strip() or None
            group_key = (row.get("group_key") or "").strip()
            term = (row.get("term") or "").strip()
            enabled_raw = (row.get("enabled") or "true").strip().lower()
            enabled = enabled_raw in {"1", "true", "yes", "y", "t"}
            if not enabled or not group_key or not term:
                continue
            rows.append((mode, group_key, term))
    return rows


def _ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fts_synonym (
                id BIGSERIAL PRIMARY KEY,
                mode TEXT,
                group_key TEXT NOT NULL,
                term TEXT NOT NULL,
                enabled BOOLEAN NOT NULL DEFAULT TRUE,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (mode, group_key, term)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS fts_synonym_mode_group_idx ON fts_synonym (mode, group_key);")
        cur.execute("CREATE INDEX IF NOT EXISTS fts_synonym_term_idx ON fts_synonym (term);")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed FTS synonyms.")
    parser.add_argument("--source", choices=["csv", "retriever"], default="csv")
    parser.add_argument("--retriever", type=Path, default=DEFAULT_RETRIEVER_PATH)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--dsn", help="Postgres DSN (or use DATABASE_URL/POSTGRES_DSN)")
    parser.add_argument("--truncate", action="store_true", help="truncate fts_synonym before insert")
    args = parser.parse_args()

    dsn = args.dsn or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")
    if not dsn:
        raise ValueError("DATABASE_URL (or POSTGRES_DSN) is not set")

    if args.source == "retriever":
        synonyms, groups = _load_from_retriever(args.retriever)
        rows = _build_rows(synonyms, groups)
    else:
        rows = _load_from_csv(args.csv)

    if not rows:
        raise ValueError("no synonyms found to insert")

    conn = psycopg.connect(dsn)
    conn.autocommit = True
    _ensure_table(conn)
    with conn.cursor() as cur:
        if args.truncate:
            cur.execute("TRUNCATE TABLE fts_synonym;")
        cur.executemany(
            """
            INSERT INTO fts_synonym (mode, group_key, term)
            VALUES (%s, %s, %s)
            ON CONFLICT (mode, group_key, term) DO NOTHING
            """,
            rows,
        )
    print(f"✅ Upserted {len(rows)} rows into fts_synonym")


if __name__ == "__main__":
    main()
