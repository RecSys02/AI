#!/usr/bin/env python3

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "embedding_json" / "embedding_cafe.json"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

VIEW_KEYS = {
    "view",
    "views",
    "viewcnt",
    "viewcount",
    "hit",
    "hitcnt",
    "hitcount",
    "readcount",
}
LIKE_KEYS = {
    "like",
    "likes",
    "likecnt",
    "likecount",
    "goodcnt",
    "goodcount",
}
BOOKMARK_KEYS = {
    "bookmark",
    "bookmarks",
    "bookmarkcnt",
    "bookmarkcount",
    "scrapcnt",
    "scrapcount",
    "zzimcnt",
    "zzimcount",
}
RATING_KEYS = {
    "rating",
    "ratingvalue",
    "averagescore",
    "score",
    "star",
    "starrating",
    "reviewscore",
}
REVIEW_KEYS = {
    "reviewcount",
    "reviewcnt",
    "reviewtotal",
    "commentcount",
    "commentcnt",
    "review",
    "reviews",
}


def _load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, data: List[dict]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _parse_siksin_id(url: str) -> Optional[str]:
    if not url:
        return None
    match = re.search(r"/P/(\d+)", url)
    if match:
        return match.group(1)
    return None


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        if stripped.isdigit():
            return int(stripped)
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if re.match(r"^\d+(\.\d+)?$", stripped):
            return float(stripped)
    return None


def _format_rating(value: float) -> str:
    return f"{value:.1f}"


def _walk_json(node: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from _walk_json(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_json(value)


def _extract_json_ld(html: str) -> List[dict]:
    blobs: List[dict] = []
    for match in re.finditer(
        r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
        html,
        flags=re.S,
    ):
        text = match.group(1).strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            blobs.extend([item for item in parsed if isinstance(item, dict)])
        elif isinstance(parsed, dict):
            blobs.append(parsed)
    return blobs


def _extract_json_blobs(html: str) -> List[dict]:
    blobs = _extract_json_ld(html)
    next_match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
        flags=re.S,
    )
    if next_match:
        try:
            blobs.append(json.loads(next_match.group(1)))
        except json.JSONDecodeError:
            pass
    preloaded_match = re.search(
        r"window\.__PRELOADED_STATE__\s*=\s*({.*?})\s*;",
        html,
        flags=re.S,
    )
    if preloaded_match:
        try:
            blobs.append(json.loads(preloaded_match.group(1)))
        except json.JSONDecodeError:
            pass
    return blobs


def _extract_stats_from_json(obj: dict) -> Dict[str, Optional[float]]:
    views: List[int] = []
    likes: List[int] = []
    bookmarks: List[int] = []
    ratings: List[float] = []
    reviews: List[int] = []

    for key, value in _walk_json(obj):
        key_lower = key.lower() if isinstance(key, str) else ""
        if key_lower in VIEW_KEYS:
            num = _coerce_int(value)
            if num is not None:
                views.append(num)
        elif key_lower in LIKE_KEYS:
            num = _coerce_int(value)
            if num is not None:
                likes.append(num)
        elif key_lower in BOOKMARK_KEYS:
            num = _coerce_int(value)
            if num is not None:
                bookmarks.append(num)
        elif key_lower in RATING_KEYS:
            num = _coerce_float(value)
            if num is not None and 0.0 <= num <= 5.0:
                ratings.append(num)
        elif key_lower in REVIEW_KEYS:
            num = _coerce_int(value)
            if num is not None:
                reviews.append(num)

    return {
        "views": max(views) if views else None,
        "likes": max(likes) if likes else None,
        "bookmarks": max(bookmarks) if bookmarks else None,
        "rating": max(ratings) if ratings else None,
        "review_count": max(reviews) if reviews else None,
    }


def _extract_stats_from_regex(html: str) -> Dict[str, Optional[float]]:
    view_vals = re.findall(r"(?:조회수?|views?)\s*([0-9,]+)", html, flags=re.I)
    like_vals = re.findall(r"(?:좋아요|likes?)\s*([0-9,]+)", html, flags=re.I)
    bookmark_vals = re.findall(
        r"(?:북마크|즐겨찾기|찜|스크랩|bookmarks?)\s*([0-9,]+)",
        html,
        flags=re.I,
    )
    review_vals = re.findall(
        r"(?:리뷰|후기|reviews?)(?:</?[^>]+>|\s|&nbsp;|&#160;|\(|:)*([0-9,]+)",
        html,
        flags=re.I,
    )
    rating_vals = re.findall(
        r"(?:평점|별점|rating)(?:</?[^>]+>|\s|&nbsp;|&#160;|\(|:)*([0-9]+(?:\.[0-9]+)?)",
        html,
        flags=re.I,
    )

    views = [v for v in (_coerce_int(x) for x in view_vals) if v is not None] if view_vals else []
    likes = [v for v in (_coerce_int(x) for x in like_vals) if v is not None] if like_vals else []
    bookmarks = [v for v in (_coerce_int(x) for x in bookmark_vals) if v is not None] if bookmark_vals else []
    reviews = [v for v in (_coerce_int(x) for x in review_vals) if v is not None] if review_vals else []
    ratings = [
        v
        for v in (_coerce_float(x) for x in rating_vals)
        if v is not None and 0.0 <= v <= 5.0
    ] if rating_vals else []

    return {
        "views": max(views) if views else None,
        "likes": max(likes) if likes else None,
        "bookmarks": max(bookmarks) if bookmarks else None,
        "rating": max(ratings) if ratings else None,
        "review_count": max(reviews) if reviews else None,
    }


def _merge_stats(
    base: Dict[str, Optional[float]],
    extra: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    merged = dict(base)
    for key, value in extra.items():
        if merged.get(key) is None and value is not None:
            merged[key] = value
    return merged


def _parse_stats(html: str) -> Dict[str, Optional[float]]:
    stats = {
        "views": None,
        "likes": None,
        "bookmarks": None,
        "rating": None,
        "review_count": None,
    }
    for blob in _extract_json_blobs(html):
        stats = _merge_stats(stats, _extract_stats_from_json(blob))
    stats = _merge_stats(stats, _extract_stats_from_regex(html))
    return stats


def _fetch_html_requests(session: requests.Session, url: str, timeout: float) -> Optional[str]:
    try:
        response = session.get(url, timeout=timeout)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    return response.text


class PlaywrightFetcher:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def _start(self) -> bool:
        if not self.enabled:
            return False
        if self._page is not None:
            return True
        try:
            from playwright.sync_api import sync_playwright
        except Exception:
            print("[playwright] not installed; skipping fallback", file=sys.stderr)
            self.enabled = False
            return False
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._context = self._browser.new_context(
            user_agent=USER_AGENT,
            locale="ko-KR",
        )
        self._page = self._context.new_page()
        return True

    def fetch(self, url: str, timeout: float) -> Optional[str]:
        if not self._start():
            return None
        try:
            self._page.goto(url, wait_until="networkidle", timeout=int(timeout * 1000))
            return self._page.content()
        except Exception:
            return None

    def close(self) -> None:
        if self._page is not None:
            try:
                self._page.close()
            except Exception:
                pass
        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _should_skip_only_missing(rec: dict, only_missing: bool) -> bool:
    if not only_missing:
        return False
    fields = ["views", "likes", "bookmarks", "counts", "starts"]
    return all(not _is_missing(rec.get(field)) for field in fields)


def _update_fields(rec: dict, stats: Dict[str, Optional[float]]) -> bool:
    updated = False
    if stats.get("views") is not None:
        value = int(stats["views"])
        if rec.get("views") != value:
            rec["views"] = value
            updated = True
    if stats.get("likes") is not None:
        value = int(stats["likes"])
        if rec.get("likes") != value:
            rec["likes"] = value
            updated = True
    if stats.get("bookmarks") is not None:
        value = int(stats["bookmarks"])
        if rec.get("bookmarks") != value:
            rec["bookmarks"] = value
            updated = True
    if stats.get("review_count") is not None:
        value = str(int(stats["review_count"]))
        if rec.get("counts") != value:
            rec["counts"] = value
            updated = True
    if stats.get("rating") is not None:
        value = _format_rating(float(stats["rating"]))
        if rec.get("starts") != value:
            rec["starts"] = value
            updated = True
    return updated


def _dump_html(dump_dir: Path, place_id: str, source: str, html: Optional[str]) -> None:
    if not dump_dir or not html:
        return
    dump_dir.mkdir(parents=True, exist_ok=True)
    safe_place_id = re.sub(r"[^0-9A-Za-z_-]", "_", place_id)
    path = dump_dir / f"{safe_place_id}_{source}.html"
    try:
        path.write_text(html, encoding="utf-8")
    except Exception:
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update Siksin stats (views/likes/bookmarks/reviews/ratings) in JSON."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input JSON path.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path. If empty, overwrite input.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max records to process (0 = all).")
    parser.add_argument("--offset", type=int, default=0, help="Start offset.")
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep between requests.")
    parser.add_argument("--timeout", type=float, default=15.0, help="Request timeout in seconds.")
    parser.add_argument(
        "--playwright",
        action="store_true",
        help="Enable Playwright fallback for JS-rendered pages.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Update only when any field is missing.",
    )
    parser.add_argument(
        "--dump-dir",
        default="",
        help="Dump HTML for failed parses to this directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose fetch information.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write output file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    data = _load_json(input_path)
    total = len(data)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        }
    )

    fetcher = PlaywrightFetcher(args.playwright)
    processed = 0
    updated = 0
    skipped = 0
    failed = 0
    dump_dir = Path(args.dump_dir) if args.dump_dir else None

    start = args.offset
    end = start + args.limit if args.limit else total

    try:
        for idx in range(start, min(end, total)):
            rec = data[idx]
            if _should_skip_only_missing(rec, args.only_missing):
                skipped += 1
                continue

            url = rec.get("links")
            if not url:
                skipped += 1
                continue

            siksin_id = _parse_siksin_id(url) or str(rec.get("place_id", idx))
            if args.verbose:
                print(f"[{idx+1}/{total}] fetch url={url} place_id={siksin_id}")

            html = _fetch_html_requests(session, url, args.timeout)
            if args.verbose and not html:
                print(f"[{idx+1}/{total}] requests fetch failed", file=sys.stderr)
            stats = _parse_stats(html) if html else {
                "views": None,
                "likes": None,
                "bookmarks": None,
                "rating": None,
                "review_count": None,
            }
            if args.playwright and any(v is None for v in stats.values()):
                pw_html = fetcher.fetch(url, args.timeout)
                if pw_html:
                    stats = _merge_stats(stats, _parse_stats(pw_html))
                if args.verbose and not pw_html:
                    print(f"[{idx+1}/{total}] playwright fetch failed", file=sys.stderr)
                if dump_dir is not None and any(v is None for v in stats.values()):
                    _dump_html(dump_dir, siksin_id, "playwright", pw_html)

            if any(v is not None for v in stats.values()):
                if _update_fields(rec, stats):
                    updated += 1
                processed += 1
                print(
                    f"[{idx+1}/{total}] place_id={siksin_id} "
                    f"views={stats.get('views')} "
                    f"likes={stats.get('likes')} "
                    f"bookmarks={stats.get('bookmarks')} "
                    f"reviews={stats.get('review_count')} "
                    f"rating={stats.get('rating')}"
                )
            else:
                failed += 1
                if dump_dir is not None:
                    _dump_html(dump_dir, siksin_id, "requests", html)
                print(f"[{idx+1}/{total}] place_id={siksin_id} failed", file=sys.stderr)

            if args.sleep > 0:
                time.sleep(args.sleep + random.uniform(0, args.sleep))
    finally:
        fetcher.close()

    print(
        f"Done. processed={processed} updated={updated} skipped={skipped} failed={failed}"
    )

    if args.dry_run:
        print("Dry run: output not written.")
        return

    _write_json_atomic(output_path, data)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
