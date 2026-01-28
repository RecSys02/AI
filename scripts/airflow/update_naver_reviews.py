#!/usr/bin/env python3

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "embedding_json" / "embedding_tourspot.json"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

VISITOR_KEYS = {"visitorReviewCount", "visitorReviews"}
BLOG_KEYS = {"blogReviewCount", "blogReviews"}
RATING_KEYS = {"averageScore", "rating", "starRating", "reviewScore", "score"}


def _load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, data: List[dict]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _parse_place_id(entry_url: Optional[str]) -> Optional[str]:
    if not entry_url:
        return None
    match = re.search(r"/place/(\d+)", entry_url)
    if match:
        return match.group(1)
    match = re.search(r"placeId=(\d+)", entry_url)
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


def _walk_json(node: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from _walk_json(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_json(value)


def _extract_json_blobs(html: str) -> List[dict]:
    blobs: List[dict] = []
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
    apollo_match = re.search(
        r"window\.__APOLLO_STATE__\s*=\s*({.*?})\s*;</script>",
        html,
        flags=re.S,
    )
    if apollo_match:
        try:
            blobs.append(json.loads(apollo_match.group(1)))
        except json.JSONDecodeError:
            pass
    return blobs


def _extract_stats_from_json(obj: dict) -> Dict[str, Optional[float]]:
    visitor_candidates: List[int] = []
    blog_candidates: List[int] = []
    rating_candidates: List[float] = []

    for key, value in _walk_json(obj):
        if key in VISITOR_KEYS:
            num = _coerce_int(value)
            if num is not None:
                visitor_candidates.append(num)
        elif key in BLOG_KEYS:
            num = _coerce_int(value)
            if num is not None:
                blog_candidates.append(num)
        elif key in RATING_KEYS:
            num = _coerce_float(value)
            if num is not None and 0.0 <= num <= 5.0:
                rating_candidates.append(num)

    return {
        "rating": max(rating_candidates) if rating_candidates else None,
        "visitor_reviews": max(visitor_candidates) if visitor_candidates else None,
        "blog_reviews": max(blog_candidates) if blog_candidates else None,
    }


def _extract_stats_from_regex(html: str) -> Dict[str, Optional[float]]:
    visitor = re.findall(r'"visitorReviewCount"\s*:\s*(\d+)', html)
    blog = re.findall(r'"blogReviewCount"\s*:\s*(\d+)', html)
    rating = re.findall(
        r'"(?:averageScore|rating|starRating|reviewScore|score)"\s*:\s*([0-9]*\.?[0-9]+)',
        html,
    )
    visitor_text = re.findall(r"방문자\s*리뷰\s*([0-9,]+)", html)
    blog_text = re.findall(r"블로그\s*리뷰\s*([0-9,]+)", html)
    rating_text = re.findall(r"(?:별점|평점)\s*([0-9]*\.?[0-9]+)", html)
    rating_html_text = re.findall(
        r"(?:별점|평점)(?:</?[^>]+>|\s|&nbsp;|&#160;)*([0-9]+(?:\.[0-9]+)?)",
        html,
    )

    if visitor_text:
        visitor.extend(visitor_text)
    if blog_text:
        blog.extend(blog_text)
    if rating_text:
        rating.extend(rating_text)
    if rating_html_text:
        rating.extend(rating_html_text)

    visitor_vals = [v for v in (_coerce_int(x) for x in visitor) if v is not None] if visitor else []
    blog_vals = [v for v in (_coerce_int(x) for x in blog) if v is not None] if blog else []
    rating_vals = [
        v
        for v in (_coerce_float(x) for x in rating)
        if v is not None and 0.0 <= v <= 5.0
    ] if rating else []
    return {
        "rating": max(rating_vals) if rating_vals else None,
        "visitor_reviews": max(visitor_vals) if visitor_vals else None,
        "blog_reviews": max(blog_vals) if blog_vals else None,
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
    stats = {"rating": None, "visitor_reviews": None, "blog_reviews": None}
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


def _should_skip_only_missing(naver: dict, only_missing: bool) -> bool:
    if not only_missing:
        return False
    has_rating = naver.get("naver_rating") is not None
    has_visitor = naver.get("naver_visitor_reviews") is not None
    has_blog = naver.get("naver_blog_reviews") is not None
    return has_rating and has_visitor and has_blog


def _update_naver_fields(naver: dict, stats: Dict[str, Optional[float]]) -> bool:
    updated = False
    if stats.get("rating") is not None:
        naver["naver_rating"] = float(stats["rating"])
        updated = True
    if stats.get("visitor_reviews") is not None:
        naver["naver_visitor_reviews"] = int(stats["visitor_reviews"])
        updated = True
    if stats.get("blog_reviews") is not None:
        naver["naver_blog_reviews"] = int(stats["blog_reviews"])
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
        description="Update Naver ratings/reviews in embedding_tourspot.json."
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
            naver = rec.get("naver")
            if not isinstance(naver, dict):
                skipped += 1
                continue
            if _should_skip_only_missing(naver, args.only_missing):
                skipped += 1
                continue

            place_id = naver.get("place_id") or _parse_place_id(naver.get("naver_entry_url"))
            entry_url = naver.get("naver_entry_url")
            if not place_id and not entry_url:
                skipped += 1
                continue
            url = f"https://m.place.naver.com/place/{place_id}/home" if place_id else entry_url

            if args.verbose:
                print(f"[{idx+1}/{total}] fetch url={url} place_id={place_id}")
            html = _fetch_html_requests(session, url, args.timeout)
            if args.verbose and not html:
                print(f"[{idx+1}/{total}] requests fetch failed", file=sys.stderr)
            stats = _parse_stats(html) if html else {"rating": None, "visitor_reviews": None, "blog_reviews": None}
            if args.playwright and any(v is None for v in stats.values()):
                pw_html = fetcher.fetch(url, args.timeout)
                if pw_html:
                    stats = _merge_stats(stats, _parse_stats(pw_html))
                if args.verbose and not pw_html:
                    print(f"[{idx+1}/{total}] playwright fetch failed", file=sys.stderr)
                if dump_dir is not None and any(v is None for v in stats.values()):
                    _dump_html(dump_dir, str(place_id or idx), "playwright", pw_html)

            if any(v is not None for v in stats.values()):
                if _update_naver_fields(naver, stats):
                    updated += 1
                processed += 1
                print(
                    f"[{idx+1}/{total}] place_id={place_id} "
                    f"rating={stats.get('rating')} "
                    f"visitor={stats.get('visitor_reviews')} "
                    f"blog={stats.get('blog_reviews')}"
                )
            else:
                failed += 1
                if dump_dir is not None:
                    _dump_html(dump_dir, str(place_id or idx), "requests", html)
                print(f"[{idx+1}/{total}] place_id={place_id} failed", file=sys.stderr)

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
