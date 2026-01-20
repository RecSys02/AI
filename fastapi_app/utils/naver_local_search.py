import json
import os
import urllib.parse
import urllib.request
from typing import List


NAVER_LOCAL_CLIENT_ID = os.getenv("NAVER_LOCAL_CLIENT_ID")
NAVER_LOCAL_CLIENT_SECRET = os.getenv("NAVER_LOCAL_CLIENT_SECRET")


def search_local(query: str, limit: int = 3) -> List[dict]:
    if not query or not NAVER_LOCAL_CLIENT_ID or not NAVER_LOCAL_CLIENT_SECRET:
        return []
    params = urllib.parse.urlencode({"query": query, "display": str(limit)})
    url = f"https://openapi.naver.com/v1/search/local.json?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "X-Naver-Client-Id": NAVER_LOCAL_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_LOCAL_CLIENT_SECRET,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    items = payload.get("items") or []
    results = []
    for item in items:
        results.append(
            {
                "title": item.get("title") or "",
                "address": item.get("address") or "",
                "roadAddress": item.get("roadAddress") or "",
                "category": item.get("category") or "",
            }
        )
    return results
