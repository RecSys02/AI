#!/usr/bin/env python3
# 1~1000
#uv run ../scripts/eval/label_candidates_llm.py --input ../data/eval/filtered/candidates.jsonl --output ../data/eval/labels_top5_0001.jsonl --limit 1000 --offset 0
# 1001~2000
#uv run ../scripts/eval/label_candidates_llm.py --input ../data/eval/filtered/candidates.jsonl --output ../data/eval/labels_top5_0002.jsonl --limit 1000 --offset 1000

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _user_text(user: dict) -> str:
    parts = []
    if user.get("preferredThemes"):
        parts.append(f"선호 테마: {', '.join(user['preferredThemes'])}")
    if user.get("preferredMoods"):
        parts.append(f"선호 분위기: {', '.join(user['preferredMoods'])}")
    if user.get("preferredRestaurantTypes"):
        parts.append(f"선호 음식점: {', '.join(user['preferredRestaurantTypes'])}")
    if user.get("preferredCafeTypes"):
        parts.append(f"선호 카페: {', '.join(user['preferredCafeTypes'])}")
    if user.get("avoid"):
        parts.append(f"회피 요소: {', '.join(user['avoid'])}")
    if user.get("activityLevel"):
        parts.append(f"활동 강도: {user['activityLevel']}")
    if user.get("companion"):
        parts.append(f"동행: {', '.join(user['companion'])}")
    if user.get("budget"):
        parts.append(f"예산: {user['budget']}")
    if user.get("region"):
        parts.append(f"지역: {user['region']}")
    return "\n".join(parts) if parts else "특별한 선호 없음"


def _sanitize_candidates(cands: list[dict], category: str) -> list[dict]:
    cleaned = []
    for cand in cands:
        keywords = cand.get("keywords") or []
        if isinstance(keywords, list):
            keywords = [str(k) for k in keywords if k][:5]
        elif isinstance(keywords, str) and keywords:
            keywords = [keywords]
        else:
            keywords = []
        item = {
            "id": cand.get("id"),
            "name": cand.get("name"),
            "distance_km": cand.get("distance_km"),
            "keywords": keywords,
        }
        if category == "restaurant":
            item["food_type"] = cand.get("food_type")
        elif category == "cafe":
            item["cafe_type"] = cand.get("cafe_type")
        else:
            item["themes"] = cand.get("themes") or cand.get("poi_type")
        cleaned.append(item)
    return cleaned


def _build_prompt(user: dict, category: str, candidates: list[dict], top_k: int) -> str:
    payload = {
        "user": _user_text(user),
        "category": category,
        "top_k": top_k,
        "candidates": _sanitize_candidates(candidates, category),
    }
    return (
        "당신은 여행 추천 평가자입니다.\n"
        "아래 사용자 페르소나에 맞는 후보 중에서 상위 항목을 선택하세요.\n"
        "규칙:\n"
        "- 반드시 후보 id만 선택\n"
        f"- 최대 {top_k}개까지 선택\n"
        "- 적합한 후보가 없으면 빈 배열\n"
        "- JSON만 반환\n\n"
        f"입력:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        "출력 형식:\n"
        '{"userId": 1, "category": "restaurant", "relevant_ids": [101, 202]}'
    )


def _parse_response(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # attempt to extract JSON substring
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Label candidates using LLM (Top-K selection)."
    )
    parser.add_argument(
        "--input",
        default="../data/eval/filtered/candidates.jsonl",
        help="Input candidates JSONL path.",
    )
    parser.add_argument(
        "--output",
        default="../data/eval/labels_top5.jsonl",
        help="Output labels JSONL path.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-K to select.")
    parser.add_argument("--model", default=os.getenv("EVAL_LABEL_MODEL", "gpt-4o-mini"))
    parser.add_argument("--limit", type=int, default=0, help="Max users to label.")
    parser.add_argument("--offset", type=int, default=0, help="Start index offset.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between calls (sec).")
    args = parser.parse_args()

    input_path = Path(args.input)
    records = _load_jsonl(input_path)

    if args.offset > 0:
        records = records[args.offset :]
    if args.limit > 0:
        records = records[: args.limit]

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    outputs = []
    for rec in records:
        user = rec.get("user") or {}
        user_id = user.get("userId")
        candidates = rec.get("candidates", [])
        by_category: dict[str, list[dict]] = {}
        for cand in candidates:
            cat = cand.get("category")
            if not cat:
                continue
            by_category.setdefault(cat, []).append(cand)

        for category, items in by_category.items():
            prompt = _build_prompt(user, category, items, args.top_k)
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "너는 엄격한 평가자다. JSON만 반환한다."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            parsed = _parse_response(content)
            if not parsed:
                parsed = {"userId": user_id, "category": category, "relevant_ids": []}
            rel_ids = parsed.get("relevant_ids", [])
            if not isinstance(rel_ids, list):
                rel_ids = []
            normalized = {
                "userId": user_id,
                "category": category,
                "relevant_ids": rel_ids,
            }
            outputs.append(normalized)
            time.sleep(args.sleep)

    _write_jsonl(Path(args.output), outputs)
    print(f"Wrote {len(outputs)} labels to {args.output}")


if __name__ == "__main__":
    main()
