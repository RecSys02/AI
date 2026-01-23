#!/usr/bin/env python3
#
'''
scripts.eval.run_recommend_sample의 Docstring

uv run python ../scripts/eval/run_recommend_sample.py \
  --users ../data/eval/samples/users_100_from_labels.jsonl \
  --output ../data/eval/recommend_top10_rerank_gemini3_sample100.jsonl \
  --limit 100 --rerank

'''
# rerank with LLM: --rerank
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "fastapi_app"))

from models.user_input import UserInput  # noqa: E402
from services.recommend_service import RecommendService  # noqa: E402


def _no_rerank(self, user, category, candidates, top_k=10, debug=False):
    result = [{k: v for k, v in item.items() if k != "_meta"} for item in candidates[:top_k]]
    if debug:
        for idx, item in enumerate(result):
            item["rank_before_rerank"] = idx + 1
            item["rank_after_rerank"] = idx + 1
    return result


def _load_users(path: Path, limit: int) -> list[UserInput]:
    users: list[UserInput] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            try:
                user = UserInput.model_validate(data)
            except AttributeError:
                user = UserInput.parse_obj(data)
            users.append(user)
            if limit and len(users) >= limit:
                break
    return users


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Run RecommendService for a sample (optional LLM rerank)."
    )
    parser.add_argument(
        "--users",
        default=str(PROJECT_ROOT / "data" / "eval" / "users_5000.jsonl"),
        help="Input users JSONL path.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "eval" / "recommend_top10_rerank_gemini_3.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Max users to run.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k per category.")
    parser.add_argument("--distance-km", type=float, default=3.0, help="Distance limit.")
    parser.add_argument("--debug", action="store_true", help="Include debug ranking.")
    parser.add_argument("--rerank", action="store_true", help="Enable LLM rerank.")
    args = parser.parse_args()

    svc = RecommendService()
    if not args.rerank:
        os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
        svc._llm_rerank = _no_rerank.__get__(svc, RecommendService)
    else:
        provider = os.getenv("RERANK_PROVIDER", "openai").lower()
        if provider == "gemini":
            if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
                raise SystemExit(
                    "GEMINI_API_KEY (or GOOGLE_API_KEY) is required when --rerank is enabled."
                )
        elif not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is required when --rerank is enabled.")

    users_path = Path(args.users)
    users = _load_users(users_path, args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for user in users:
            recs = svc.recommend(
                user,
                top_k_per_category=args.top_k,
                distance_max_km=args.distance_km,
                debug=args.debug,
            )
            record = {"userId": user.user_id, "recommendations": recs}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(users)} records to {output_path}")


if __name__ == "__main__":
    main()
