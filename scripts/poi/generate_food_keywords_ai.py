# -*- coding: utf-8 -*-
"""
embedding_json의 카페/레스토랑 데이터를 입력으로 받아 OpenAI API를 호출해
5개 키워드만 추출한 새 JSON 파일을 생성합니다.

제외 규칙: 한식/중식/일식/양식/분식/카페 등 직접적인 음식 분류어는 키워드에 포함하지 않도록 프롬프트에서 차단합니다.

사용 예:
    python scripts/poi/generate_food_keywords_ai.py --category cafe --limit 100
    python scripts/poi/generate_food_keywords_ai.py --category restaurant --model gpt-4o-mini --output data/processed/restaurant_ai_keywords.json
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]  # /AI
EMBED_JSON_DIR = ROOT / "data" / "embedding_json"
OUTPUT_DIR = ROOT / "data" / "processed"

# 음식/태그/사용자 입력 키워드 블랙리스트 (포함 시 제거)
EXCLUDED_TOKENS = [
    # 음식 분류
    "한식",
    "중식",
    "일식",
    "양식",
    "분식",
    "카페",
    "퓨전",
    "세계음식",
    # 테마/분위기 입력값
    "관광",
    "랜드마크",
    "역사",
    "유적지",
    "박물관",
    "미술관",
    "전시",
    "쇼핑",
    "백화점",
    "시장",
    "로컬 상점",
    "명품",
    "자연",
    "공원",
    "호수",
    "해변",
    "등산",
    "섬",
    "체험",
    "액티비티",
    "스파",
    "힐링",
    "야경",
    "감성",
    "포토존",
    "전망대",
    "루프탑",
    "엔터테인먼트",
    "테마파크",
    "클럽",
    "바",
    "라이브",
    "인기많은",
    "고급스러운",
    "세련된",
    "아기자기한",
    "로컬",
    "서민적인",
    "조용한",
    "활기찬",
    "자연친화적인",
    "감성적인",
    "힙한",
    "젊은",
    "가족 친화적",
    "인기 많은곳",
    # 카페/프랜차이즈 태그
    "디저트 위주",
    "커피가 맛있다",
    "분위기 좋은",
    "뷰가 좋은",
    "프렌차이즈",
]

# 붙어 있는 토큰을 분리하기 위한 사전 토큰
TOKEN_SPLIT_PATTERNS = [
    # 맛/공간/서비스 단어들
    "달콤한맛",
    "깊은맛",
    "담백한맛",
    "테라스",
    "예약 가능",
    "예약",
    "포장",
    "테이크아웃",
    "좌식",
    "콘센트",
    "와이파이",
    "창가",
    "혼밥",
    "조용한",
    "아늑한",
    "감성",
    "트렌디한",
]

# 프랜차이즈 이름 패턴
FRANCHISE_KEYWORDS = [
    "스타벅스",
    "이디야",
    "투썸",
    "파스쿠찌",
    "컴포즈",
    "빽다방",
    "메가커피",
    "할리스",
    "던킨",
    "배스킨",
    "맥도날드",
    "버거킹",
    "롯데리아",
    "피자헛",
    "도미노",
    "파파존스",
    "피자마루",
]


def build_prompt(title: str, description: str, existing_keywords: List[str]) -> str:
    excluded = ", ".join(EXCLUDED_TOKENS)
    existing_line = ", ".join(existing_keywords) if existing_keywords else ""
    desc_with_existing = description or ""
    if existing_line:
        desc_with_existing = f"{desc_with_existing} (기존 키워드 참고: {existing_line})"

    return f"""
다음 장소의 이름/설명/현재 키워드를 보고, 장소의 분위기/특징/경험을 나타내는 짧은 키워드 5개를 새로 제안하세요.
- 음식/테마/분위기 분류어({excluded})나 메뉴 장르명은 절대 포함하지 마세요.
- “아늑한, 편안한, cozy, comfortable” 같은 진부한 표현은 피하세요.
- 기존 키워드가 붙어 있거나 의미가 섞여 있으면, 의미 단위로 나누어 자연스러운 키워드 5개를 새로 제안하세요.
- 서로 다른 의미의 키워드 5개를 출력하세요(중복 금지).
- 키워드는 1~3단어의 자연스러운 표현으로 작성하세요. 숫자·예약/포장/점심/저녁 같은 단순 운영 정보는 피하세요.
- 장소의 공간 느낌, 서비스 특징, 이용 맥락 등을 짧게 표현하세요.
- 기존 키워드는 그대로 복사하지 말고, 의미만 참고해 새로 5개를 제안하세요.
- 오직 아래 JSON 한 줄로만 답변하세요. 코드블록(```), 설명, 불필요한 텍스트는 쓰지 마세요.
  {{"keywords": ["...", "...", "...", "...", "..."]}}

이름: {title}
설명 및 기존 키워드 참고: {desc_with_existing}
""".strip()


def _filter_keywords(kws: List[str]) -> List[str]:
    cleaned = []
    for kw in kws:
        k = kw.strip()
        if not k:
            continue
        # 길이가 너무 길거나(>15) 너무 짧은(<2) 경우 제외
        if len(k) > 15 or len(k) < 2:
            continue
        if any(ex in k for ex in EXCLUDED_TOKENS):
            continue
        cleaned.append(k)
    # 중복 제거, 최대 5개
    dedup = []
    for k in cleaned:
        if k not in dedup:
            dedup.append(k)
        if len(dedup) >= 5:
            break
    return dedup


def _parse_existing_keywords(raw: Any) -> List[str]:
    """기존 keywords 필드를 리스트로 변환하고 정제."""
    if raw is None:
        tokens = []
    elif isinstance(raw, list):
        tokens = [str(k).strip() for k in raw if str(k).strip()]
    elif isinstance(raw, str):
        # 요청: 반드시 콤마만 구분자로 사용
        s = raw.replace("\n", " ")
        tokens = [t.strip() for t in s.split(",") if t.strip()]
    else:
        tokens = [str(raw).strip()] if str(raw).strip() else []

    # 길이/블랙리스트/중복 필터 적용
    tokens = _filter_keywords(tokens)
    return tokens


def call_openai(client: OpenAI, model: str, title: str, description: str, existing_keywords: List[str]) -> List[str]:
    prompt = build_prompt(title, description, existing_keywords)
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.4,
        max_output_tokens=200,
    )
    content = resp.output_text.strip()
    # 코드블록 제거
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    try:
        data = json.loads(content)
        kws = data.get("keywords") or []
        # 문자열만 남기고 공백 제거
        kws = [str(k).strip() for k in kws if str(k).strip()]
        return _filter_keywords(kws)
    except Exception:
        # JSON 파싱 실패 시 전체 텍스트를 콤마 기준으로 나눠서 반환
        parts = [p.strip() for p in content.replace("\n", " ").split(",") if p.strip()]
        return _filter_keywords(parts)


def main():
    if load_dotenv:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Generate 5 keywords for cafe/restaurant via OpenAI.")
    parser.add_argument("--category", choices=["cafe", "restaurant"], required=True)
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--input", type=Path, help="기본: data/embedding_json/embedding_<category>.json")
    parser.add_argument("--output", type=Path, help="기본: data/processed/<category>_ai_keywords.json")
    parser.add_argument("--limit", type=int, default=None, help="처리할 상위 N개만 변환")
    parser.add_argument("--verbose", action="store_true", help="처리된 결과를 즉시 출력")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")
    client = OpenAI(api_key=api_key)

    input_path = args.input or (EMBED_JSON_DIR / f"embedding_{args.category}.json")
    output_path = args.output or (OUTPUT_DIR / f"{args.category}_ai_keywords.json")

    data: List[Dict[str, Any]] = json.loads(input_path.read_text())
    if args.limit:
        data = data[: args.limit]

    results = []
    for idx, item in enumerate(data, start=1):
        # 이름 필드는 title/name 어느 쪽이든 받아서 사용
        title = item.get("title") or item.get("name") or ""
        description = item.get("description") or ""
        if idx % 10 == 1:
            print(f"[{idx}/{len(data)}] processing: {title}...")

        # 기존 keywords 파싱
        existing = _parse_existing_keywords(item.get("keywords"))

        # 설명이 없으면 기존 키워드도 사용하지 않음
        if not description or not str(description).strip():
            keywords = []
        else:
            # 프랜차이즈 감지
            lower_title = title.lower()
            is_franchise = any(token.lower() in lower_title for token in FRANCHISE_KEYWORDS)

            if is_franchise:
                keywords = ["프랜차이즈", "테이크아웃", "접근성 좋음", "합리적 가격", "표준화된 메뉴"]
            else:
                keywords = call_openai(client, args.model, title=title, description=description, existing_keywords=existing)

        # AI 키워드만 사용, 최대 5개
        merged = _filter_keywords(keywords)
        merged = merged[:5]

        out_item = {
            "place_id": int(item.get("place_id")),
            "category": item.get("category"),
            "province": item.get("province"),
            "name": item.get("title"),
            "address": item.get("address"),
            "duration": item.get("duration"),
            "description": item.get("description"),
            "images": item.get("imglinks"),
            "latitude": item.get("latitude"),
            "longitude": item.get("longitude"),
            "keywords": merged,
        }
        if args.verbose:
            print(json.dumps(out_item, ensure_ascii=False))
        results.append(out_item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ saved {len(results)} rows -> {output_path}")


if __name__ == "__main__":
    main()
