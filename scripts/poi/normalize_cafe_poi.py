import json
import re
from typing import Optional, Dict, Any


# =========================
# Utils
# =========================

def to_int(x) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def to_float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_review_count(counts: Optional[str]) -> Optional[int]:
    """
    "(6)" -> 6
    """
    if not counts:
        return None
    try:
        return int(counts.strip("()"))
    except ValueError:
        return None


# =========================
# content 파서들
# =========================

def parse_sub_type(content: Optional[str]) -> Optional[str]:
    """
    다음 패턴 모두 지원:
    - 업종세계음식 > 카페/커피숍
    - 세계음식 > 카페/커피숍
    """

    if not content:
        return None

    # 편의시설/영업시간 제거
    text = content
    if "편의시설" in text:
        text = text.split("편의시설", 1)[0]
    if "영업시간" in text:
        text = text.split("영업시간", 1)[-1]

    # '>' 기반 업종 추출
    if ">" in text:
        return text.split(">")[-1].strip()

    return None





def parse_open_time_advanced(content: Optional[str]) -> Dict[str, Any]:
    """
    영업시간 파싱 (안전 버전)
    - 매일
    - 주중 / 주말
    """

    result = {
        "default": {"open": None, "close": None},
        "weekday": {"open": None, "close": None},
        "weekend": {"open": None, "close": None},
        "raw_text": None
    }

    if not content or "영업시간" not in content:
        return result

    # 영업시간 이후 텍스트만 추출 (open 유무 무관)
    m = re.search(r"영업시간\s*open?\s*(.*)", content)
    if not m:
        return result

    part = m.group(1)

    # 업종 앞에서 컷
    if "업종" in part:
        part = part.split("업종", 1)[0].strip()

    result["raw_text"] = part

    # 주중 / 주말
    weekday_match = re.search(r"주중\s*(\d{1,2}:\d{2})\s*~\s*(\d{1,2}:\d{2})", part)
    weekend_match = re.search(r"주말\s*(\d{1,2}:\d{2})\s*~\s*(\d{1,2}:\d{2})", part)

    if weekday_match:
        result["weekday"]["open"] = weekday_match.group(1)
        result["weekday"]["close"] = weekday_match.group(2)

    if weekend_match:
        result["weekend"]["open"] = weekend_match.group(1)
        result["weekend"]["close"] = weekend_match.group(2)

    # 매일 / 단일 시간
    if not weekday_match and not weekend_match:
        default_match = re.search(r"(매일)?\s*(\d{1,2}:\d{2})\s*~\s*(\d{1,2}:\d{2})", part)
        if default_match:
            result["default"]["open"] = default_match.group(2)
            result["default"]["close"] = default_match.group(3)

    return result



# =========================
# Normalizer
# =========================

def normalize_cafe_poi(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "poi_id": raw.get("id"),
        "type": "cafe",
        "sub_type": parse_sub_type(raw.get("content")),

        "name": raw.get("title"),
        "address": raw.get("address"),

        # ✅ 이미지 추가
        "image": raw.get("imglinks"),  # 예: "827309.png"

        "location": {
            "lat": to_float(raw.get("latitude")),
            "lng": to_float(raw.get("longitude"))
        },

        "description": raw.get("description"),
        "attributes": {},
        "stats": {
            "views": to_int(raw.get("views")),
            "likes": to_int(raw.get("likes")),
            "bookmarks": to_int(raw.get("bookmarks")),
            "rating": to_float(raw.get("starts")),
            "review_count": parse_review_count(raw.get("counts"))
        },
    }



# =========================
# Main
# =========================

# =========================
# Main
# =========================

def main():
    input_path = "data/raw/cafe_data.json"
    output_path = "data/interim/cafe_poi_normalized.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized_data = []
    for row in data:
        normalized = normalize_cafe_poi(row)
        normalized_data.append(normalized)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(normalized_data)} rows → {output_path}")


if __name__ == "__main__":
    main()
