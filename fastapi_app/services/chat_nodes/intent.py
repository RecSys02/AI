import re

from utils.geo import normalize_text

REGION_SUFFIXES = ("도", "시", "군", "구", "읍", "면", "동", "리", "가", "역", "로", "길")
REGION_TOPIC_HINTS = ("여행", "일정", "계획", "코스", "맛집", "식당", "카페", "관광", "놀거리", "명소", "장소")


def _has_region_token(query: str) -> bool:
    q = str(query or "").strip()
    if not q:
        return False

    words = re.findall(r"[0-9A-Za-z가-힣]+", q)
    for word in words:
        if len(word) >= 2 and any(word.endswith(suffix) for suffix in REGION_SUFFIXES):
            return True

    if re.search(r"[0-9A-Za-z가-힣]{2,}\s*(?:에서|으로|쪽|근처|주변|인근)", q):
        return True
    if re.search(r"[0-9A-Za-z가-힣]{2,}\s*(?:->|→|⇒|➡)\s*[0-9A-Za-z가-힣]{2,}", q):
        return True

    for topic in REGION_TOPIC_HINTS:
        if re.search(rf"[0-9A-Za-z가-힣]{{2,}}\s*{topic}", q):
            return True
    return False


def is_region_non_recommend_query(query: str) -> bool:
    q = str(query or "").strip()
    if not q:
        return False
    if "추천" in q:
        return False
    return _has_region_token(q)


def detect_intent(query: str) -> str:
    q = query.lower()
    # 추천 의도가 포함된 경우 "recommend" 반환
    recommend_terms = ["추천", "어디", "가볼", "뭐가 있어", "top", "best", "3개", "5곳"]
    if is_expand_query(query):
        return "recommend"
    if is_region_non_recommend_query(query):
        return "general"
    return "recommend" if any(t in q for t in recommend_terms) else "general"


def is_expand_query(query: str) -> bool:
    q = query.replace(" ", "")
    expand_terms = [
        "범위넓",
        "범위늘",
        "반경넓",
        "반경늘",
        "더넓",
        "더멀",
        "확대",
        "넓혀",
        "늘려",
        "거리넓",
    ]
    return any(t in q for t in expand_terms)


def is_nearby_query(query: str) -> bool:
    q = query.replace(" ", "")
    nearby_terms = [
        "근처",
        "주변",
        "인근",
        "근방",
        "주위",
        "가까운",
        "가까이",
        "인접",
    ]
    return any(t in q for t in nearby_terms)


def is_date_query(query: str) -> bool:
    q = normalize_text(query)
    date_terms = [
        "데이트",
        "커플",
        "여자친구",
        "남자친구",
        "연인",
        "기념일",
        "소개팅",
        "프로포즈",
        "썸",
        "둘이",
        "둘만",
        "2인",
    ]
    return any(t in q for t in date_terms)


def augment_query_for_date(query: str) -> str:
    if not query:
        return query
    hints = ["데이트", "로맨틱", "분위기", "기념일", "와인", "코스", "조용", "야경", "뷰"]
    q_norm = normalize_text(query)
    extras = [h for h in hints if normalize_text(h) not in q_norm]
    if not extras:
        return query
    return f"{query} " + " ".join(extras)
