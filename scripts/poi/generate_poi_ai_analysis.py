# generate_poi_ai_analysis.py
# -*- coding: utf-8 -*-
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ⚙️ 파일 경로는 프로젝트에 맞게 수정해서 사용
INPUT_PATH = "./data/poi_final_normalized.jsonl"      # poi_id/type/category/... 가 들어있는 최종 마스터
OUTPUT_PATH = "./data/poi_analysis_ai_all.jsonl"      # type 상관 없이 ai 태그 모아서 저장

MAX_ITEMS = None  # 테스트용으로 50 같은 숫자 주면 50개만 돌고 종료


def _build_type_example_block(poi_type: str) -> str:
    """타입별로 예시 JSON과 설명을 다르게 구성."""
    if poi_type == "shopping":
        return """
[출력 형식 예시 - 쇼핑]

{
  "themes": ["쇼핑", "명품", "실내 데이트"],
  "mood": ["고급스러운", "세련된", "활기찬"],
  "visitor_type": ["연인", "친구", "20~40대 직장인", "관광객"],
  "best_time": ["주말", "저녁"],
  "best_time_flags": {
    "weekday": true,
    "weekend": true,
    "morning": false,
    "afternoon": true,
    "evening": true,
    "night": false
  },
  "duration": "2~3시간",
  "activity": {
    "label": "거의 걷지 않는",
    "level": 2
  },
  "photospot": true,
  "indoor_outdoor": "실내",
  "keywords": ["백화점", "명품 쇼핑", "실내 데이트", "우천시 추천"],
  "summary_one_sentence": "우천 시에도 쾌적하게 쇼핑과 데이트를 즐길 수 있는 도심형 고급 쇼핑 스팟.",
  "avoid_for": ["쇼핑에 관심 없는 여행자", "아웃도어 활동 선호자"],
  "ideal_schedule_position": "비가 오거나 더운 오후 일정에 배치 추천"
}

[타입: 쇼핑 (shopping)일 때 가이드]

- themes: '쇼핑', '시장', '플리마켓', '명품', '생활 쇼핑' 등 쇼핑 맥락을 중심으로 작성
- mood: '복잡한', '붐비는', '고급스러운', '일상적인' 등 쇼핑 공간의 분위기
- visitor_type: '가족', '연인', '주부', '관광객', '직장인' 등 실제로 많이 올 것 같은 사람들
- best_time / best_time_flags: 
  · 주말·저녁에 더 붐비는지, 평일 낮이 여유로운지 등 쇼핑 특성을 반영
- activity.level:
  · 1~2: 주로 실내, 걷는 양이 적거나 엘베/에스컬레이터 위주
  · 3: 매장 간 이동이 적당히 있는 대형 쇼핑몰
  · 4~5: 아웃렛 단지, 시장 골목을 많이 돌아다니는 경우 등
""".strip()

    if poi_type == "leisure":
        return """
[출력 형식 예시 - 레저/액티비티]

{
  "themes": ["레저", "액티비티", "체험", "운동"],
  "mood": ["역동적인", "스릴 있는", "활기찬"],
  "visitor_type": ["친구", "연인", "가족", "젊은 층"],
  "best_time": ["주말", "낮"],
  "best_time_flags": {
    "weekday": true,
    "weekend": true,
    "morning": true,
    "afternoon": true,
    "evening": false,
    "night": false
  },
  "duration": "2~4시간",
  "activity": {
    "label": "많이 걷는 또는 고강도 활동",
    "level": 5
  },
  "photospot": true,
  "indoor_outdoor": "실외",
  "keywords": ["체험 활동", "야외 레저", "가족 나들이", "주말 액티비티"],
  "summary_one_sentence": "주말에 신나게 땀 흘리고 스트레스를 풀 수 있는 대표 야외 레저 스팟.",
  "avoid_for": ["고강도 활동이 어려운 여행자", "조용한 휴식을 원하는 사람"],
  "ideal_schedule_position": "날씨가 좋은 주말 오전~오후 일정에 배치 추천"
}

[타입: 레저/액티비티(leisure)일 때 가이드]

- themes: 야외 레포츠, 실내 스포츠, 체험 프로그램 등 '몸을 쓰는 활동'을 중심으로 구성
- mood: '신나는', '스릴 있는', '에너지 넘치는', '가족 친화적인' 등
- activity.level:
  · 1~2: 라이트한 체험, 실내 가벼운 활동
  · 3: 일반적인 실내 체육, 볼링, 가벼운 체험 등
  · 4~5: 등산, 트래킹, 수상레포츠, 놀이공원, 고강도 레저 등
- best_time_flags:
  · 야외라면 낮/주말 비중, 날씨 좋은 시즌을 고려
""".strip()

    # default: attraction
    return """
[출력 형식 예시 - 관광/볼거리(attraction)]

{
  "themes": ["관광", "도심 산책", "거리 구경"],
  "mood": ["활기찬", "도시적인", "트렌디한"],
  "visitor_type": ["친구", "연인", "20~30대", "외국인 관광객"],
  "best_time": ["주중", "주말", "저녁"],
  "best_time_flags": {
    "weekday": true,
    "weekend": true,
    "morning": false,
    "afternoon": true,
    "evening": true,
    "night": false
  },
  "duration": "1~2시간",
  "activity": {
    "label": "적당히 걷는",
    "level": 3
  },
  "photospot": true,
  "indoor_outdoor": "복합",
  "keywords": ["도심 산책", "핫플레이스", "야경 명소", "만남의 장소"],
  "summary_one_sentence": "도심 속에서 쇼핑과 만남, 구경까지 한 번에 즐길 수 있는 대표 거리.",
  "avoid_for": ["조용한 자연 속 휴식을 원하는 여행자"],
  "ideal_schedule_position": "저녁에 가볍게 산책·구경하는 일정에 배치 추천"
}

[타입: 관광/볼거리(attraction)일 때 가이드]

- themes: '역사', '문화', '전망', '거리 구경', '산책', '야경' 등을 적절히 조합
- mood: '고즈넉한', '역사적인', '현대적인', '인파가 많은' 등 장소의 분위기를 표현
- activity.level:
  · 2: 박물관/전시 위주, 이동이 적은 코스
  · 3: 일반적인 시내 관광, 산책 위주
  · 4~5: 언덕/계단이 많거나 이동 동선이 긴 관광지
- photospot: 인스타/사진 명소인지, 포토 스팟이 많은지 여부를 반영
""".strip()

def build_prompt_for_ai_poi(poi: dict) -> str:
    """정규화된 POI(one row from poi_final_normalized.jsonl)를 받아 LLM 프롬프트 생성."""

    poi_type = poi.get("type") or "attraction"  # 기본은 attraction으로
    type_block = _build_type_example_block(poi_type)

    # 기본 필드
    name = poi.get("name") or "이름 정보 없음"
    gu_name = poi.get("gu_name") or ""

    # location
    loc = poi.get("location") or {}
    addr1 = loc.get("addr1") or ""
    addr2 = loc.get("addr2") or ""
    zipcode = loc.get("zipcode") or ""
    lat = loc.get("lat")
    lng = loc.get("lng")

    address = " / ".join([s for s in [gu_name, addr1, addr2, zipcode] if s]) or "주소 정보 없음"

    # category + 이름
    cat = poi.get("category") or {}
    cat1 = cat.get("cat1") or ""
    cat2 = cat.get("cat2") or ""
    cat3 = cat.get("cat3") or ""
    cat1_name = cat.get("cat1_name") or "카테고리 정보 없음"
    cat2_name = cat.get("cat2_name") or ""
    cat3_name = cat.get("cat3_name") or ""

    category_string = f"{cat1_name}"
    if cat2_name:
        category_string += f" > {cat2_name}"
    if cat3_name:
        category_string += f" > {cat3_name}"
    category_codes = f"{cat1} > {cat2} > {cat3}"

    overview = poi.get("overview") or "개요 정보 없음"

    # intro (정규화 버전)
    intro = poi.get("intro") or {}
    infocenter = intro.get("infocenter") or ""
    open_time = intro.get("open_time") or ""
    rest_date = intro.get("rest_date") or ""
    parking_info = intro.get("parking_info") or ""
    baby_carriage = intro.get("baby_carriage") or ""
    pet_allowed = intro.get("pet_allowed") or ""
    credit_card = intro.get("credit_card") or ""

    # 네이버 매칭 신뢰도
    naver_match = poi.get("naver_match") or "none"

    # 네이버 참고 정보
    naver = poi.get("naver") or {}

    # naver_match 값에 따라 어떻게 쓸지 결정
    if naver_match == "main":
        # 메인 장소와 잘 맞는 경우 → 그대로 사용
        naver_category = naver.get("naver_category") or "정보 없음"
        naver_rating = naver.get("naver_rating")
        naver_visitor_reviews = naver.get("naver_visitor_reviews")
        naver_blog_reviews = naver.get("naver_blog_reviews")
        naver_reliability_line = "- 네이버 매칭 신뢰도: 높음(메인 장소와 잘 일치함)"

    elif naver_match == "weak":
        # 하위 장소(예: 파르나스몰 안의 식당)일 가능성이 높은 경우
        # 👉 카테고리는 신뢰하지 않고, 인기도(리뷰 수) 정도만 약하게 참고
        naver_category = "정보 없음 (하위 장소일 가능성이 높아 카테고리는 사용하지 마세요)"
        # 그래도 '인기도' 정도는 참고 가능하다고 보고 그대로 둠
        naver_rating = naver.get("naver_rating")
        naver_visitor_reviews = naver.get("naver_visitor_reviews")
        naver_blog_reviews = naver.get("naver_blog_reviews")
        naver_reliability_line = (
            "- 네이버 매칭 신뢰도: 낮음(하위 매장/세부 장소일 가능성이 있으므로, "
            "인기도만 약하게 참고하고 카테고리는 사용하지 마세요)"
        )

    else:  # "none" 또는 기타
        # 네이버 매칭 자체가 없다고 보고 완전히 무시
        naver_category = "정보 없음"
        naver_rating = None
        naver_visitor_reviews = None
        naver_blog_reviews = None
        naver_reliability_line = "- 네이버 매칭 신뢰도: 매칭된 장소 없음(네이버 정보는 무시하세요)"

    naver_info_lines = [
        f"- 네이버 카테고리: {naver_category}",
        f"- 네이버 평점: {naver_rating}" if naver_rating is not None else "- 네이버 평점: 정보 없음",
        f"- 방문자 리뷰 수: {naver_visitor_reviews}" if naver_visitor_reviews is not None else "- 방문자 리뷰 수: 정보 없음",
        f"- 블로그 리뷰 수: {naver_blog_reviews}" if naver_blog_reviews is not None else "- 블로그 리뷰 수: 정보 없음",
        naver_reliability_line,
    ]
    naver_info_str = "\n".join(naver_info_lines)

    return f"""
아래는 이 POI 타입({poi_type})에 맞춘 출력 예시와 가이드입니다.
예시는 샘플일 뿐이며, 그대로 복사하지 말고 현재 장소 정보에 맞게 새로 생성해야 합니다.

{type_block}

[공통 출력 규칙]

1. 전체 응답은 JSON 객체 한 개만 출력하세요.
2. 한국어로 작성하세요.
3. 배열 안의 값도 모두 한국어로 작성하세요.
4. "best_time_flags"는 반드시 아래 6개 키를 모두 포함해야 합니다:
   - weekday, weekend, morning, afternoon, evening, night
   각 값은 true 또는 false만 사용하세요.
5. "activity" 객체는 반드시 다음 두 키를 가져야 합니다:
   - "label": 문자열 (예: "거의 걷지 않는", "적당히 걷는", "많이 걷는 또는 고강도 활동" 등)
   - "level": 1~5 사이의 정수
6. 아래 네이버 정보는 '장소의 인기도, 타겟 방문객, 분위기'를 추론하기 위한 참고용입니다.
   - 출력 JSON에는 네이버 평점, 리뷰 수, 카테고리, URL, place_id 등의 값을
     직접 숫자나 텍스트로 포함하지 마세요.
   - 특히, 네이버 매칭 신뢰도가 낮은(weak/none) 경우에는
     인기도(리뷰 수) 정도만 약하게 참고하고, 카테고리 정보는 사용하지 마세요.

이제 아래 [관광지 정보]를 분석하여,
위 예시와 동일한 구조의 JSON을 생성하세요.

[관광지 기본 정보]
- 이름: {name}
- 구 이름: {gu_name}
- 타입(type): {poi_type}
- 카테고리(코드): {category_codes}
- 카테고리(한글): {category_string}

[위치 정보]
- 주소: {address}
- 위도(lat): {lat}
- 경도(lng): {lng}

[운영/편의 정보]
- 안내소/문의(infocenter): {infocenter}
- 운영 시간(open_time): {open_time}
- 휴무일(rest_date): {rest_date}
- 주차 정보(parking_info): {parking_info}
- 유모차 대여/가능 여부(baby_carriage): {baby_carriage}
- 반려동물 동반 가능 여부(pet_allowed): {pet_allowed}
- 카드 결제 가능 여부(credit_card): {credit_card}

[네이버 방문/리뷰 정보 (참고용)]
{naver_info_str}

[관광지 개요]
{overview}
""".strip()



def analyze_poi_with_llm(poi: dict) -> dict:
    """단일 POI에 대해 LLM 호출해서 태그 JSON 얻기."""
    prompt = build_prompt_for_ai_poi(poi)

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",  # 필요하면 gpt-4.1-mini ↔ gpt-4.1 또는 gpt-5-mini로 교체
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 한국 여행지를 분석하는 큐레이션 전문가입니다. "
                    "제공된 장소 정보를 바탕으로 여행 태그를 설계하고, "
                    "항상 유효한 JSON 객체만 출력해야 합니다. "
                    "설명 문장, 마크다운, 코드블록을 출력하지 마세요. "
                    "네이버 평점/리뷰 수/카테고리/URL/ID 등은 출력 JSON에 직접 포함하지 말고, "
                    "단지 장소의 인기, 분위기, 방문객 유형을 추론하는 용도로만 사용하십시오."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content
    return json.loads(content)


def main():
    processed = 0

    # 이미 처리된 id set 불러오기 (중단 후 재실행 대비)
    processed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    processed_ids.add(str(record.get("id")))
                except Exception:
                    continue

    with open(INPUT_PATH, "r", encoding="utf-8") as infile, \
         open(OUTPUT_PATH, "a", encoding="utf-8") as outfile:

        for line_no, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                poi = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[라인 {line_no}] JSON 파싱 오류: {e}")
                continue

            poi_id = (
                poi.get("poi_id")
                or poi.get("id")
                or poi.get("contentid")
                or f"line-{line_no}"
            )

            if str(poi_id) in processed_ids:
                print(f"[스킵] 이미 처리됨 → id={poi_id}")
                continue

            if not poi.get("overview"):
                print(f"[라인 {line_no}] overview 없음 → 그래도 진행")

            try:
                result = analyze_poi_with_llm(poi)
                final_result = {
                    "id": str(poi_id),
                    "poi_type": poi.get("type"),
                    **result,
                }

                outfile.write(json.dumps(final_result, ensure_ascii=False) + "\n")
                outfile.flush()

                processed += 1
                print(f"[{processed}] 처리 완료 → id={poi_id}, name={poi.get('name')}")
            except Exception as e:
                print(f"⚠️ POI 처리 중 오류 (id={poi_id}, line={line_no}): {e}")
                continue

            time.sleep(0.3)

            if MAX_ITEMS is not None and processed >= MAX_ITEMS:
                print(f"\nMAX_ITEMS={MAX_ITEMS}개 처리 후 중단")
                break

    print("\n🎉 전체 작업 종료")
    print(f"총 처리 개수: {processed}")
    print(f"출력 파일: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
