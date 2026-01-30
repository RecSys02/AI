import json
import requests
import time
import os
from datetime import datetime
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 설정 - 환경 변수에서 가져오기
API_KEY = os.getenv("GOOGLE_PLACES_KEY") 
INPUT_FILE = "embedding_tourspot.json"
OUTPUT_FILE = "embedding_tourspot_with_ids.json"

if not API_KEY:
    raise ValueError(".env 파일에 GOOGLE_API_KEY가 설정되어 있지 않습니다.")

def get_place_id_and_stats(name, address, lat, lng):
    """이름과 주소를 조합해 검색하고, 좌표 근처 장소를 반환"""
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.id,places.rating,places.userRatingCount"
    }
    
    # 쿼리 조합: 이름 + 주소 (정확도 향상)
    query = f"{name} {address}"
    
    payload = {
        "textQuery": query,
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": 500.0
            }
        },
        "maxResultCount": 1
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            results = response.json().get("places", [])
            return results[0] if results else None
        return None
    except Exception as e:
        print(f"Error searching for {name}: {e}")
        return None

def run_seeding():
    # 파일 로드
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} 파일을 찾을 수 없습니다.")
        return

    total = len(data)
    updated = 0
    today_str = datetime.now().strftime("%Y-%m-%d")

    for index, item in enumerate(data):
        # 이미 place_id가 있는 경우는 스킵
        if item.get('google', {}).get('place_id'):
            continue

        name = item.get('name')
        address = item.get('location', {}).get('addr1', '')
        # lat/lng 데이터 타입 확인 (간혹 string으로 들어오는 경우 대비)
        try:
            lat = float(item.get('location', {}).get('lat'))
            lng = float(item.get('location', {}).get('lng'))
        except (TypeError, ValueError):
            print(f"[{index+1}/{total}] Skip: Invalid coordinates for {name}")
            continue

        print(f"[{index+1}/{total}] Searching for: {name} ({address})...")
        
        res = get_place_id_and_stats(name, address, lat, lng)
        
        if res:
            # 기존 google 항목이 없으면 새로 생성, 있으면 업데이트
            if 'google' not in item or not isinstance(item['google'], dict):
                item['google'] = {}
                
            item['google'].update({
                "place_id": res.get("id"),
                "rating": res.get("rating"),
                "user_ratings_total": res.get("userRatingCount"),
                "last_updated": today_str
            })
            updated += 1
            print(f"  -> Found! ID: {res.get('id')}")
        else:
            print(f"  -> Not Found.")

        # API 할당량(QPS) 및 안정성을 위해 10개마다 짧게 휴식
        if (index + 1) % 10 == 0:
            time.sleep(0.5)

    # 결과 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n작업 완료! 총 {updated}개의 항목에 ID를 매칭했습니다.")

if __name__ == "__main__":
    run_seeding()