import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
'''

python scripts/poi/update_latlng_from_csv.py \
  --csv data/embedding_json/latlon_missing_report_v3.csv \
  --result-csv data/embedding_json/latlon_geocode_results.csv
'''


'''
이 스크립트는 CSV 파일을 읽어 각 항목의 주소를 기반으로 위도와 경도를 구글 지오코딩 API를 통해 조회하고,
서울 지역에 속하는 항목만 남기고 나머지는 삭제합니다. 결과는 지정된 CSV 파일에 저장됩니다.
업데이트 된 주소는 embedding_json 디렉토리에 있는 JSON 파일에 반영됩니다.
'''
# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# API 키 확인
if not os.getenv("GOOGLE_GEOCODE_KEY"):
    raise SystemExit("GOOGLE_GEOCODE_KEY is not set. 터미널에 export GOOGLE_GEOCODE_KEY='내키'를 입력하세요.")

from fastapi_app.utils.google_geocode import geocode_address

DEFAULT_CSV = "data/embedding_json/latlon_missing_report_v3.csv"
DEFAULT_RESULT = "data/embedding_json/latlon_geocode_results.csv"

# --- [유틸리티 함수들] ---

def _normalize_int(value: Optional[str]) -> Optional[int]:
    if value is None: return None
    try: return int(float(str(value).strip()))
    except ValueError: return None

def _normalize_str(value: Optional[str]) -> Optional[str]:
    if value is None: return None
    text = str(value).strip()
    return text if text else None

def _is_missing(value: object, is_lat: bool) -> bool:
    if value is None: return True
    try:
        num = float(value)
        if num == 0: return True
        if is_lat and not (-90 <= num <= 90): return True
        if not is_lat and not (-180 <= num <= 180): return True
        return False
    except (TypeError, ValueError): return True

def _is_seoul(address: Optional[str]) -> bool:
    """서울 지역인지 확인합니다."""
    if not address: return False
    addr_lower = address.lower()
    return "서울" in addr_lower or "seoul" in addr_lower

def _resolve_address(row: Dict[str, str], item: Dict[str, object], field: str) -> Optional[str]:
    address = _normalize_str(row.get(field)) or _normalize_str(row.get("address"))
    if address: return address
    address = _normalize_str(item.get("address"))
    if address: return address
    location = item.get("location") if isinstance(item.get("location"), dict) else {}
    return _normalize_str(location.get("addr1")) or _normalize_str(item.get("road"))

def _get_target_item(data: List[Dict[str, object]], row: Dict[str, str], place_map: Dict[int, int], poi_map: Dict[str, int]) -> Tuple[Optional[Dict[str, object]], Optional[int], str]:
    pid = _normalize_int(row.get("place_id"))
    if pid in place_map: return data[place_map[pid]], place_map[pid], "place_id"
    poi = _normalize_str(row.get("poi_id"))
    if poi in poi_map: return data[poi_map[poi]], poi_map[poi], "poi_id"
    idx = _normalize_int(row.get("index"))
    if idx is not None and 0 <= idx < len(data): return data[idx], idx, "index"
    return None, None, "not_found"

def _determine_target(item: Dict[str, object], row: Dict[str, str]) -> str:
    if _normalize_str(row.get("source")) == "location.lat/lng" or "location" in item:
        return "location"
    return "root"

def _update_item(item: Dict[str, object], target: str, lat: float, lng: float) -> None:
    if target == "location":
        if "location" not in item: item["location"] = {}
        item["location"]["lat"], item["location"]["lng"] = float(lat), float(lng)
    else:
        item["latitude"], item["longitude"] = str(lat), str(lng)

def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f: return list(csv.DictReader(f))

def _resolve_path(data_root: Path, file_value: str) -> Path:
    raw = Path(file_value)
    return raw if raw.is_absolute() and raw.exists() else data_root / file_value

# --- [메인 함수 시작] ---

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--data-root", default="data/embedding_json")
    parser.add_argument("--result-csv", default=DEFAULT_RESULT)
    parser.add_argument("--address-field", default="address")
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    rows = _load_csv(Path(args.csv))
    
    # 파일을 기준으로 그룹화
    rows_by_file: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        fn = _normalize_str(r.get("file"))
        if fn: rows_by_file.setdefault(fn, []).append(r)

    results = []
    cache = {}

    for file_name, file_rows in rows_by_file.items():
        path = _resolve_path(data_root, file_name)
        if not path.exists():
            print(f"Skipping: {file_name} (File not found at {path})")
            continue

        print(f"\n>>> Processing: {file_name} ({len(file_rows)} rows)")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        to_delete_indices = set()
        
        # 맵 생성 (빠른 검색용)
        place_map = {(_normalize_int(item.get("place_id"))): i for i, item in enumerate(data) if item.get("place_id") is not None}
        poi_map = {(_normalize_str(item.get("poi_id"))): i for i, item in enumerate(data) if item.get("poi_id")}

        updated_count = 0

        for row in file_rows:
            item, idx, match_type = _get_target_item(data, row, place_map, poi_map)
            if item is None: continue

            address = _resolve_address(row, item, args.address_field)

            # 1. 서울 지역 필터링 (아니면 삭제 리스트에 추가)
            if not _is_seoul(address):
                to_delete_indices.add(idx)
                row_out = dict(row)
                row_out.update({"status": "deleted", "reason": "out_of_seoul/no_address", "resolved_address": address})
                results.append(row_out)
                continue

            # 2. 지오코딩 수행
            if address in cache: 
                geo = cache[address]
            else:
                geo = geocode_address(address)
                cache[address] = geo
                time.sleep(args.sleep)

            # 3. 데이터 업데이트 또는 삭제 결정
            if geo:
                _update_item(item, _determine_target(item, row), geo["lat"], geo["lng"])
                updated_count += 1
                row_out = dict(row)
                row_out.update({
                    "status": "updated", 
                    "resolved_address": geo.get("address"), 
                    "geocode_lat": str(geo["lat"]), 
                    "geocode_lng": str(geo["lng"])
                })
                results.append(row_out)
            else:
                to_delete_indices.add(idx)  # 지오코딩 결과 없으면 삭제
                row_out = dict(row)
                row_out.update({"status": "deleted", "reason": "geocode_failed", "resolved_address": address})
                results.append(row_out)

        # 4. 파일 저장 (중요: dry-run이 아닐 때 무조건 실행)
        if not args.dry_run:
            if to_delete_indices:
                final_data = [item for i, item in enumerate(data) if i not in to_delete_indices]
                print(f"   - {len(to_delete_indices)} items marked for deletion.")
            else:
                final_data = data
            
            with path.open("w", encoding="utf-8") as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)
            print(f"   - Successfully saved. (Updated: {updated_count} items)")

    # 5. 결과 리포트 CSV 저장
    if results:
        all_keys = set().union(*(res.keys() for res in results))
        result_path = Path(args.result_csv)
        with result_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(list(all_keys)), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nReport saved to: {result_path}")

if __name__ == "__main__":
    main()