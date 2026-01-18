# jsonl 파일을 json 파일로 변환하는 스크립트
import json

INPUT_PATH = "data/processed/poi_merged.jsonl"
OUTPUT_PATH = "data/processed/poi_merged.json"

def main():
    data = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"❌ skip line {idx}: {e}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Converted {len(data)} POIs → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
