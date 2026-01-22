'''
uv run ../scripts/eval/gemini/label_candidates_gemini.py \
  --input ../data/eval/missing_candidates.jsonl \
  --output ../data/eval/labels_top5_missing_gemini.jsonl \
  --top-k 5 --sleep 0.1


'''
import argparse
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

def get_recommendations(model, user_data, top_k: int):
    user_id = user_data.get('user', {}).get('userId')
    
    # 2.0 모델의 지능을 활용하되, 형식을 엄격히 제한하는 프롬프트
    prompt = f"""
    당신은 데이터 엔지니어입니다. 제공된 유저의 취향과 후보지 데이터를 분석하여 최적의 추천 장소를 선정하세요.
    결과는 반드시 다음 JSON 배열 형식을 따라야 하며, 다른 텍스트는 포함하지 마세요.

    [형식]
    [
      {{"userId": {user_id}, "category": "restaurant", "relevant_ids": [5개 ID]}},
      {{"userId": {user_id}, "category": "cafe", "relevant_ids": [5개 ID]}},
      {{"userId": {user_id}, "category": "tourspot", "relevant_ids": [5개 ID]}}
    ]

    [데이터]
    {json.dumps(user_data, ensure_ascii=False)}
    """
    
    try:
        # 2.0 모델은 지능이 높아 긴 컨텍스트도 잘 이해합니다.
        response = model.generate_content(prompt)
        content = response.text.strip()
        
        # JSON 모드가 활성화되어 있어 응답 자체가 JSON 배열로 옵니다.
        parsed = json.loads(content)
        
        if isinstance(parsed, list):
            # JSONL 형식(한 줄에 객체 하나)으로 변환
            return "\n".join([json.dumps(item, ensure_ascii=False) for item in parsed])
        return json.dumps(parsed, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error for user {user_id}: {e}")
        return None
    
def main():
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")
    
    parser = argparse.ArgumentParser(description="Label candidates using Gemini.")
    parser.add_argument("--input", default="../data/eval/missing_candidates.jsonl")
    parser.add_argument("--output", default="../data/eval/labels_top5_missing_gemini.jsonl")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    # TPM 안전을 위해 1.0초 권장 (2.0-flash는 이 속도로도 충분히 빠릅니다)
    parser.add_argument("--sleep", type=float, default=1.0)
    # 3-flash 차단을 피해 한도가 높은 2.0-flash 사용
    parser.add_argument("--model", default="gemini-2.0-flash")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("환경변수 GEMINI_API_KEY가 없습니다.")
        return

    genai.configure(api_key=api_key)
    
    # [핵심] JSON 모드 강제 설정
    model = genai.GenerativeModel(
        model_name=args.model,
        generation_config={"response_mime_type": "application/json"}
    )

    if not os.path.exists(args.input):
        print(f"파일을 찾을 수 없습니다: {args.input}")
        return

    # 'a' 모드로 열어 중간에 끊겨도 이어서 저장 가능하게 설정
    with open(args.input, "r", encoding="utf-8") as f_in, open(
        args.output, "a", encoding="utf-8"
    ) as f_out:
        lines = f_in.readlines()
        if args.offset:
            lines = lines[args.offset :]
        if args.limit:
            lines = lines[: args.limit]

        print(f"총 {len(lines)}명의 유저 처리를 시작합니다. (모델: {args.model})")

        for i, line in enumerate(lines):
            try:
                user_data = json.loads(line)
                user_id = user_data.get("user", {}).get("userId", "Unknown")

                current_pos = (args.offset or 0) + i + 1
                print(f"[{current_pos}] 유저 {user_id} 분석 중...")

                result = get_recommendations(model, user_data, args.top_k)

                if result:
                    f_out.write(result + "\n")
                    f_out.flush()
                
                # 요청 간 간격 유지
                time.sleep(args.sleep)
                
            except Exception as e:
                print(f"라인 처리 중 오류 발생: {e}")
                continue

    print(f"처리가 완료되었습니다. 결과 파일: {args.output}")

if __name__ == "__main__":
    main()