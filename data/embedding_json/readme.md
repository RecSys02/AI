이 프로젝트는 여행지(POI) 데이터 기반 추천 시스템을 구축하기 위한 데이터 파이프라인과 분석 코드로 구성되어 있습니다.
TourAPI 원천 데이터 + 네이버 지도 리뷰 데이터를 기반으로 AI 태깅, 임베딩 생성, 추천 실험까지 진행합니다.

프로젝트 구조
AI/
 ├─ .venv/                     #  Python 가상환경 (Git ignore)
 ├─ data/
 │   ├─ raw/                   # 수집된 원본 POI / 외부 데이터
 │   ├─ interim/               # 중간 처리된 데이터
 │   ├─ processed/             # 학습/분석에 사용 가능한 최종 결과 데이터
 │   └─ user/                  # 사용자 프로필 데이터
 │
 ├─ eda/
 │   ├─ experiments/           # 실험용 스크립트, 테스트 코드
 │   ├─ figures/               # 시각화 결과
 │   ├─ notebook/              # Jupyter notebook
 │   └─ reports/               # 분석 결과 정리 파일
 │
 ├─ scripts/
 │   ├─ embedding/             # 임베딩 관련 스크립트
 │   │   ├─ generate_poi_embeddings.py
 │   │   └─ generate_user_embeddings.py
 │   └─ poi/                   # POI 메타 분석 스크립트
 │       └─ generate_poi_ai_analysis.py
 │
 ├─ .env                       # OpenAI KEY 등 환경 변수 (Git ignore)
 ├─ .gitignore
 ├─ requirements.txt
 └─ readme.md

