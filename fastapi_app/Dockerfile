# 1. uv 설치 이미지를 멀티 스테이지 빌드로 가져옵니다.
FROM ghcr.io/astral-sh/uv:latest AS uv_bin

# 2. 메인 파이썬 이미지 설정
FROM python:3.10-slim

# 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. uv 바이너리 복사
COPY --from=uv_bin /uv /uvx /bin/

WORKDIR /app

# 4. 의존성 파일 먼저 복사 (캐싱 효율화)
# uv.lock 파일이 로컬에 있다면 함께 복사하는 것이 좋습니다.
COPY pyproject.toml uv.lock* README.md /app/

# 5. 패키지 설치
# --system 옵션을 사용하여 컨테이너의 시스템 파이썬에 직접 설치합니다.
# pyproject.toml에 정의된 모든 의존성을 초고속으로 설치합니다.
RUN uv pip install --system --no-cache -r pyproject.toml

# 6. 소스 코드 복사
COPY . /app

# Cloud Run에서 GCS를 마운트할 경로 확인용 (선택 사항)
# VOLUME 설정은 Cloud Run의 GCS FUSE 마운트와는 별개로 작동하므로 생략 가능합니다.

EXPOSE 8000

# 7. 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]