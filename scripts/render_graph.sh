#!/usr/bin/env bash
# 경로는 fastapi_app 에서 실행 기준
# mmd 파일 만들기 : uv run ../scripts/render_graph.sh
# png 변환 : mmdc -i ../data/langgraph.mmd -o ../data/langgraph.png

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHONPATH="$ROOT_DIR/fastapi_app" \
OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}" \
ROOT_DIR="$ROOT_DIR" \
python - <<'PY'
from pathlib import Path
import os

from services.chat_graph import chat_app

root = Path(os.environ["ROOT_DIR"])
mmd = chat_app.get_graph().draw_mermaid()
out_path = root / "data" / "langgraph.mmd"
out_path.write_text(mmd, encoding="utf-8")
print(f"written: {out_path}")
PY
