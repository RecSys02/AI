import os

from langchain_openai import ChatOpenAI

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
DETECT_MODEL = os.getenv("DETECT_MODEL", CHAT_MODEL)

llm = ChatOpenAI(model=CHAT_MODEL, streaming=True, temperature=0.0)
# 모드 감지/리랭크용은 스트리밍 없이 호출

detect_llm = ChatOpenAI(model=DETECT_MODEL, streaming=False, temperature=0.0)
