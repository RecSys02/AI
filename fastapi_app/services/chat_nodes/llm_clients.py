import os


def _provider() -> str:
    provider = (os.getenv("LLM_PROVIDER") or os.getenv("CHAT_PROVIDER") or "").lower()
    if provider:
        return provider
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "openai"


def _gemini_models() -> tuple[str, str]:
    chat_model = (
        os.getenv("GEMINI_CHAT_MODEL")
        or os.getenv("GEMINI_MODEL")
        or os.getenv("GEMINI_RERANK_MODEL")
        or "gemini-2.0-flash"
    )
    detect_model = os.getenv("GEMINI_DETECT_MODEL") or chat_model
    return chat_model, detect_model


provider = _provider()
if provider == "gemini":
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as exc:
        raise RuntimeError("langchain-google-genai is required for Gemini chat models.") from exc

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for Gemini chat models.")
    chat_model, detect_model = _gemini_models()
    llm = ChatGoogleGenerativeAI(
        model=chat_model,
        temperature=0.0,
        streaming=True,
        google_api_key=api_key,
        convert_system_message_to_human=True,
    )
    # 모드 감지/리랭크용은 스트리밍 없이 호출
    detect_llm = ChatGoogleGenerativeAI(
        model=detect_model,
        temperature=0.0,
        streaming=False,
        google_api_key=api_key,
        convert_system_message_to_human=True,
    )
else:
    from langchain_openai import ChatOpenAI

    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    detect_model = os.getenv("DETECT_MODEL", chat_model)
    llm = ChatOpenAI(model=chat_model, streaming=True, temperature=0.0)
    # 모드 감지/리랭크용은 스트리밍 없이 호출
    detect_llm = ChatOpenAI(model=detect_model, streaming=False, temperature=0.0)
