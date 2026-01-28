import json
import os
import re


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
_MAX_TOKENS_KEY = "max_output_tokens" if provider == "gemini" else "max_tokens"
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\\s*|\\s*```$", re.IGNORECASE)


def max_tokens_kwargs(max_tokens: int) -> dict:
    """Return the provider-appropriate max token argument."""
    return {_MAX_TOKENS_KEY: max_tokens}


def parse_json_response(raw: str):
    """Best-effort JSON parser for LLM responses (handles code fences/extraneous text)."""
    if not raw:
        return None
    text = raw.strip()
    if "```" in text:
        text = _JSON_FENCE_RE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        try:
            return json.loads(text[obj_start : obj_end + 1])
        except json.JSONDecodeError:
            pass
    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start >= 0 and arr_end > arr_start:
        try:
            return json.loads(text[arr_start : arr_end + 1])
        except json.JSONDecodeError:
            pass
    return None
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
