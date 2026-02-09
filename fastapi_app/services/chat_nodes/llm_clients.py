import json
import os
import re


def _normalize_provider(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"openai", "open_ai"}:
        return "openai"
    if normalized == "clova":
        return "clova"
    if normalized == "gemini":
        return "gemini"
    return ""


def _provider() -> str:
    provider = _normalize_provider(os.getenv("CHAT_PROVIDER") or "")
    if not provider:
        raise RuntimeError("CHAT_PROVIDER must be set to one of: openai, clova, gemini")
    return provider


def _models() -> tuple[str, str]:
    chat_model = os.getenv("CHAT_MODEL")
    if not chat_model:
        raise RuntimeError("CHAT_MODEL must be set.")
    detect_model = os.getenv("DETECT_MODEL") or chat_model
    return chat_model, detect_model


def _openai_config(provider: str) -> tuple[str, str | None]:
    if provider == "clova":
        api_key = os.getenv("CLOVA_KEY")
        if not api_key:
            raise RuntimeError("CLOVA_KEY is required for Clova chat models.")
        base_url = os.getenv(
            "CLOVA_BASE_URL",
            "https://clovastudio.stream.ntruss.com/v1/openai",
        )
        return api_key, base_url
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI chat models.")
    return api_key, os.getenv("OPENAI_BASE_URL")


def _build_openai_llms(
    provider: str,
    chat_model: str,
    detect_model: str,
) -> tuple["ChatOpenAI", "ChatOpenAI"]:
    from langchain_openai import ChatOpenAI

    api_key, base_url = _openai_config(provider)

    def _make(model: str, streaming: bool) -> ChatOpenAI:
        kwargs = {
            "model": model,
            "streaming": streaming,
            "temperature": 0.0,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        try:
            return ChatOpenAI(**kwargs)
        except TypeError:
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            if base_url:
                os.environ["OPENAI_BASE_URL"] = base_url
                os.environ["OPENAI_API_BASE"] = base_url
            return ChatOpenAI(model=model, streaming=streaming, temperature=0.0)

    return _make(chat_model, True), _make(detect_model, False)


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
    chat_model, detect_model = _models()
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
    chat_model, detect_model = _models()
    llm, detect_llm = _build_openai_llms(provider, chat_model, detect_model)
