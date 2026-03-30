import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import random
import re
import weakref

try:
    from openai import RateLimitError as OpenAIRateLimitError
except Exception:
    OpenAIRateLimitError = None

logger = logging.getLogger("uvicorn.error")


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


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
_LOOP_SEMAPHORES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Semaphore]]" = (
    weakref.WeakKeyDictionary()
)

CHAT_LLM_MAX_CONCURRENCY = max(1, _env_int("CHAT_LLM_MAX_CONCURRENCY", 4))
DETECT_LLM_MAX_CONCURRENCY = max(1, _env_int("DETECT_LLM_MAX_CONCURRENCY", 8))
LLM_RATE_LIMIT_RETRIES = max(0, _env_int("LLM_RATE_LIMIT_RETRIES", 2))
LLM_RATE_LIMIT_BACKOFF_BASE_SEC = max(0.0, _env_float("LLM_RATE_LIMIT_BACKOFF_BASE_SEC", 1.0))
LLM_RATE_LIMIT_BACKOFF_MAX_SEC = max(
    LLM_RATE_LIMIT_BACKOFF_BASE_SEC,
    _env_float("LLM_RATE_LIMIT_BACKOFF_MAX_SEC", 8.0),
)
DEFAULT_RATE_LIMIT_FALLBACK_TEXT = (
    "요청이 몰려 응답 생성이 지연되고 있어요. 잠시 후 다시 시도해 주세요."
)


def max_tokens_kwargs(max_tokens: int) -> dict:
    """Return the provider-appropriate max token argument."""
    return {_MAX_TOKENS_KEY: max_tokens}


class LLMRateLimitError(RuntimeError):
    """Raised when upstream LLM calls keep failing with rate-limit errors."""


def rate_limit_fallback_text() -> str:
    return os.getenv("CHAT_RATE_LIMIT_FALLBACK_TEXT", DEFAULT_RATE_LIMIT_FALLBACK_TEXT).strip()


def _get_loop_semaphore(name: str, limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphores = _LOOP_SEMAPHORES.setdefault(loop, {})
    semaphore = semaphores.get(name)
    if semaphore is None:
        semaphore = asyncio.Semaphore(limit)
        semaphores[name] = semaphore
    return semaphore


@asynccontextmanager
async def _semaphore_guard(name: str, limit: int):
    semaphore = _get_loop_semaphore(name, limit)
    async with semaphore:
        yield


def _extract_retry_after(exc: Exception) -> float | None:
    for attr in ("retry_after", "retryAfter"):
        value = getattr(exc, attr, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        for key in ("retry-after", "Retry-After"):
            value = headers.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, LLMRateLimitError):
        return True
    if OpenAIRateLimitError is not None and isinstance(exc, OpenAIRateLimitError):
        return True
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status_code == 429 or str(status_code) == "429":
        return True
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    text = str(exc or "").lower()
    keywords = (
        "429",
        "rate limit",
        "rate_limit",
        "too many requests",
        "resource exhausted",
        "quota exceeded",
    )
    return any(keyword in text for keyword in keywords)


def _rate_limit_delay(exc: Exception, attempt: int) -> float:
    retry_after = _extract_retry_after(exc)
    if retry_after is not None and retry_after >= 0:
        return min(retry_after, LLM_RATE_LIMIT_BACKOFF_MAX_SEC)
    backoff = LLM_RATE_LIMIT_BACKOFF_BASE_SEC * (2 ** attempt)
    jitter = random.uniform(0.0, max(LLM_RATE_LIMIT_BACKOFF_BASE_SEC, 0.1))
    return min(backoff + jitter, LLM_RATE_LIMIT_BACKOFF_MAX_SEC)


class RateLimitedAsyncLLM:
    def __init__(
        self,
        inner,
        *,
        semaphore_name: str,
        max_concurrency: int,
        retries: int,
        label: str,
    ):
        self._inner = inner
        self._semaphore_name = semaphore_name
        self._max_concurrency = max_concurrency
        self._retries = retries
        self._label = label

    async def ainvoke(self, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                async with _semaphore_guard(self._semaphore_name, self._max_concurrency):
                    return await self._inner.ainvoke(*args, **kwargs)
            except Exception as exc:
                if not _is_rate_limit_error(exc):
                    raise
                last_exc = exc
                if attempt >= self._retries:
                    break
                delay = _rate_limit_delay(exc, attempt)
                logger.warning(
                    "llm rate limited label=%s attempt=%s/%s delay=%.2fs",
                    self._label,
                    attempt + 1,
                    self._retries + 1,
                    delay,
                )
                await asyncio.sleep(delay)
        raise LLMRateLimitError(f"{self._label} rate limited after retries") from last_exc

    async def astream(self, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(self._retries + 1):
            emitted = False
            try:
                async with _semaphore_guard(self._semaphore_name, self._max_concurrency):
                    async for chunk in self._inner.astream(*args, **kwargs):
                        emitted = True
                        yield chunk
                return
            except Exception as exc:
                if not _is_rate_limit_error(exc):
                    raise
                last_exc = exc
                if emitted or attempt >= self._retries:
                    break
                delay = _rate_limit_delay(exc, attempt)
                logger.warning(
                    "llm stream rate limited label=%s attempt=%s/%s delay=%.2fs",
                    self._label,
                    attempt + 1,
                    self._retries + 1,
                    delay,
                )
                await asyncio.sleep(delay)
        raise LLMRateLimitError(f"{self._label} rate limited after retries") from last_exc

    def __getattr__(self, name):
        return getattr(self._inner, name)


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

llm = RateLimitedAsyncLLM(
    llm,
    semaphore_name="chat_llm",
    max_concurrency=CHAT_LLM_MAX_CONCURRENCY,
    retries=LLM_RATE_LIMIT_RETRIES,
    label="chat_llm",
)
detect_llm = RateLimitedAsyncLLM(
    detect_llm,
    semaphore_name="detect_llm",
    max_concurrency=DETECT_LLM_MAX_CONCURRENCY,
    retries=LLM_RATE_LIMIT_RETRIES,
    label="detect_llm",
)
