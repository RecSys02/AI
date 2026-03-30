import asyncio
import importlib
import os
import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_llm_clients_module():
    os.environ["CHAT_PROVIDER"] = "openai"
    os.environ["CHAT_MODEL"] = "dummy-chat"
    os.environ["DETECT_MODEL"] = "dummy-detect"
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["LLM_RATE_LIMIT_RETRIES"] = "2"
    os.environ["LLM_RATE_LIMIT_BACKOFF_BASE_SEC"] = "0"
    os.environ["LLM_RATE_LIMIT_BACKOFF_MAX_SEC"] = "0"
    os.environ["CHAT_LLM_MAX_CONCURRENCY"] = "2"
    os.environ["DETECT_LLM_MAX_CONCURRENCY"] = "2"

    dummy_langchain_openai = types.ModuleType("langchain_openai")

    class DummyResp:
        def __init__(self, content):
            self.content = content

    class DummyChatOpenAI:
        def __init__(self, model=None, streaming=False, temperature=0.0, **kwargs):
            self.model = model
            self.streaming = streaming
            self.temperature = temperature
            self.attempts = 0

        async def ainvoke(self, messages, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise Exception("429 too many requests")
            return DummyResp("ok")

        async def astream(self, messages, **kwargs):
            yield DummyResp("ok")

    dummy_langchain_openai.ChatOpenAI = DummyChatOpenAI
    sys.modules["langchain_openai"] = dummy_langchain_openai

    if "services.chat_nodes.llm_clients" in sys.modules:
        del sys.modules["services.chat_nodes.llm_clients"]

    return importlib.import_module("services.chat_nodes.llm_clients")


def test_detect_llm_retries_rate_limit_then_succeeds():
    llm_clients = _load_llm_clients_module()

    async def _run():
        resp = await llm_clients.detect_llm.ainvoke([("user", "test")])
        return resp.content

    content = asyncio.run(_run())

    assert content == "ok"
