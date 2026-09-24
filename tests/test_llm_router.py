"""M3: retries, fallback, cache and cost tracking, with fake providers (no API calls)."""

import pytest
from fakes import FakeProvider

from querynest.llm import LLMError, LLMRouter
from querynest.llm.router import price
from querynest.llm.types import Usage


def chat(router, text_sink=None):
    sink = text_sink if text_sink is not None else []
    return router.chat("system", [{"role": "user", "content": "hi"}], [], sink.append)


def test_retry_then_success():
    p = FakeProvider([("error", True), ("text", "hello")])
    response = chat(LLMRouter([p], use_cache=False))
    assert response.text == "hello" and len(p.requests) == 2


def test_fallback_to_next_provider():
    first = FakeProvider([("error", True)] * 5, name="first")
    second = FakeProvider([("text", "from second")], name="second")
    response = chat(LLMRouter([first, second], use_cache=False))
    assert response.provider == "second"


def test_non_retryable_error_falls_back_immediately():
    first = FakeProvider([("error", False)], name="first")
    second = FakeProvider([("text", "ok")], name="second")
    chat(LLMRouter([first, second], use_cache=False))
    assert len(first.requests) == 1


def test_all_providers_failing_raises():
    with pytest.raises(LLMError, match="All LLM providers failed"):
        chat(LLMRouter([FakeProvider([("error", False)])], use_cache=False))


def test_cache_serves_identical_request():
    p = FakeProvider([("text", "cached answer")])
    router = LLMRouter([p])
    chat(router)
    sink = []
    second = chat(router, sink)
    assert second.cached and second.text == "cached answer" and sink == ["cached answer"]
    assert len(p.requests) == 1  # the provider was called only once
    assert second.usage.input_tokens == 0  # a cache hit costs nothing


def test_price_uses_price_table():
    assert price("claude-sonnet-5", Usage(input_tokens=1_000_000, output_tokens=1_000_000)) == pytest.approx(12.0)
    assert price("unknown-free-model", Usage(1000, 1000)) == 0.0
