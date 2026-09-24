"""The LLM router: one object the agent calls, with middleware wrapped around the providers.

  agent -> cache -> [provider 1 -> retries] -> fallback -> [provider 2 -> retries] -> ...
                                                     \\-> cost tracking on every response

- Retry: temporary errors (429 rate limit, 503 overloaded, network) are retried with
  exponential backoff (1s, 2s, 4s...), honouring the provider's Retry-After when given.
- Fallback: if a provider still fails, the next one in PROVIDER_ORDER takes over.
  We only fall back BEFORE any answer text was streamed, so the user never sees two answers
  glued together.
- Cache: an identical request (same model chain, system prompt, messages, tools) within
  LLM_CACHE_TTL_S is answered from memory: free and instant.
- Cost: tokens x price table in config.
"""

import hashlib
import json
import logging
import random
import time
from collections import OrderedDict
from dataclasses import asdict
from typing import Callable

from querynest.config import settings
from querynest.llm.anthropic_provider import AnthropicProvider
from querynest.llm.openai_provider import OpenAICompatProvider
from querynest.llm.types import LLMError, LLMProvider, LLMResponse, Message, OnText, ToolSpec, Usage

log = logging.getLogger("querynest.llm")
OnStatus = Callable[[str], None]  # progress messages, e.g. "gemini overloaded, switching to ollama"


def build_providers() -> list[LLMProvider]:
    """Every configured provider, in PROVIDER_ORDER."""
    s = settings
    factories: dict[str, Callable[[], LLMProvider | None]] = {
        "claude": lambda: s.anthropic_api_key and AnthropicProvider(
            s.anthropic_api_key.get_secret_value(), s.claude_model, s.llm_max_tokens, s.llm_timeout_s),
        "gemini": lambda: s.gemini_api_key and OpenAICompatProvider(
            "gemini", s.gemini_api_key.get_secret_value(), s.gemini_model, s.gemini_base_url,
            s.llm_max_tokens, s.llm_timeout_s),
        "openai": lambda: s.openai_api_key and s.openai_model and OpenAICompatProvider(
            "openai", s.openai_api_key.get_secret_value(), s.openai_model, s.openai_base_url,
            s.llm_max_tokens, s.llm_timeout_s),
        "grok": lambda: s.grok_api_key and s.grok_model and OpenAICompatProvider(
            "grok", s.grok_api_key.get_secret_value(), s.grok_model, s.grok_base_url,
            s.llm_max_tokens, s.llm_timeout_s),
        "ollama": lambda: s.ollama_enabled and OpenAICompatProvider(
            "ollama", "ollama", s.ollama_model, s.ollama_base_url, s.llm_max_tokens, s.llm_timeout_s,
            is_local=True),
    }
    providers = [p for name in s.provider_order if name in factories and (p := factories[name]())]
    if not providers:
        raise RuntimeError("No LLM provider configured. Set ANTHROPIC_API_KEY or GEMINI_API_KEY "
                           "(or OLLAMA_ENABLED=true) in Agent-Backend/.env.")
    return providers


def price(model: str, usage: Usage) -> float:
    """USD cost of one call. Cache reads are billed at ~10% of the input price (Claude)."""
    input_price, output_price = settings.model_prices.get(model, [0.0, 0.0])
    fresh_input = usage.input_tokens - usage.cache_read_tokens
    return (fresh_input * input_price + usage.cache_read_tokens * input_price * 0.1
            + usage.output_tokens * output_price) / 1_000_000


class ResponseCache:
    """In-memory LRU cache with a time-to-live. OrderedDict keeps entries in use order."""

    def __init__(self, size: int, ttl_s: int):
        self.size, self.ttl_s = size, ttl_s
        self._items: OrderedDict[str, tuple[float, LLMResponse]] = OrderedDict()

    @staticmethod
    def key(chain: list[str], system: str, messages: list[Message], tools: list[ToolSpec]) -> str:
        def neutral(m: Message) -> dict:  # drop provider raw blocks; they're not JSON
            return {k: ([asdict(c) for c in v] if k == "tool_calls" else v) for k, v in m.items() if k != "raw"}

        payload = json.dumps([chain, system, [neutral(m) for m in messages], [asdict(t) for t in tools]],
                             sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> LLMResponse | None:
        item = self._items.get(key)
        if item is None or time.monotonic() - item[0] > self.ttl_s:
            self._items.pop(key, None)
            return None
        self._items.move_to_end(key)
        return item[1]

    def put(self, key: str, response: LLMResponse) -> None:
        self._items[key] = (time.monotonic(), response)
        self._items.move_to_end(key)
        while len(self._items) > self.size:
            self._items.popitem(last=False)


class LLMRouter:
    def __init__(self, providers: list[LLMProvider] | None = None, use_cache: bool = True):
        self.providers = providers if providers is not None else build_providers()
        self.cache = ResponseCache(settings.llm_cache_size, settings.llm_cache_ttl_s) if use_cache else None

    @property
    def primary(self) -> LLMProvider:
        return self.providers[0]

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec], on_text: OnText,
             on_status: OnStatus = lambda s: None,
             system_for: Callable[[LLMProvider], str] | None = None,
             messages_for: Callable[[LLMProvider], list[Message]] | None = None) -> LLMResponse:
        """Get one response. system_for / messages_for let the caller adapt the input per provider
        (e.g. masked data for cloud providers, real data for local ones)."""
        chain = [f"{p.name}:{p.model}" for p in self.providers]
        cache_key = self.cache.key(chain, system, messages, tools) if self.cache else None
        if cache_key and (hit := self.cache.get(cache_key)):
            on_text(hit.text)
            log.info("LLM cache hit (%s)", hit.provider)
            return LLMResponse(**{**hit.__dict__, "usage": Usage(), "cached": True})

        streamed = False

        def track_text(t: str) -> None:
            nonlocal streamed
            streamed = True
            on_text(t)

        errors = []
        for provider in self.providers:
            p_system = system_for(provider) if system_for else system
            p_messages = messages_for(provider) if messages_for else messages
            for attempt in range(settings.llm_max_retries + 1):
                started = time.monotonic()
                try:
                    response = provider.stream(p_system, p_messages, tools, track_text)
                except LLMError as e:
                    log.warning("LLM %s attempt %d failed: %s", provider.name, attempt + 1, e)
                    if streamed:  # text already shown to the user: can't switch providers now
                        raise
                    errors.append(str(e))
                    if not e.retryable or attempt == settings.llm_max_retries:
                        break
                    wait = min(e.retry_after or (2 ** attempt + random.random()), settings.llm_retry_max_wait_s)
                    on_status(f"{provider.name} is busy, retrying in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                response.usage.cost_usd = price(response.model, response.usage)
                log.info("LLM %s/%s: %d in, %d out tokens, $%.5f, %.1fs", provider.name, response.model,
                         response.usage.input_tokens, response.usage.output_tokens,
                         response.usage.cost_usd, time.monotonic() - started)
                if cache_key:
                    self.cache.put(cache_key, response)
                return response
            if provider is not self.providers[-1]:
                on_status(f"{provider.name} unavailable, switching to the next model")
        raise LLMError("All LLM providers failed: " + " | ".join(errors[-3:]), "router", retryable=False)
