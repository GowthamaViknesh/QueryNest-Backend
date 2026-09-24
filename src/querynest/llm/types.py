"""Provider-neutral types: the agent speaks this format; each provider translates it.

Messages are plain dicts (easy to log, store and hash):
  {"role": "user", "content": "..."}
  {"role": "assistant", "content": "...", "tool_calls": [ToolCall, ...], "raw": {provider: ...}}
  {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}

"raw" keeps the provider's original assistant message. When the SAME provider continues the
conversation it gets its raw message back unchanged (Claude requires its thinking blocks to be
returned as-is); a different provider (after a fallback) gets the neutral version instead.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

Message = dict[str, Any]
OnText = Callable[[str], None]  # called with each piece of streamed answer text


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]  # JSON schema of the tool's input


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)  # provider-specific fields to echo back


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cost_usd += other.cost_usd


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # "end" | "tool_use" | "max_tokens" | "refusal"
    usage: Usage
    provider: str
    model: str
    raw: Any = None  # provider's assistant message, replayed to the same provider
    cached: bool = False


class LLMProvider(Protocol):
    """What every provider implements. Like a TypeScript interface."""

    name: str
    model: str
    is_local: bool  # local models get unmasked data (nothing leaves the machine)

    def stream(self, system: str, messages: list[Message], tools: list[ToolSpec], on_text: OnText) -> LLMResponse: ...


class LLMError(Exception):
    """A provider failed. retryable=True for temporary problems (rate limit, overload, network)."""

    def __init__(self, message: str, provider: str, retryable: bool, retry_after: float | None = None):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.retry_after = retry_after
