"""Claude through the official Anthropic SDK, with streaming."""

import anthropic

from querynest.llm.types import LLMError, LLMResponse, Message, OnText, ToolCall, ToolSpec, Usage

STOP_REASONS = {"end_turn": "end", "stop_sequence": "end", "tool_use": "tool_use",
                "max_tokens": "max_tokens", "refusal": "refusal", "pause_turn": "end"}
MAX_JSON_RETRIES = 2  # re-issue a turn whose streamed tool input was unparseable


class AnthropicProvider:
    is_local = False

    def __init__(self, api_key: str, model: str, max_tokens: int, timeout: float):
        self.name = "claude"
        self.model = model
        self.max_tokens = max_tokens
        # max_retries=0: our router does retries + fallback, so the SDK must not retry silently
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=timeout)

    def _to_claude(self, messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                raw = m.get("raw", {}).get(self.name)
                if raw is not None:  # same provider: replay unchanged (keeps thinking blocks)
                    out.append({"role": "assistant", "content": raw})
                    continue
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for c in m.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.args})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no text)"}]})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                # All results of one turn must go back in ONE user message (keeps parallel calls working)
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def stream(self, system: str, messages: list[Message], tools: list[ToolSpec], on_text: OnText) -> LLMResponse:
        claude_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.schema,
             "eager_input_streaming": True}  # tool inputs stream as generated; we validate them
            for t in tools
        ]
        extra = {"tools": claude_tools} if claude_tools else {}  # planning calls have no tools
        for attempt in range(MAX_JSON_RETRIES + 1):
            try:
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=self._to_claude(messages),
                    **extra,
                    cache_control={"type": "ephemeral"},  # cache the stable prefix (tools + system)
                ) as stream:
                    for event in stream:
                        if event.type == "text":
                            on_text(event.text)
                    final = stream.get_final_message()
                break
            except ValueError:  # streamed tool input was not parseable JSON: re-issue the turn
                if attempt == MAX_JSON_RETRIES:
                    raise LLMError("Claude sent an unparseable tool input", self.name, retryable=True)
            except anthropic.RateLimitError as e:
                retry_after = e.response.headers.get("retry-after")
                raise LLMError(f"Claude rate limit: {e.message}", self.name, True,
                               float(retry_after) if retry_after else None) from None
            except anthropic.APIConnectionError as e:  # includes timeouts
                raise LLMError(f"Claude connection error: {e}", self.name, True) from None
            except anthropic.BadRequestError as e:
                raise LLMError(f"Claude rejected the request: {e.message}", self.name, False) from None
            except anthropic.APIStatusError as e:
                raise LLMError(f"Claude error {e.status_code}: {e.message}", self.name,
                               e.status_code >= 500 or e.status_code == 429) from None

        text = "".join(b.text for b in final.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, args=b.input if isinstance(b.input, dict) else {})
                 for b in final.content if b.type == "tool_use"]
        stop = STOP_REASONS.get(final.stop_reason or "end_turn", "end")
        if stop == "refusal":
            text = text or "The model declined to answer this request."
            calls = []
        u = final.usage
        usage = Usage(input_tokens=u.input_tokens + (u.cache_creation_input_tokens or 0) + (u.cache_read_input_tokens or 0),
                      output_tokens=u.output_tokens, cache_read_tokens=u.cache_read_input_tokens or 0)
        return LLMResponse(text=text, tool_calls=calls, stop_reason=stop, usage=usage,
                           provider=self.name, model=self.model, raw=final.content)
