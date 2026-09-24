"""Any OpenAI-compatible API: Gemini, OpenAI, Grok, and local Ollama (Qwen). Streaming."""

import json

import openai

from querynest.llm.types import LLMError, LLMResponse, Message, OnText, ToolCall, ToolSpec, Usage

STOP_REASONS = {"stop": "end", "tool_calls": "tool_use", "length": "max_tokens", "content_filter": "refusal"}
KNOWN_TOOL_CALL_KEYS = {"id", "type", "function", "index"}


class OpenAICompatProvider:
    def __init__(self, name: str, api_key: str, model: str, base_url: str | None,
                 max_tokens: int, timeout: float, is_local: bool = False):
        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self.is_local = is_local
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout)

    def _to_openai(self, system: str, messages: list[Message]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg: dict = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        # extra: provider-specific fields we must echo back (Gemini's thought signature)
                        {"id": c.id, "type": "function",
                         "function": {"name": c.name, "arguments": json.dumps(c.args)}, **c.extra}
                        for c in m["tool_calls"]
                    ]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        return out

    def stream(self, system: str, messages: list[Message], tools: list[ToolSpec], on_text: OnText) -> LLMResponse:
        oa_tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                      "parameters": t.schema}} for t in tools]
        text_parts: list[str] = []
        calls: dict[int, dict] = {}  # tool calls arrive in pieces, keyed by index
        finish, usage = None, Usage()
        try:
            kwargs = {"tools": oa_tools} if oa_tools else {}
            chunks = self.client.chat.completions.create(
                model=self.model,
                messages=self._to_openai(system, messages),
                max_tokens=self.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )
            for chunk in chunks:
                if chunk.usage:
                    usage.input_tokens = chunk.usage.prompt_tokens or 0
                    usage.output_tokens = chunk.usage.completion_tokens or 0
                for choice in chunk.choices:
                    delta = choice.delta
                    if delta.content:
                        text_parts.append(delta.content)
                        on_text(delta.content)
                    for tc in delta.tool_calls or []:
                        slot = calls.setdefault(tc.index or 0, {"id": "", "name": "", "args": "", "extra": {}})
                        slot["id"] = tc.id or slot["id"]
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["args"] += tc.function.arguments
                        extra = {k: v for k, v in tc.model_dump(exclude_none=True).items()
                                 if k not in KNOWN_TOOL_CALL_KEYS}
                        slot["extra"].update(extra)
                    finish = choice.finish_reason or finish
        except openai.RateLimitError as e:
            retry_after = e.response.headers.get("retry-after")
            # A daily quota won't reset soon: don't retry, fall back to the next provider
            daily = "PerDay" in str(e.body)
            raise LLMError(f"{self.name} rate limit: {str(e)[:200]}", self.name, not daily,
                           float(retry_after) if retry_after else None) from None
        except openai.APIConnectionError as e:
            raise LLMError(f"{self.name} connection error: {e}", self.name, True) from None
        except openai.BadRequestError as e:
            raise LLMError(f"{self.name} rejected the request: {str(e)[:300]}", self.name, False) from None
        except openai.APIStatusError as e:
            raise LLMError(f"{self.name} error {e.status_code}: {str(e)[:200]}", self.name,
                           e.status_code >= 500) from None

        tool_calls = []
        for i, c in sorted(calls.items()):
            try:
                args = json.loads(c["args"] or "{}")
            except json.JSONDecodeError:
                args = {"__invalid_json__": c["args"]}  # run_tool reports it back to the model
            tool_calls.append(ToolCall(id=c["id"] or f"call_{i}", name=c["name"], args=args, extra=c["extra"]))
        stop = "tool_use" if tool_calls else STOP_REASONS.get(finish or "stop", "end")
        return LLMResponse(text="".join(text_parts), tool_calls=tool_calls, stop_reason=stop,
                           usage=usage, provider=self.name, model=self.model)
