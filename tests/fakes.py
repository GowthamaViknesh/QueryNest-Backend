"""A scripted fake LLM: tests the whole pipeline without calling (or paying for) a real model."""

from querynest.llm.types import LLMError, LLMResponse, ToolCall, Usage


class FakeProvider:
    """Replies with the scripted steps in order. A step is either:
      ("text", "answer")                       -> final answer
      ("tool", name, {args})                   -> one tool call
      ("error", retryable)                     -> raise LLMError
    Records every request so tests can inspect what the model was sent."""

    def __init__(self, steps, name="fake", is_local=False, model="fake-model"):
        self.steps = list(steps)
        self.name, self.model, self.is_local = name, model, is_local
        self.requests = []

    def stream(self, system, messages, tools, on_text):
        self.requests.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        step = self.steps.pop(0)
        if step[0] == "error":
            raise LLMError("fake failure", self.name, retryable=step[1])
        if step[0] == "text":
            for word in step[1].split(" "):
                on_text(word + " ")
            return LLMResponse(step[1], [], "end", Usage(100, 20), self.name, self.model)
        _, tool, args = step
        call = ToolCall(id=f"call_{len(self.requests)}", name=tool, args=args)
        return LLMResponse("", [call], "tool_use", Usage(100, 10), self.name, self.model)
