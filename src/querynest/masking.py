"""Masking of personal data sent to cloud LLMs (M11).

Values in sensitive columns (customer names, salesman names, ...) are replaced by tokens like
[P1], [P2] before anything goes to a cloud model. The model reasons with the tokens ("[P1] is
the top customer") and may even use them in SQL (WHERE customername = '[P1]'). We put the
real values back in the SQL before running it, and in the answer before showing it.

Result: the user sees real names; the cloud provider never does. Local models (Ollama) get
the real data, because nothing leaves the machine.
The mapping is stored per conversation, so [P1] means the same thing in follow-up questions.
"""

import json
import re
from typing import Any

from querynest.config import settings

TOKEN = re.compile(r"\[P\d+\]")
PARTIAL_TOKEN_AT_END = re.compile(r"\[(P\d*)?$")


class Masker:
    def __init__(self, mapping: dict[str, str] | None = None, columns: list[str] | None = None):
        self.to_real: dict[str, str] = dict(mapping or {})  # "[P1]" -> "Customer 0352"
        self.to_token: dict[str, str] = {v: k for k, v in self.to_real.items()}
        self.columns = {c.lower() for c in (columns if columns is not None else settings.sensitive_columns)}

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self.to_real)

    def token(self, value: str) -> str:
        if value not in self.to_token:
            token = f"[P{len(self.to_real) + 1}]"
            self.to_real[token] = value
            self.to_token[value] = token
        return self.to_token[value]

    # ---------------------------------------------------------------- masking (to the LLM)
    def mask_json(self, value: Any, key: str | None = None) -> Any:
        """Mask a tool result: values under sensitive keys, plus any string already known."""
        if isinstance(value, dict):
            return {k: self.mask_json(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [self.mask_json(v, key) for v in value]
        if isinstance(value, str):
            if key and key.lower() in self.columns and value.strip():
                return self.token(value)
            return self.mask_text(value)
        return value

    def mask_text(self, text: str) -> str:
        """Replace known real values inside free text (longest first, so 'Customer 12' doesn't
        clobber 'Customer 123')."""
        for real in sorted(self.to_token, key=len, reverse=True):
            if real and real in text:
                text = text.replace(real, self.to_token[real])
        return text

    def mask_tool_content(self, content: str) -> str:
        try:
            return json.dumps(self.mask_json(json.loads(content)), ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return self.mask_text(content)

    # ---------------------------------------------------------------- unmasking (from the LLM)
    def unmask_text(self, text: str) -> str:
        return TOKEN.sub(lambda m: self.to_real.get(m.group(0), m.group(0)), text)

    def unmask_sql(self, sql: str) -> str:
        # Tokens normally sit inside a string literal: '[P1]' -> 'O''Brien LLC' (quotes escaped)
        return TOKEN.sub(lambda m: self.to_real.get(m.group(0), m.group(0)).replace("'", "''"), sql)

    def unmask_json(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self.unmask_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.unmask_json(v) for v in value]
        if isinstance(value, str):
            return self.unmask_text(value)
        return value


class StreamUnmasker:
    """Unmask streamed text. A token can arrive split across chunks ("[P" + "12]"), so we hold
    back a possible token start until the next chunk completes it."""

    def __init__(self, masker: Masker):
        self.masker = masker
        self.pending = ""

    def feed(self, chunk: str) -> str:
        text = self.pending + chunk
        match = PARTIAL_TOKEN_AT_END.search(text)
        if match:
            self.pending, text = text[match.start():], text[:match.start()]
        else:
            self.pending = ""
        return self.masker.unmask_text(text)

    def flush(self) -> str:
        text, self.pending = self.pending, ""
        return self.masker.unmask_text(text)
