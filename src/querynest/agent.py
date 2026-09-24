"""The agent: question -> context -> (fast path | agent loop) -> answer -> follow-up suggestions.

  1. Context: matching glossary terms + similar SQL examples (knowledge.py) and the schema of
     the most relevant tables (schema.py), put straight into the prompt.
  2a. Fast path (idea from SQLBot): ONE planning call without tools returns a JSON plan
      {"mode": "simple", "sql": ..., "chart": ...}. We run the SQL (and chart) ourselves, then
      ONE answer call writes the reply. Two LLM calls for most questions.
  2b. Agent loop: for everything else (pivots, Excel, several queries, a failed plan) the model
      calls tools step by step, as before.
  3. Suggestions: one small call proposes 3 follow-up questions (no data is sent, only the
     question and the result's column names).

It produces a stream of EVENTS, used by the CLI (printed) and the API (sent to the browser
as Server-Sent Events):
  {"type": "status", "text": "..."}                       progress ("Running SQL...")
  {"type": "text", "delta": "..."}                        a piece of the answer as it's written
  {"type": "tool", "name": ..., "args": ..., "error": ...}   a tool call and its outcome
  {"type": "block", "block": {...}}                       result / chart / pivot / report / suggestions
  {"type": "done", "text": ..., "blocks": [...], "usage": {...}, "example_id": ..., "path": ...}
  {"type": "error", "message": "..."}
"""

import json
import logging
import queue
import re
import sys
import threading
import time
from datetime import date
from typing import Any, Iterator

from querynest import knowledge, schema
from querynest.config import settings
from querynest.guardrails import GuardrailError, ToolBudget, validate_sql
from querynest.hooks import Rejected, hooks
from querynest.llm import LLMError, LLMRouter, Message, Usage
from querynest.masking import TOKEN, Masker, StreamUnmasker
from querynest.permissions import UserContext, system_user
from querynest.tools import ToolContext, run_tool, to_json, tool_specs

log = logging.getLogger("querynest.agent")
TODAY = date.today().isoformat()

SYSTEM_PROMPT = f"""You are QueryNest, a data assistant for our company's sales database (PostgreSQL).
Today is {TODAY}.

How to work:
- The message lists the relevant tables and columns. Use list_tables / describe_table only when
  you need a table that isn't listed.
- Write PostgreSQL SELECT queries with schema-qualified names (e.g. sales.invoices).
- Prefer aggregated queries (SUM, COUNT, GROUP BY) over fetching raw rows.
- If a query fails, read the error, fix the SQL and try again.
- The user sees every query result as a table. Add a chart (create_chart) when a visual helps,
  a pivot (create_pivot) for two-dimensional breakdowns, an Excel report (create_excel_report)
  when the user asks for Excel, a report or a download, and a template report
  (create_template_report) when one of the listed templates fits a request for a pivot report.

Rules:
- Only answer using data from your tools. Never invent numbers.
- If the data can't answer the question, or the user has no access to it, say so plainly.
- Amounts are in Omani Rial (OMR) with 3 decimals.
- Give the answer first, then the key figures. Mention which table(s) you used.
- Messages may include "Business terms" and "Similar questions": use them when they fit.
- For greetings, thanks or small talk, reply in ONE short friendly sentence with an emoji.
  Don't list what you can do."""

# Plain greetings get this reply instantly, without an LLM call (faster, and saves quota)
GREETING_REPLY = "Hello! 👋 I'm QueryNest, your personal office data assistant. How can I help you today? 📊"
GREETING_WORDS = {"hi", "hii", "hiii", "hello", "helo", "hey", "hai", "hola", "greetings", "yo",
                  "good morning", "good afternoon", "good evening", "morning"}


def is_greeting(question: str) -> bool:
    """True for a bare greeting like "hi", "Hello!", "hey there", "good morning QueryNest"."""
    words = re.sub(r"[^\w\s]", " ", question.lower()).split()
    if not words or len(words) > 4:
        return False
    first = " ".join(words[:2]) if " ".join(words[:2]) in GREETING_WORDS else words[0]
    rest = set(words[len(first.split()):])
    return first in GREETING_WORDS and rest <= {"there", "querynest", "team", "all", "everyone", "bot"}

PLAN_PROMPT = f"""You plan how to answer a question about a PostgreSQL sales database. Today is {TODAY}.
Reply with ONLY a JSON object, no other text.

If ONE SELECT query answers the question (optionally shown as one chart), reply:
{{"mode": "simple", "sql": "<one PostgreSQL SELECT, schema-qualified tables>", "chart": "none|auto|bar|line|pie", "title": "<short title for the result>"}}
Use "chart": "none" unless a chart was asked for or clearly helps (trends, shares, comparisons).

Reply {{"mode": "agent"}} instead if the user asks for a pivot table, an Excel file, a report or
download, needs several separate queries, refers to something you can't resolve, or the
question isn't about the data.

Rules: use only the tables and columns listed in the message. Aggregate instead of listing raw
rows. Amounts are in OMR. Messages may include "Business terms" and "Similar questions"."""

ANSWER_PROMPT = f"""You are QueryNest, a data assistant. Today is {TODAY}.
The query in the message has already run; the user sees its result as a table (and the chart, if
one is mentioned). Answer the question in plain language using ONLY this data: the answer first,
then the key figures, and which table(s) were used. Don't repeat the whole table.
Amounts are in Omani Rial (OMR) with 3 decimals. If the data doesn't answer the question, say so."""

SUGGEST_PROMPT = """Suggest exactly 3 short follow-up questions the user might ask next about their
sales data, based on their question and the result's columns. Each under 90 characters, specific,
answerable from the same tables. Reply with ONLY a JSON array of 3 strings."""

MASKING_NOTE = """
Privacy: names and codes appear as placeholders like [P1]. Use them exactly as given, in SQL
too (e.g. WHERE customername = '[P1]'), and in your answer; they are replaced for the user."""


def parse_json(text: str) -> Any:
    """Parse the JSON a model returned, tolerating ```json fences and text around it."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


class Agent:
    def __init__(self, user: UserContext, router: LLMRouter, masker: Masker | None = None,
                 history: list[Message] | None = None, emit=lambda event: None):
        self.user = user
        self.router = router
        self.masker = masker or Masker()
        self.history = history or []
        self.emit = emit
        self.blocks: list[dict] = []
        self.usage = Usage()
        self.answer_parts: list[str] = []
        self.last_response = None

    # ------------------------------------------------------------------ per-provider input
    def _cloud(self, provider) -> bool:
        return settings.mask_cloud_data and not provider.is_local

    def _system(self, prompt: str):
        return lambda provider: prompt + (MASKING_NOTE if self._cloud(provider) else "")

    def _messages_for(self, messages: list[Message]):
        """Per provider: materialize attached data, and mask everything for cloud models."""
        def build(provider) -> list[Message]:
            cloud = self._cloud(provider)
            out = []
            for m in messages:
                m = dict(m)
                data = m.pop("data", None)
                if data is not None:  # structured data for the model, masked by column name
                    payload = self.masker.mask_json(json.loads(to_json(data))) if cloud else data
                    m["content"] = f"{m['content']}\n\nData:\n{to_json(payload)}"
                if cloud:
                    if m["role"] == "tool":
                        m["content"] = self.masker.mask_tool_content(m["content"])
                    elif m.get("content"):
                        m["content"] = self.masker.mask_text(m["content"])
                    if m["role"] == "assistant":
                        m.pop("raw", None)  # raw blocks may contain unmasked text; use the neutral form
                out.append(m)
            return out
        return build

    def _llm(self, prompt: str, messages: list[Message], specs, on_text=lambda t: None, masking_note: bool = True):
        started = time.monotonic()
        system_for = self._system(prompt) if masking_note else (lambda provider: prompt)
        response = self.router.chat(prompt, messages, specs, on_text,
                                    on_status=lambda s: self.emit({"type": "status", "text": s}),
                                    system_for=system_for, messages_for=self._messages_for(messages))
        self.usage.add(response.usage)
        self.last_response = response
        hooks.emit("after_llm", user=self.user, response=response,
                   duration_ms=int((time.monotonic() - started) * 1000))
        return response

    def _tool(self, name: str, args: dict, budget: ToolBudget) -> Any:
        self.emit({"type": "status", "text": f"Running {name}"})
        started = time.monotonic()
        result = budget.run(name, args)
        hooks.emit("after_tool", user=self.user, tool=name, args=args, result=result,
                   duration_ms=int((time.monotonic() - started) * 1000))
        error = result.get("error") if isinstance(result, dict) else None
        self.emit({"type": "tool", "name": name, "args": args, "error": error})
        return result

    def _streamer(self):
        unmask = StreamUnmasker(self.masker)

        def on_text(chunk: str) -> None:
            if text := unmask.feed(chunk):
                self.answer_parts.append(text)
                self.emit({"type": "text", "delta": text})

        def flush() -> None:
            if tail := unmask.flush():
                self.answer_parts.append(tail)
                self.emit({"type": "text", "delta": tail})
        return on_text, flush

    # ------------------------------------------------------------------ 1. context
    def _context(self, question: str) -> str:
        found = knowledge.retrieve(question)
        # Only offer examples this user could actually run (e.g. no ledger SQL for sales users)
        found.examples = [(ex, s) for ex, s in found.examples if self._allowed(ex.sql)]
        knowledge.mark_used(found.examples)
        tables = schema.select_tables(question, self.user, found)
        self.emit({"type": "status", "text": f"Context: {len(tables)} tables, {len(found.terms)} business terms, "
                                             f"{len(found.examples)} similar examples"})
        parts = [found.as_prompt(), schema.schema_prompt(tables) if tables else ""]
        if self.user.can_use_tool("create_template_report"):
            from querynest.report_templates import load_templates
            ready = [t for t in load_templates().values() if t.ready]
            if ready:
                parts.append("Excel templates with a real PivotTable (create_template_report):\n" +
                             "\n".join(f"- {t.name}: {t.description}" for t in ready))
        context = "\n\n".join(p for p in parts if p)
        return f"{context}\n\nQuestion: {question}" if context else question

    # ------------------------------------------------------------------ 2a. fast path
    def _fast_path(self, question: str, user_content: str, ctx: ToolContext, budget: ToolBudget) -> bool:
        """Plan -> run -> answer. Returns False (nothing shown yet) if the full agent must take over."""
        self.emit({"type": "status", "text": "Planning"})
        plan = parse_json(self._llm(PLAN_PROMPT, [*self.history, {"role": "user", "content": user_content}], []).text)
        if not isinstance(plan, dict) or plan.get("mode") != "simple" or not plan.get("sql"):
            log.info("fast path declined: %s", plan)
            return False
        sql = self.masker.unmask_sql(str(plan["sql"]))
        title = self.masker.unmask_text(str(plan.get("title") or "Query result"))[:200]
        result = self._tool("run_sql_query", {"sql": sql, "title": title}, budget)
        if "error" in result:
            self.emit({"type": "status", "text": "The quick plan didn't work; switching to the full agent"})
            return False
        chart = None
        if plan.get("chart") in ("auto", "bar", "line", "pie") and result.get("row_count", 0) > 1:
            chart = self._tool("create_chart", {"result_id": result["result_id"], "chart_type": plan["chart"],
                                                "title": title}, budget)
        on_text, flush = self._streamer()
        data = {"sql": result["executed_sql"], "columns": result["columns"], "rows": result["rows"],
                "row_count": result["row_count"], **({"note": result["note"]} if "note" in result else {}),
                **({"chart_shown": chart} if chart and "error" not in chart else {})}
        self._llm(ANSWER_PROMPT, [*self.history, {"role": "user", "content": question, "data": data}], [], on_text)
        flush()
        return True

    # ------------------------------------------------------------------ 2b. agent loop
    def _agent_loop(self, user_content: str, ctx: ToolContext, budget: ToolBudget) -> None:
        messages: list[Message] = [*self.history, {"role": "user", "content": user_content}]
        specs = tool_specs(self.user)
        on_text, flush = self._streamer()
        for _ in range(settings.max_rounds):
            response = self._llm(SYSTEM_PROMPT, messages, specs, on_text)
            flush()
            messages.append({"role": "assistant", "content": response.text, "tool_calls": response.tool_calls,
                             **({"raw": {response.provider: response.raw}} if response.raw is not None else {})})
            if not response.tool_calls:
                return
            if self.answer_parts and not self.answer_parts[-1].endswith("\n"):
                self.answer_parts.append("\n\n")
                self.emit({"type": "text", "delta": "\n\n"})
            for call in response.tool_calls:
                args = self.masker.unmask_json(call.args)  # tokens like [P1] back to real values
                if "sql" in call.args:
                    args["sql"] = self.masker.unmask_sql(call.args["sql"])
                result = self._tool(call.name, args, budget)
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                 "content": to_json(result)})
        stopped = f"\n\n(Stopped after {settings.max_rounds} steps without a final answer.)"
        self.answer_parts.append(stopped)
        self.emit({"type": "text", "delta": stopped})

    # ------------------------------------------------------------------ 3. suggestions
    def _suggestions(self, question: str, ctx: ToolContext) -> None:
        columns = ctx.state.get("last_columns", [])
        content = f"User question: {question}\nColumns in the result: {', '.join(columns) or 'none'}"
        try:
            # No data goes into this call, so no placeholder note either (the model would imitate it)
            items = parse_json(self._llm(SUGGEST_PROMPT, [{"role": "user", "content": content}], [],
                                         masking_note=False).text)
        except LLMError:
            return  # suggestions are a nice-to-have; never fail the answer for them
        if isinstance(items, list):
            items = [self.masker.unmask_text(str(s)).strip()[:120] for s in items if str(s).strip()]
            items = [s for s in items if not TOKEN.search(s)][:3]  # drop any unresolved [P#] placeholder
            if items:
                self._block({"kind": "suggestions", "items": items})

    # ------------------------------------------------------------------ run
    def run(self, question: str) -> dict[str, Any]:
        hooks.emit("before_question", user=self.user, question=question)
        if is_greeting(question):
            self.answer_parts.append(GREETING_REPLY)
            self.emit({"type": "text", "delta": GREETING_REPLY})
            return {"text": GREETING_REPLY, "blocks": [], "sql": [], "example_id": None, "path": "greeting",
                    "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "path": "greeting"}}
        user_content = self._context(question)
        ctx = ToolContext(user=self.user, emit=self._block)
        budget = ToolBudget(lambda name, args: run_tool(name, args, ctx))

        path = "agent"
        if settings.fast_path and self._fast_path(question, user_content, ctx, budget):
            path = "fast"
        else:
            self._agent_loop(user_content, ctx, budget)
        text = "".join(self.answer_parts).strip()
        answer_response = self.last_response
        if settings.suggest_followups and text:
            self._suggestions(question, ctx)

        sql = ctx.state.get("sql", [])
        example_id = knowledge.remember(question, sql[-1], self.user.user_id) if sql else None
        hooks.emit("on_answer", user=self.user, question=question, text=text, usage=self.usage, sql=sql)
        return {"text": text, "blocks": self.blocks, "sql": sql, "example_id": example_id, "path": path,
                "usage": {"input_tokens": self.usage.input_tokens, "output_tokens": self.usage.output_tokens,
                          "cost_usd": round(self.usage.cost_usd, 6), "provider": answer_response.provider,
                          "model": answer_response.model, "path": path}}

    def _block(self, block: dict) -> None:
        self.blocks.append(block)
        self.emit({"type": "block", "block": block})

    def _allowed(self, sql: str) -> bool:
        try:
            validate_sql(sql, self.user, columns_of=lambda t: ["*"])
            return True
        except GuardrailError:
            return False


def stream_agent(question: str, user: UserContext, router: LLMRouter, masker: Masker | None = None,
                 history: list[Message] | None = None) -> Iterator[dict]:
    """Run the agent in a thread; yield its events as they happen. Ends with done or error."""
    events: queue.Queue = queue.Queue()
    agent = Agent(user, router, masker, history, emit=events.put)

    def work() -> None:
        try:
            events.put({"type": "done", **agent.run(question)})
        except Rejected as e:
            events.put({"type": "error", "message": str(e)})
        except LLMError as e:
            hooks.emit("on_error", user=user, error=str(e))
            events.put({"type": "error", "message": "The AI service is unavailable right now. "
                                                    "Please try again in a minute."})
        except Exception as e:
            log.exception("agent failed")
            hooks.emit("on_error", user=user, error=repr(e))
            events.put({"type": "error", "message": "Something went wrong while answering."})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True, name="agent").start()
    while (event := events.get()) is not None:
        yield event


# ---------------------------------------------------------------------------
# CLI:  uv run agent "question"   (runs as a system user with the manager role)
# ---------------------------------------------------------------------------
def main() -> None:
    from querynest.logging_setup import setup_logging

    setup_logging(console_level=logging.WARNING)
    sys.stdout.reconfigure(encoding="utf-8")
    question = " ".join(sys.argv[1:]) or "Which were our top 5 customers by sales in 2025, and how much did each buy?"
    router = LLMRouter()
    print("Models:", " -> ".join(f"{p.name}:{p.model}" for p in router.providers))
    print(f"\nUSER: {question}\n\nASSISTANT: ", end="", flush=True)
    for event in stream_agent(question, system_user("manager"), router):
        kind = event["type"]
        if kind == "text":
            print(event["delta"], end="", flush=True)
        elif kind == "status":
            print(f"\n  [{event['text']}]", flush=True)
        elif kind == "tool":
            detail = event["args"].get("sql") or event["args"]
            print(f"  -> {event['name']}: {detail}" + (f"\n     ERROR: {event['error']}" if event["error"] else ""))
        elif kind == "block":
            b = event["block"]
            if b["kind"] == "suggestions":
                print("\n  Follow-ups: " + " | ".join(b["items"]))
            else:
                print(f"  [{b['kind']}] {b.get('title', '')}" + (f" ({b['row_count']} rows)" if b["kind"] == "result" else ""))
        elif kind == "done":
            u = event["usage"]
            print(f"\n\n({event['path']} path · {u['provider']}/{u['model']}: {u['input_tokens']} in, "
                  f"{u['output_tokens']} out tokens, ${u['cost_usd']:.4f})")
        elif kind == "error":
            print(f"\nERROR: {event['message']}")


if __name__ == "__main__":
    main()
