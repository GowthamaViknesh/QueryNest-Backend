"""Lifecycle hooks (M4): run our own code at fixed points of every question, like NestJS
interceptors/guards. The agent calls hooks.emit(<event>, ...) and every registered function runs.

Events:
  before_question(user, question)                  may raise Rejected (quota, filters)
  after_llm(user, response, duration_ms)
  after_tool(user, tool, args, result, duration_ms)
  on_answer(user, question, text, usage, sql)
  on_error(user, error)

Built-in hooks below: input filter, rate limit, daily quota, audit log.
Add your own with @hooks.on("event").
"""

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy import func, select

from querynest.config import settings
from querynest.permissions import UserContext

log = logging.getLogger("querynest.hooks")


class Rejected(Exception):
    """A hook refused the request. The message is shown to the user."""


class Hooks:
    def __init__(self):
        self._handlers: dict[str, list[Callable[..., Any]]] = defaultdict(list)

    def on(self, event: str):
        def register(fn):
            self._handlers[event].append(fn)
            return fn
        return register

    def emit(self, event: str, **payload) -> None:
        for handler in self._handlers[event]:
            try:
                handler(**payload)
            except Rejected:
                raise
            except Exception:  # a broken audit write must never break the user's answer
                log.exception("hook %s failed in %s", handler.__name__, event)


hooks = Hooks()


# ------------------------------------------------------------------ audit log
def audit(user: UserContext, event: str, **fields) -> None:
    from querynest.appdb import AuditLog, session

    with session() as s:
        s.add(AuditLog(user_id=user.user_id, username=user.username, conversation_id=user.conversation_id,
                       event=event, **fields))
        s.commit()


@hooks.on("after_llm")
def audit_llm(user: UserContext, response, duration_ms: int) -> None:
    audit(user, "llm", provider=response.provider, model=response.model,
          input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
          cost_usd=response.usage.cost_usd, duration_ms=duration_ms,
          detail={"cached": response.cached, "stop": response.stop_reason, "tool_calls": [c.name for c in response.tool_calls]})


@hooks.on("after_tool")
def audit_tool(user: UserContext, tool: str, args: dict, result: Any, duration_ms: int) -> None:
    error = result.get("error") if isinstance(result, dict) else None
    event = "blocked" if error and "guardrails" in error else "tool"
    audit(user, event, tool=tool, sql=args.get("sql"), duration_ms=duration_ms,
          row_count=result.get("row_count") if isinstance(result, dict) else None,
          detail={"args": {k: v for k, v in args.items() if k != "sql"}, "error": error,
                  "executed_sql": result.get("executed_sql") if isinstance(result, dict) else None})


@hooks.on("on_answer")
def audit_answer(user: UserContext, question: str, text: str, usage, sql: list[str]) -> None:
    audit(user, "answer", input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
          cost_usd=usage.cost_usd, detail={"chars": len(text), "queries": len(sql)})


@hooks.on("on_error")
def audit_error(user: UserContext, error: str) -> None:
    audit(user, "error", detail={"error": error[:2000]})


# ------------------------------------------------------------------ input filter, limits
@hooks.on("before_question")
def input_filter(user: UserContext, question: str) -> None:
    if not question.strip():
        raise Rejected("Please type a question.")
    if len(question) > settings.max_question_chars:
        raise Rejected(f"Questions are limited to {settings.max_question_chars} characters.")


_recent: dict[str, deque] = defaultdict(deque)
_recent_lock = threading.Lock()


@hooks.on("before_question")
def rate_limit(user: UserContext, question: str) -> None:
    now = time.monotonic()
    with _recent_lock:
        window = _recent[user.username]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= settings.requests_per_minute:
            raise Rejected("Too many questions in the last minute. Please wait a moment.")
        window.append(now)


@hooks.on("before_question")
def daily_quota(user: UserContext, question: str) -> None:
    if user.user_id is None:
        return  # CLI / MCP system users
    from querynest.appdb import AuditLog, session

    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    with session() as s:
        used = s.scalar(select(func.count()).select_from(AuditLog).where(
            AuditLog.user_id == user.user_id, AuditLog.event == "question", AuditLog.created_at >= start))
    if used >= settings.questions_per_user_per_day:
        raise Rejected(f"Daily limit of {settings.questions_per_user_per_day} questions reached.")


# Registered last on purpose: only questions that passed every check above are logged
# (and counted by daily_quota).
@hooks.on("before_question")
def audit_question(user: UserContext, question: str) -> None:
    audit(user, "question", detail={"question": question})
