"""User-facing endpoints: login, conversations, streaming chat, results, Excel, feedback."""

import json
import threading
import time
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from querynest import excel, knowledge, results
from querynest.agent import stream_agent
from querynest.api.deps import current_user, llm_router, user_context
from querynest.api.schemas import (AskRequest, ConversationDetail, ConversationOut, FeedbackRequest, LoginRequest,
                                   LoginResponse, MessageOut, ResultOut, UserOut)
from querynest.appdb import ChatMessage, Conversation, Report, User, now, session
from querynest.auth import create_token, verify_password
from querynest.config import settings
from querynest.llm import LLMRouter
from querynest.masking import Masker
from querynest.permissions import ROLES
from querynest.tools import TOOLS

router = APIRouter(prefix="/api")


def user_out(user: User) -> UserOut:
    role = ROLES[user.role]
    return UserOut(id=user.id, username=user.username, display_name=user.display_name, role=user.role,
                   role_description=role.description, is_admin=role.is_admin,
                   tools=[t for t in TOOLS if role.tools is None or t in role.tools])


# ------------------------------------------------------------------ auth
_attempts: dict[str, deque] = defaultdict(deque)
_attempts_lock = threading.Lock()


@router.post("/auth/login", response_model=LoginResponse)
def login(body: LoginRequest):
    key = body.username.lower()
    with _attempts_lock:  # slow down password guessing
        window = _attempts[key]
        while window and time.monotonic() - window[0] > 60:
            window.popleft()
        if len(window) >= settings.login_attempts_per_minute:
            raise HTTPException(429, "Too many login attempts. Wait a minute.")
        window.append(time.monotonic())
    with session() as s:
        user = s.scalars(select(User).where(User.username == body.username)).first()
    if user is None or not user.active or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Wrong username or password")
    return LoginResponse(access_token=create_token(user.id, user.username), user=user_out(user))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user_out(user)


# ------------------------------------------------------------------ conversations
def own_conversation(s, conversation_id: str, user: User) -> Conversation:
    conv = s.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(404, "Conversation not found")
    return conv


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(user: User = Depends(current_user)):
    with session() as s:
        return s.scalars(select(Conversation).where(Conversation.user_id == user.id)
                         .order_by(Conversation.updated_at.desc()).limit(100)).all()


@router.post("/conversations", response_model=ConversationOut)
def create_conversation(user: User = Depends(current_user)):
    with session() as s:
        conv = Conversation(user_id=user.id)
        s.add(conv)
        s.commit()
        return conv


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str, user: User = Depends(current_user)):
    with session() as s:
        conv = own_conversation(s, conversation_id, user)
        return ConversationDetail(id=conv.id, title=conv.title, created_at=conv.created_at, updated_at=conv.updated_at,
                                  messages=[MessageOut.model_validate(m, from_attributes=True) for m in conv.messages])


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, user: User = Depends(current_user)):
    with session() as s:
        s.delete(own_conversation(s, conversation_id, user))
        s.commit()


def history_for_llm(messages: list[ChatMessage]) -> list[dict]:
    """Previous turns as plain text (plus the SQL used), so follow-ups like 'now only 2024' work."""
    turns = []
    for m in messages[-settings.history_turns * 2:]:
        content = m.content
        if m.role == "assistant":
            sql = [b["sql"] for b in m.blocks if b.get("kind") == "result"]
            if sql:
                content += "\n\n(SQL used: " + " | ".join(sql) + ")"
        turns.append({"role": m.role, "content": content or "(empty)"})
    return turns


def sse(event: dict) -> str:
    """One Server-Sent Event: 'data: <json>' followed by a blank line."""
    return f"data: {json.dumps(event, default=str, ensure_ascii=False)}\n\n"


@router.post("/conversations/{conversation_id}/messages")
def ask(conversation_id: str, body: AskRequest, user: User = Depends(current_user),
        llm: LLMRouter = Depends(llm_router)):
    """Ask a question. The answer streams back as Server-Sent Events (text/event-stream)."""
    with session() as s:
        conv = own_conversation(s, conversation_id, user)
        history = history_for_llm(conv.messages)
        masker = Masker(conv.mask_map)
        s.add(ChatMessage(conversation_id=conv.id, role="user", content=body.question))
        if conv.title == "New chat":
            conv.title = body.question[:80]
        conv.updated_at = now()
        s.commit()
    ctx = user_context(user, conversation_id)

    def events():
        yield sse({"type": "start", "conversation_id": conversation_id})
        for event in stream_agent(body.question, ctx, llm, masker, history):
            if event["type"] in ("done", "error"):
                with session() as s:
                    done = event["type"] == "done"
                    msg = ChatMessage(conversation_id=conversation_id, role="assistant",
                                      content=event["text"] if done else f"Error: {event['message']}",
                                      blocks=event.get("blocks", []), usage=event.get("usage", {}))
                    if done and event.get("example_id"):
                        msg.usage = {**msg.usage, "example_id": event["example_id"]}
                    s.add(msg)
                    conv = s.get(Conversation, conversation_id)
                    conv.mask_map = masker.mapping
                    conv.updated_at = now()
                    s.commit()
                    event = {**event, "message_id": msg.id}
            yield sse(event)

    # X-Accel-Buffering: tell proxies (nginx) not to buffer, so events arrive immediately
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/messages/{message_id}/feedback", status_code=204)
def message_feedback(message_id: str, body: FeedbackRequest, user: User = Depends(current_user)):
    """Thumbs up/down. Also verifies (or forgets) the SQL example the agent learned from it."""
    with session() as s:
        msg = s.get(ChatMessage, message_id)
        if msg is None or msg.conversation.user_id != user.id:
            raise HTTPException(404, "Message not found")
        msg.feedback = 1 if body.helpful else -1
        example_id = (msg.usage or {}).get("example_id")
        s.commit()
    if example_id:
        knowledge.feedback(example_id, body.helpful)


# ------------------------------------------------------------------ results and Excel
def load_result(result_id: str, user: User):
    try:
        return results.load(result_id, user_context(user))
    except LookupError:
        raise HTTPException(404, "Result not found") from None
    except PermissionError:
        raise HTTPException(403, "Not your result") from None


def xlsx(content: bytes, title: str) -> Response:
    return Response(content, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{excel.filename(title)}"'})


@router.get("/results/{result_id}", response_model=ResultOut)
def get_result(result_id: str, user: User = Depends(current_user)):
    return load_result(result_id, user)


@router.get("/results/{result_id}/excel")
def result_excel(result_id: str, user: User = Depends(current_user)):
    result = load_result(result_id, user)
    return xlsx(excel.build_report(result.title, user.display_name, [{"result": result}]), result.title)


def template_report(name: str, user: User) -> bytes:
    """Run the template's query NOW, as the downloading user (their permissions and row filters),
    and fill the template. The file therefore holds only data this user may see."""
    from querynest.db import fetch_all
    from querynest.guardrails import GuardrailError, validate_sql
    from querynest.report_templates import fill_template, load_templates
    from querynest.tools import ToolContext

    template = load_templates().get(name)
    if template is None or not template.ready:
        raise HTTPException(404, "Template not available")
    ctx = ToolContext(user=user_context(user))
    try:
        checked = validate_sql(template.sql, ctx.user, ctx.columns_of, max_rows=settings.report_max_rows)
    except GuardrailError as e:
        raise HTTPException(403, f"Template not available to you: {e}") from None
    data = fetch_all(checked.sql, None, ctx.user)
    rows = [{c: results.plain(r[c]) for c in data["columns"]} for r in data["rows"]]
    return fill_template(template, data["columns"], rows, results.column_types(data["columns"], data["rows"]),
                         user.display_name)


@router.get("/report-templates")
def report_templates(user: User = Depends(current_user)):
    """Templates this user's role can use (for the UI / API clients)."""
    from querynest.report_templates import load_templates

    if not user_context(user).can_use_tool("create_template_report"):
        return []
    return [{"name": t.name, "title": t.title, "description": t.description, "ready": t.ready}
            for t in load_templates().values()]


@router.get("/reports/{report_id}/excel")
def report_excel(report_id: str, user: User = Depends(current_user)):
    with session() as s:
        report = s.get(Report, report_id)
    if report is None or (report.user_id != user.id and not ROLES[user.role].is_admin):
        raise HTTPException(404, "Report not found")
    if "template" in report.spec:
        return xlsx(template_report(report.spec["template"], user), report.title)
    sections = [{"result": load_result(sec["result_id"], user), "charts": sec.get("charts", []),
                 "pivots": sec.get("pivots", [])} for sec in report.spec["sections"]]
    return xlsx(excel.build_report(report.title, user.display_name, sections), report.title)
