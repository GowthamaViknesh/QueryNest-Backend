"""Admin endpoints: knowledge (glossary, SQL examples), audit log, usage/cost, users."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select

from querynest.api.deps import admin_user
from querynest.api.schemas import (AuditOut, ExampleIn, ExampleOut, GlossaryIn, GlossaryOut, UsageRow, UserAdminIn,
                                   UserAdminOut)
from querynest.appdb import AuditLog, GlossaryTerm, SqlExample, User, session
from querynest.auth import hash_password
from querynest.guardrails import GuardrailError, validate_sql
from querynest.permissions import ROLES

router = APIRouter(prefix="/api/admin", dependencies=[Depends(admin_user)])


# ------------------------------------------------------------------ glossary
@router.get("/glossary", response_model=list[GlossaryOut])
def list_glossary():
    with session() as s:
        return s.scalars(select(GlossaryTerm).order_by(GlossaryTerm.term)).all()


@router.post("/glossary", response_model=GlossaryOut)
def upsert_glossary(body: GlossaryIn):
    with session() as s:
        term = s.scalars(select(GlossaryTerm).where(GlossaryTerm.term == body.term)).first() or GlossaryTerm()
        for k, v in body.model_dump().items():
            setattr(term, k, v)
        s.add(term)
        s.commit()
        return term


@router.delete("/glossary/{term_id}", status_code=204)
def delete_glossary(term_id: int):
    with session() as s:
        if term := s.get(GlossaryTerm, term_id):
            s.delete(term)
            s.commit()


# ------------------------------------------------------------------ SQL examples (calibration + memory)
@router.get("/examples", response_model=list[ExampleOut])
def list_examples():
    with session() as s:
        return s.scalars(select(SqlExample).order_by(SqlExample.verified, SqlExample.created_at.desc())).all()


@router.post("/examples", response_model=ExampleOut)
def add_example(body: ExampleIn, admin: User = Depends(admin_user)):
    try:
        validate_sql(body.sql)  # never store an example the guardrails would reject
    except GuardrailError as e:
        raise HTTPException(422, f"SQL rejected: {e}") from None
    with session() as s:
        example = SqlExample(question=body.question, sql=body.sql, source="curated", verified=True,
                             created_by=admin.id)
        s.add(example)
        s.commit()
        return example


@router.post("/examples/{example_id}/verify", response_model=ExampleOut)
def verify_example(example_id: int):
    with session() as s:
        example = s.get(SqlExample, example_id)
        if example is None:
            raise HTTPException(404, "Example not found")
        example.verified = True
        s.commit()
        return example


@router.delete("/examples/{example_id}", status_code=204)
def delete_example(example_id: int):
    with session() as s:
        if example := s.get(SqlExample, example_id):
            s.delete(example)
            s.commit()


# ------------------------------------------------------------------ audit + usage
@router.get("/audit", response_model=list[AuditOut])
def audit_log(event: str | None = None, username: str | None = None, limit: int = 200):
    with session() as s:
        stmt = select(AuditLog).order_by(AuditLog.id.desc()).limit(min(limit, 1000))
        if event:
            stmt = stmt.where(AuditLog.event == event)
        if username:
            stmt = stmt.where(AuditLog.username == username)
        return s.scalars(stmt).all()


@router.get("/usage", response_model=list[UsageRow])
def usage(days: int = 30):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    with session() as s:
        rows = s.execute(
            select(AuditLog.username,
                   func.count().filter(AuditLog.event == "question"),
                   func.coalesce(func.sum(AuditLog.input_tokens).filter(AuditLog.event == "llm"), 0),
                   func.coalesce(func.sum(AuditLog.output_tokens).filter(AuditLog.event == "llm"), 0),
                   func.coalesce(func.sum(AuditLog.cost_usd).filter(AuditLog.event == "llm"), 0.0))
            .where(AuditLog.created_at >= since).group_by(AuditLog.username).order_by(AuditLog.username)).all()
    return [UsageRow(username=u, questions=q, input_tokens=i, output_tokens=o, cost_usd=round(c, 6))
            for u, q, i, o, c in rows]


# ------------------------------------------------------------------ users
@router.get("/users", response_model=list[UserAdminOut])
def list_users():
    with session() as s:
        return s.scalars(select(User).order_by(User.username)).all()


@router.post("/users", response_model=UserAdminOut)
def create_user(body: UserAdminIn):
    if body.role not in ROLES:
        raise HTTPException(422, f"Unknown role. Roles: {list(ROLES)}")
    with session() as s:
        if s.scalars(select(User).where(User.username == body.username)).first():
            raise HTTPException(409, "Username taken")
        user = User(username=body.username, display_name=body.display_name or body.username, role=body.role,
                    password_hash=hash_password(body.password),
                    attributes={"salesman": body.salesman} if body.salesman else {})
        s.add(user)
        s.commit()
        return user


@router.post("/users/{user_id}/active", response_model=UserAdminOut)
def set_active(user_id: int, active: bool, admin: User = Depends(admin_user)):
    if user_id == admin.id and not active:
        raise HTTPException(422, "You can't deactivate yourself")
    with session() as s:
        user = s.get(User, user_id)
        if user is None:
            raise HTTPException(404, "User not found")
        user.active = active
        s.commit()
        return user
