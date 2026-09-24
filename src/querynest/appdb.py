"""The API's own tables (schema "app"), via SQLAlchemy 2 (the Python equivalent of TypeORM).

Kept apart from business data: the agent's database role cannot read these tables, and the
API's role (querynest_app) cannot read the sales data directly.
Schema changes go through Alembic migrations (alembic/versions), like TypeORM migrations.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from querynest.config import settings

SCHEMA = "app"
Json = JSON().with_variant(JSONB(), "postgresql")


def now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    display_name: Mapped[str] = mapped_column(String(128))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32))  # key of permissions.ROLES
    attributes: Mapped[dict[str, Any]] = mapped_column(Json, default=dict)  # e.g. {"salesman": "Salesman 12"}
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[int] = mapped_column(ForeignKey(f"{SCHEMA}.users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200), default="New chat")
    # Masking tokens for this chat (M11): {"[P1]": "Customer 0352", ...}
    mask_map: Mapped[dict[str, str]] = mapped_column(Json, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="conversation", order_by="ChatMessage.created_at", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "messages"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, default="")
    # Rich parts shown in the UI: results, charts, pivots, reports, SQL, tool steps
    blocks: Mapped[list[dict[str, Any]]] = mapped_column(Json, default=list)
    usage: Mapped[dict[str, Any]] = mapped_column(Json, default=dict)
    feedback: Mapped[int | None] = mapped_column(Integer, nullable=True)  # +1 / -1
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class ResultSet(Base):
    """Full query results (up to MAX_QUERY_ROWS). The LLM sees 50 rows; the UI/Excel get all."""

    __tablename__ = "result_sets"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[int | None] = mapped_column(ForeignKey(f"{SCHEMA}.users.id"), nullable=True, index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    title: Mapped[str] = mapped_column(String(200), default="Query result")
    sql: Mapped[str] = mapped_column(Text)
    columns: Mapped[list[str]] = mapped_column(Json)
    column_types: Mapped[dict[str, str]] = mapped_column(Json, default=dict)  # "number" | "date" | "text"
    rows: Mapped[list[dict[str, Any]]] = mapped_column(Json)
    row_count: Mapped[int] = mapped_column(Integer)
    truncated: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Report(Base):
    """An Excel report: several results, charts and pivots in one workbook."""

    __tablename__ = "reports"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[int | None] = mapped_column(ForeignKey(f"{SCHEMA}.users.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(200))
    spec: Mapped[dict[str, Any]] = mapped_column(Json)  # {"sections": [...]}
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class GlossaryTerm(Base):
    """Terminology library (M9): what a business word means in this database."""

    __tablename__ = "glossary"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[int] = mapped_column(primary_key=True)
    term: Mapped[str] = mapped_column(String(100), unique=True)
    synonyms: Mapped[list[str]] = mapped_column(Json, default=list)
    meaning: Mapped[str] = mapped_column(Text)
    sql_hint: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class SqlExample(Base):
    """Question -> SQL pairs (M9): curated by admins (calibration) or learned from good answers."""

    __tablename__ = "sql_examples"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    sql: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(16), default="learned")  # "curated" | "learned"
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    uses: Mapped[int] = mapped_column(Integer, default=0)
    created_by: Mapped[int | None] = mapped_column(ForeignKey(f"{SCHEMA}.users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AuditLog(Base):
    """Every question, LLM call, tool call, query and block (M4): who, what, when, cost."""

    __tablename__ = "audit_log"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    username: Mapped[str] = mapped_column(String(64), default="")
    conversation_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    event: Mapped[str] = mapped_column(String(32), index=True)  # question|llm|tool|sql|blocked|answer|error
    provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sql: Mapped[str | None] = mapped_column(Text, nullable=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detail: Mapped[dict[str, Any]] = mapped_column(Json, default=dict)


_engine = None
_Session = None


def engine():
    global _engine, _Session
    if _engine is None:
        _engine = create_engine(settings.app_db_url(), pool_pre_ping=True, pool_size=5, max_overflow=5)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def session():
    """A new ORM session: `with session() as s: ...` (commit with s.commit())."""
    engine()
    return _Session()
