"""Request/response models (like NestJS DTOs). FastAPI validates requests against them and
publishes them in the OpenAPI schema, from which the frontend's TypeScript types are generated."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class UserOut(BaseModel):
    id: int
    username: str
    display_name: str
    role: str
    role_description: str
    is_admin: bool
    tools: list[str]


class LoginResponse(BaseModel):
    access_token: str
    user: UserOut


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    blocks: list[dict[str, Any]]
    usage: dict[str, Any]
    feedback: int | None
    created_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class FeedbackRequest(BaseModel):
    helpful: bool


class ResultOut(BaseModel):
    id: str
    title: str
    sql: str
    columns: list[str]
    column_types: dict[str, str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool


class GlossaryIn(BaseModel):
    term: str = Field(min_length=1, max_length=100)
    synonyms: list[str] = []
    meaning: str = Field(min_length=1)
    sql_hint: str = ""


class GlossaryOut(GlossaryIn):
    id: int


class ExampleIn(BaseModel):
    question: str = Field(min_length=1)
    sql: str = Field(min_length=1)


class ExampleOut(ExampleIn):
    id: int
    source: str
    verified: bool
    uses: int
    created_at: datetime


class AuditOut(BaseModel):
    id: int
    created_at: datetime
    username: str
    event: str
    provider: str | None
    model: str | None
    tool: str | None
    sql: str | None
    row_count: int | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_ms: int | None
    detail: dict[str, Any]


class UsageRow(BaseModel):
    username: str
    questions: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class UserAdminIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    display_name: str = ""
    role: str
    password: str = Field(min_length=8)
    salesman: str | None = None


class UserAdminOut(BaseModel):
    id: int
    username: str
    display_name: str
    role: str
    attributes: dict[str, Any]
    active: bool
