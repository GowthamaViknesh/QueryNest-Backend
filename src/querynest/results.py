"""Result sets (M6): full query results kept server-side, referenced by id.

Design rule: the LLM decides the query and the shape; the full rows go DB -> UI/Excel
directly. The LLM only sees the first LLM_RESULT_ROWS rows (and, for cloud models, masked).
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from querynest.appdb import ResultSet, session
from querynest.permissions import UserContext


def plain(value: Any) -> Any:
    """Postgres values -> JSON-friendly values (numbers stay numbers, dates become ISO text)."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def column_types(columns: list[str], rows: list[dict[str, Any]]) -> dict[str, str]:
    types = {}
    for col in columns:
        sample = next((r[col] for r in rows if r.get(col) is not None), None)
        if isinstance(sample, (int, float, Decimal)) and not isinstance(sample, bool):
            types[col] = "number"
        elif isinstance(sample, (date, datetime)):
            types[col] = "date"
        else:
            types[col] = "text"
    return types


def save(user: UserContext, sql: str, columns: list[str], rows: list[dict[str, Any]],
         truncated: bool, title: str) -> ResultSet:
    result = ResultSet(
        user_id=user.user_id, conversation_id=user.conversation_id, title=title[:200] or "Query result",
        sql=sql, columns=columns, column_types=column_types(columns, rows),
        rows=[{c: plain(r[c]) for c in columns} for r in rows],
        row_count=len(rows), truncated=truncated,
    )
    with session() as s:
        s.add(result)
        s.commit()
    return result


def load(result_id: str, user: UserContext) -> ResultSet:
    """Fetch a result the user is allowed to see: their own, or any for admins."""
    with session() as s:
        result = s.get(ResultSet, result_id)
    if result is None:
        raise LookupError(f"Result '{result_id}' not found")
    if result.user_id != user.user_id and not user.role_def.is_admin:
        raise PermissionError("This result belongs to another user")
    return result
