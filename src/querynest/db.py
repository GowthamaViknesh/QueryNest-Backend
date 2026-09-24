"""Database access for the agent's tools. Always connects as the read-only agent_reader.

Every query runs in its own read-only transaction that first:
  1. SET LOCAL ROLE qn_<role>        -> Postgres enforces the user's table/column GRANTs
  2. set_config('querynest.<attr>')  -> row-level-security policies see e.g. the salesman name
  3. SET LOCAL statement_timeout     -> runaway queries are cancelled
"LOCAL" means: only for this transaction. When it ends, the pooled connection is clean again.
"""

import re
import threading
from typing import Any

from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from querynest.config import settings
from querynest.permissions import UserContext, pg_role

_pool: ConnectionPool | None = None
_lock = threading.Lock()
ATTRIBUTE_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")


def _configure(conn) -> None:
    conn.read_only = True  # every transaction on this connection starts with BEGIN READ ONLY


def pool() -> ConnectionPool:
    """Shared pool of agent_reader connections (reused instead of reconnecting per query)."""
    global _pool
    with _lock:
        if _pool is None:
            _pool = ConnectionPool(
                kwargs={**settings.agent_conninfo(), "row_factory": dict_row},
                configure=_configure, min_size=1, max_size=10, open=True, name="agent",
            )
    return _pool


def fetch_all(query: str, params: list[Any] | None, user: UserContext,
              max_rows: int | None = None) -> dict[str, Any]:
    """Run one query with the user's permissions. Returns columns, rows, row_count, truncated."""
    with pool().connection() as conn:
        conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(pg_role(user.role))))
        conn.execute(sql.SQL("SET LOCAL statement_timeout = {}").format(sql.Literal(settings.statement_timeout_ms)))
        for name, value in user.attributes.items():
            if ATTRIBUTE_NAME.match(name):
                conn.execute("SELECT set_config(%s, %s, true)", [f"querynest.{name}", str(value)])
        cur = conn.execute(query, params)
        if cur.description is None:
            return {"columns": [], "rows": [], "row_count": 0, "truncated": False}
        columns = [c.name for c in cur.description]
        rows = cur.fetchmany(max_rows + 1) if max_rows else cur.fetchall()
        truncated = max_rows is not None and len(rows) > max_rows
        rows = rows[:max_rows] if max_rows else rows
        return {"columns": columns, "rows": rows, "row_count": len(rows), "truncated": truncated}


def close_pool() -> None:
    global _pool
    with _lock:
        if _pool is not None:
            _pool.close()
            _pool = None
