"""Shared test setup. `db_available` lets DB-dependent tests skip cleanly when Postgres is down."""

import psycopg
import pytest

from querynest.config import settings


def _db_ok() -> bool:
    try:
        with psycopg.connect(**settings.agent_conninfo(), connect_timeout=3):
            return True
    except Exception:
        return False


DB_OK = _db_ok()
needs_db = pytest.mark.skipif(not DB_OK, reason="Postgres not reachable")


@pytest.fixture(scope="session", autouse=True)
def _clean_up_test_rows():
    """Tests run against the dev database. Remember the newest row ids before the session and
    delete everything the tests added (audit rows, learned examples, results, reports) after it,
    so test runs never pollute the knowledge base or the audit log."""
    if not DB_OK:
        yield
        return
    from sqlalchemy import delete, func, select

    from querynest.appdb import AuditLog, Report, ResultSet, SqlExample, now, session

    with session() as s:
        audit_mark = s.scalar(select(func.coalesce(func.max(AuditLog.id), 0)))
        example_mark = s.scalar(select(func.coalesce(func.max(SqlExample.id), 0)))
    started = now()
    yield
    with session() as s:
        s.execute(delete(AuditLog).where(AuditLog.id > audit_mark))
        s.execute(delete(SqlExample).where(SqlExample.id > example_mark))
        s.execute(delete(ResultSet).where(ResultSet.created_at >= started))
        s.execute(delete(Report).where(Report.created_at >= started))
        s.commit()


@pytest.fixture(autouse=True)
def _classic_agent(monkeypatch):
    """Scripted fake models in older tests expect the plain agent loop. Tests for the fast path
    and suggestions switch them on explicitly."""
    monkeypatch.setattr(settings, "fast_path", False)
    monkeypatch.setattr(settings, "suggest_followups", False)


@pytest.fixture(autouse=True)
def _no_llm_retry_sleep(monkeypatch):
    """Retries back off with time.sleep; skip the waiting in tests."""
    import querynest.llm.router as router

    monkeypatch.setattr(router.time, "sleep", lambda s: None)
