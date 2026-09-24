"""Unit tests for the SQL validator. No database needed.

Run:  uv run pytest            (all tests)
      uv run pytest -v -k limit (only tests with 'limit' in the name, verbose)
"""

import pytest

from querynest.config import settings
from querynest.guardrails import GuardrailError, ToolBudget, validate_sql

CAP = settings.max_query_rows


# ---------------------------------------------------------------------------
# Queries that must be ALLOWED
# pytest.mark.parametrize runs the test once per item, like Jest's test.each
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "SELECT * FROM sales.invoices",
        "select customername, sum(salesamt) from sales.invoices group by 1 order by 2 desc",
        "SELECT i.docid, a.accountname FROM sales.invoices i JOIN sales.account_entries a ON a.mvchno = i.docid",
        "WITH t AS (SELECT cyyyy, SUM(salesamt) s FROM sales.invoices GROUP BY cyyyy) SELECT * FROM t",
        "SELECT docid FROM sales.invoices UNION SELECT mvchno FROM sales.account_entries",
        "SELECT * FROM sales.invoices WHERE docid IN (SELECT mvchno FROM sales.account_entries)",
        "SELECT date_trunc('month', docdt)::date, SUM(salesamt) FILTER (WHERE cyyyy = 2025) FROM sales.invoices GROUP BY 1",
        "SELECT g FROM generate_series(1, 3) AS g",
        "SELECT 1;",  # trailing semicolon is fine
    ],
)
def test_allowed(sql):
    validate_sql(sql)  # must not raise


# ---------------------------------------------------------------------------
# Queries that must be BLOCKED, with a word the error message must contain
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql, reason",
    [
        # Not a SELECT at all
        ("DELETE FROM sales.invoices", "Only SELECT"),
        ("UPDATE sales.invoices SET salesamt = 0", "Only SELECT"),
        ("INSERT INTO sales.invoices (docid) VALUES ('x')", "Only SELECT"),
        ("DROP TABLE sales.invoices", "Only SELECT"),
        ("TRUNCATE sales.invoices", "Only SELECT"),
        ("CREATE TABLE sales.x (id int)", "Only SELECT"),
        ("ALTER TABLE sales.invoices ADD COLUMN x int", "Only SELECT"),
        ("GRANT SELECT ON sales.invoices TO public", "Only SELECT"),
        ("COPY sales.invoices TO 'C:/leak.csv'", "Only SELECT"),
        ("SET statement_timeout = 0", "Only SELECT"),
        ("VACUUM", "Only SELECT"),
        ("CALL some_procedure()", "Only SELECT"),
        ("DO $$ BEGIN DELETE FROM sales.invoices; END $$", "Only SELECT"),
        ("EXPLAIN ANALYZE DELETE FROM sales.invoices", "Only SELECT"),
        # More than one statement
        ("SELECT 1; DELETE FROM sales.invoices", "exactly one"),
        ("SELECT 1; SELECT 2", "exactly one"),
        ("SELECT 1 -- harmless comment\n; DROP TABLE sales.invoices", "exactly one"),
        # Writes hidden inside a SELECT
        ("WITH d AS (DELETE FROM sales.invoices RETURNING *) SELECT * FROM d", "DELETE"),
        ("SELECT * INTO sales.copy FROM sales.invoices", "INTO"),
        ("SELECT * FROM sales.invoices FOR UPDATE", "LOCK"),
        # Tables outside the allowlist
        ("SELECT * FROM pg_catalog.pg_authid", "not allowed"),
        ("SELECT * FROM information_schema.tables", "not allowed"),
        ("SELECT * FROM public.users", "not allowed"),
        ("SELECT * FROM pg_user", "schema-qualified"),
        ("SELECT * FROM invoices", "schema-qualified"),
        ("SELECT * FROM otherdb.sales.invoices", "not allowed"),
        # Dangerous functions
        ("SELECT pg_sleep(60)", "pg_sleep"),
        ("SELECT pg_read_file('postgresql.conf')", "pg_read_file"),
        ("SELECT lo_import('C:/secrets.txt')", "lo_import"),
        ("SELECT pg_terminate_backend(1234)", "pg_terminate_backend"),
        ("SELECT set_config('statement_timeout', '0', false)", "set_config"),
        ("SELECT query_to_xml('DELETE FROM sales.invoices', true, true, '')", "query_to_xml"),
        ("SELECT * FROM dblink('host=evil', 'SELECT 1') AS t(x int)", "dblink"),
        # Garbage
        ("SELEC * FORM sales.invoices", "Could not parse"),
        ("", "exactly one"),
    ],
)
def test_blocked(sql, reason):
    with pytest.raises(GuardrailError, match=reason):
        validate_sql(sql)


# ---------------------------------------------------------------------------
# Forced LIMIT
# ---------------------------------------------------------------------------
def test_limit_added_when_missing():
    checked = validate_sql("SELECT * FROM sales.invoices")
    assert checked.sql.endswith(f"LIMIT {CAP}")
    assert checked.changes  # the change is reported


def test_small_limit_kept():
    checked = validate_sql("SELECT * FROM sales.invoices LIMIT 5")
    assert checked.sql.endswith("LIMIT 5")
    assert checked.changes == []


def test_large_limit_lowered():
    checked = validate_sql(f"SELECT * FROM sales.invoices LIMIT {CAP * 100}")
    assert checked.sql.endswith(f"LIMIT {CAP}")


def test_limit_all_replaced():
    checked = validate_sql("SELECT * FROM sales.invoices LIMIT ALL")
    assert checked.sql.endswith(f"LIMIT {CAP}")


def test_limit_applies_to_whole_union():
    checked = validate_sql("SELECT docid FROM sales.invoices UNION SELECT mvchno FROM sales.account_entries")
    assert checked.sql.endswith(f"LIMIT {CAP}")


def test_inner_limit_does_not_count():
    # Only the OUTER limit protects us; a limit inside a subquery doesn't
    checked = validate_sql("SELECT * FROM (SELECT * FROM sales.invoices LIMIT 5) t")
    assert checked.sql.endswith(f"LIMIT {CAP}")


# ---------------------------------------------------------------------------
# Tool budget
# ---------------------------------------------------------------------------
def test_tool_budget_stops_after_limit():
    calls = []
    budget = ToolBudget(lambda name, args: calls.append(name) or "ok", limit=2)
    assert budget.run("a", {}) == "ok"
    assert budget.run("b", {}) == "ok"
    third = budget.run("c", {})
    assert "limit reached" in third["error"]
    assert calls == ["a", "b"]  # the third tool never ran
