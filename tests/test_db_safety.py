"""Integration tests: real queries through run_sql_query against Postgres, per role."""

from conftest import needs_db

from querynest.config import settings
from querynest.permissions import UserContext, system_user
from querynest.tools import ToolContext, run_tool

pytestmark = needs_db
MANAGER = system_user("manager")
SALES = UserContext(username="t-sales", role="sales", attributes={"salesman": "Salesman 12"})


def query(sql: str, user: UserContext = MANAGER) -> dict:
    return run_tool("run_sql_query", {"sql": sql}, ToolContext(user=user))


def test_normal_query_works():
    assert query("SELECT COUNT(*) AS n FROM sales.invoices")["rows"] == [{"n": 10509}]


def test_postgres_specific_sql_survives_regeneration():
    result = query("SELECT date_trunc('month', docdt)::date AS m, SUM(salesamt) FILTER (WHERE cyyyy = 2025) AS s "
                   "FROM sales.invoices GROUP BY 1 ORDER BY 1 DESC LIMIT 3")
    assert "error" not in result, result
    assert result["row_count"] == 3


def test_blocked_query_never_reaches_db():
    assert "blocked by guardrails" in query("DELETE FROM sales.invoices")["error"]
    assert query("SELECT COUNT(*) AS n FROM sales.invoices")["rows"] == [{"n": 10509}]


def test_llm_sees_capped_rows_but_all_rows_are_stored():
    result = query("SELECT docid FROM sales.invoices")
    assert len(result["rows"]) == settings.llm_result_rows
    assert result["row_count"] == settings.max_query_rows  # stored for the UI / Excel
    assert "note" in result and result["guardrails"]


def test_slow_query_is_cancelled(monkeypatch):
    monkeypatch.setattr(settings, "statement_timeout_ms", 1000)
    result = query("SELECT COUNT(*) FROM sales.account_entries a CROSS JOIN sales.account_entries b")
    assert "cancelled" in result["error"]


def test_sql_error_is_returned_not_raised():
    assert "no_such_column" in query("SELECT no_such_column FROM sales.invoices")["error"]


# ------------------------------------------------------------------ M10 permissions, end to end
def test_sales_user_sees_only_own_rows():
    result = query("SELECT DISTINCT salesmna FROM sales.invoices", SALES)
    assert result["rows"] == [{"salesmna": "Salesman 12"}]


def test_sales_user_cannot_see_ledger_or_cost():
    assert "error" in query("SELECT COUNT(*) FROM sales.account_entries", SALES)
    assert "costvalue" in query("SELECT SUM(costvalue) FROM sales.invoices", SALES)["error"]


def test_sales_user_list_and_describe_hide_restricted_things():
    ctx = ToolContext(user=SALES)
    tables = [t["table_name"] for t in run_tool("list_tables", {}, ctx)["tables"]]
    assert tables == ["sales.invoices"]
    columns = [c["name"] for c in run_tool("describe_table", {"table_name": "sales.invoices"}, ctx)["columns"]]
    assert "costvalue" not in columns and "salesamt" in columns


def test_sales_user_without_salesman_attribute_sees_nothing():
    nobody = UserContext(username="t-nobody", role="sales")  # fail closed
    assert query("SELECT COUNT(*) AS n FROM sales.invoices", nobody)["rows"] == [{"n": 0}]


def test_role_without_tool_cannot_use_it():
    result = run_tool("create_excel_report", {"title": "x", "result_ids": []}, ToolContext(user=SALES))
    assert "may not use" in result["error"]
