"""M10: user-aware guardrails (no database needed)."""

import pytest

from querynest.guardrails import GuardrailError, validate_sql
from querynest.permissions import UserContext

SALES = UserContext(username="s", role="sales", attributes={"salesman": "Salesman 12"})
COLS = lambda table: ["docid", "customername", "salesamt", "salesmna"]  # noqa: E731


def check(sql, user=SALES):
    return validate_sql(sql, user, COLS)


def test_table_not_in_role_is_blocked():
    with pytest.raises(GuardrailError, match="don't have access"):
        check("SELECT * FROM sales.account_entries")


def test_denied_column_is_blocked():
    with pytest.raises(GuardrailError, match="costvalue"):
        check("SELECT SUM(costvalue) FROM sales.invoices")


def test_denied_column_hidden_in_subquery_is_blocked():
    with pytest.raises(GuardrailError, match="costvalue"):
        check("SELECT x FROM (SELECT costvalue AS x FROM sales.invoices) t")


def test_select_star_blocked_when_columns_are_restricted():
    with pytest.raises(GuardrailError, match="SELECT \\*"):
        check("SELECT * FROM sales.invoices")
    with pytest.raises(GuardrailError, match="SELECT \\*"):
        check("SELECT i.* FROM sales.invoices i")


def test_count_star_is_fine():
    check("SELECT COUNT(*) FROM sales.invoices")


def test_row_filter_rewrites_every_reference():
    sql = check("SELECT customername FROM sales.invoices i WHERE docid IN (SELECT docid FROM sales.invoices)").sql
    assert sql.count("salesmna = 'Salesman 12'") == 2
    assert "AS i" in sql  # the original alias still works


def test_row_filter_value_is_escaped():
    evil = UserContext(username="e", role="sales", attributes={"salesman": "x' OR '1'='1"})
    sql = validate_sql("SELECT docid FROM sales.invoices", evil, COLS).sql
    assert "'x'' OR ''1''=''1'" in sql


def test_admin_has_no_restrictions():
    admin = UserContext(username="a", role="admin")
    checked = validate_sql("SELECT * FROM sales.account_entries", admin)
    assert "salesmna" not in checked.sql
