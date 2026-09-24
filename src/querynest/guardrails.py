"""Guardrails: checks that run BEFORE any LLM-written SQL reaches Postgres, plus the agent's
tool-call budget.

Why check in Python when Postgres already blocks writes (agent_reader is read-only)?
Defense in depth. Each layer catches what another might miss:
  1. This validator: rejects anything that isn't one plain SELECT on allowed tables,
     and gives the LLM a clear reason so it can fix its query.
  2. Connection settings (db.py): read-only transaction + statement timeout.
  3. Postgres privileges (setup_db.py): agent_reader can only SELECT on schema 'sales'.

How: sqlglot parses the SQL into a tree (an AST, "abstract syntax tree"), like a browser
turns HTML into a DOM. We inspect the tree instead of searching the text, so tricks like
comments, odd spacing or a DELETE hidden inside a WITH clause can't fool us.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from querynest.config import settings
from querynest.permissions import UserContext, system_user

# sqlglot logs a warning for statements it can't fully parse (VACUUM, CALL...). We reject
# those anyway, so keep the console clean.
logging.getLogger("sqlglot").setLevel(logging.ERROR)


class GuardrailError(Exception):
    """A query was blocked. The message is sent to the LLM, so it explains how to fix it."""


# Nodes that change data or the database, wherever they appear in the tree
WRITE_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge,
    exp.Create, exp.Drop, exp.Alter, exp.TruncateTable,
    exp.Copy, exp.Grant, exp.Set, exp.Command,
    exp.Into,  # SELECT ... INTO new_table creates a table
    exp.Lock,  # SELECT ... FOR UPDATE locks rows
)

# Functions that can sleep, read server files, run SQL from a string, or change settings
BLOCKED_FUNCTION = re.compile(
    r"^(pg_|lo_|dblink)"  # pg_sleep, pg_read_file, pg_terminate_backend, lo_import, dblink...
    r"|_to_xml|to_xmlschema$"  # query_to_xml('any SQL') runs SQL hidden in a string
    r"|^set_config$",  # changes settings, e.g. the statement timeout
    re.IGNORECASE,
)


@dataclass
class CheckedQuery:
    sql: str  # the SQL to execute (regenerated from the checked tree)
    changes: list[str] = field(default_factory=list)  # what the guardrails changed, e.g. LIMIT


def function_name(func: exp.Func) -> str:
    # Unknown functions (pg_sleep) are 'Anonymous' nodes holding their name;
    # known ones (SUM, COUNT) are their own node types.
    return func.name if isinstance(func, exp.Anonymous) else func.sql_name()


def validate_sql(sql: str, user: UserContext | None = None,
                 columns_of: Callable[[str], list[str]] | None = None,
                 max_rows: int | None = None) -> CheckedQuery:
    """Return a safe version of `sql`, or raise GuardrailError explaining what's wrong.

    user: whose permissions apply (M10). None = admin (used by tests and internal tools).
    columns_of: "schema.table" -> the columns this user may see; needed for row filters.
    max_rows: the forced LIMIT (default MAX_QUERY_ROWS; template reports use REPORT_MAX_ROWS).
    """
    user = user or system_user("admin")
    role = user.role_def
    # 1. Parse. SQL we can't parse is SQL we can't check.
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except ParseError as e:
        raise GuardrailError(f"Could not parse the SQL ({e.errors[0]['description']}). "
                             "Write one plain PostgreSQL SELECT query.") from None

    # 2. Exactly one statement: blocks "SELECT 1; DELETE FROM ..."
    if len(statements) != 1:
        raise GuardrailError(f"Send exactly one SQL statement (got {len(statements)}).")
    tree = statements[0]

    # 3. The statement must be a query (SELECT, WITH ... SELECT, UNION of SELECTs)
    if not isinstance(tree, exp.Query):
        raise GuardrailError(f"Only SELECT queries are allowed (got {tree.key.upper()}).")

    # 4. No writes anywhere inside, e.g. WITH d AS (DELETE ... RETURNING *) SELECT * FROM d
    for node in tree.walk():
        if isinstance(node, WRITE_NODES):
            raise GuardrailError(f"Only read-only SELECT queries are allowed "
                                 f"({node.key.upper()} found inside the query).")

    # 5. Only allowlisted schemas. CTE names (WITH x AS ...) are allowed unqualified.
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    allowed = {s.lower() for s in settings.allowed_schemas}
    used_tables: set[str] = set()
    for table in tree.find_all(exp.Table):
        if isinstance(table.this, exp.Func):
            continue  # table function like generate_series(); checked in step 6
        if not table.db:
            if table.name.lower() in cte_names:
                continue
            raise GuardrailError(f"Table '{table.name}' must be schema-qualified, "
                                 f"e.g. {settings.allowed_schemas[0]}.{table.name}.")
        if table.db.lower() not in allowed or table.catalog not in ("", settings.postgres_db):
            raise GuardrailError(f"Table '{table.sql(dialect='postgres')}' is not allowed. "
                                 f"Only schemas {settings.allowed_schemas} can be queried.")
        # 5b. Per-user table permissions (M10)
        full = f"{table.db}.{table.name}".lower()
        if full not in role.tables:
            raise GuardrailError(f"You don't have access to table {full}. "
                                 f"Available: {sorted(role.tables)}.")
        used_tables.add(full)

    # 6. No dangerous functions
    for func in tree.find_all(exp.Func):
        name = function_name(func)
        if BLOCKED_FUNCTION.search(name):
            raise GuardrailError(f"Function '{name}' is not allowed.")

    # 6b. Denied columns (M10): no reference by name, and no SELECT * on tables that have some
    denied = {c for t in used_tables for c in role.tables[t].denied_columns}
    if denied:
        for column in tree.find_all(exp.Column):
            if column.name.lower() in denied:
                raise GuardrailError(f"Column '{column.name}' is not available to you.")
        for select in tree.find_all(exp.Select):
            for projection in select.expressions:
                if isinstance(projection, exp.Star) or (
                        isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star)):
                    raise GuardrailError("SELECT * is not allowed on tables with restricted columns. "
                                         "List the columns you need (see describe_table).")

    # 6c. Row filters (M10): replace each filtered table with a subquery of the user's rows,
    #     e.g. sales.invoices -> (SELECT ... FROM sales.invoices WHERE salesmna = 'Salesman 12')
    changes = []
    for table in [t for t in tree.find_all(exp.Table) if not isinstance(t.this, exp.Func) and t.db]:
        full = f"{table.db}.{table.name}".lower()
        restriction = user.row_filter_value(full)
        if restriction is None:
            continue
        column, value = restriction
        if columns_of is None:
            raise GuardrailError("Row filter can't be applied (no column list).")
        condition = (exp.EQ(this=exp.column(column), expression=exp.Literal.string(value))
                     if value is not None else exp.false())  # missing attribute: no rows at all
        inner = (exp.select(*[exp.column(c) for c in columns_of(full)])
                 .from_(exp.table_(table.name, db=table.db))
                 .where(condition))
        table.replace(inner.subquery(table.alias_or_name))
        changes.append(f"only your rows of {full} are included")

    # 7. Force a LIMIT so no query can pull millions of rows
    cap = max_rows or settings.max_query_rows
    limit = tree.args.get("limit")
    current = limit.expression if isinstance(limit, exp.Limit) else None
    if not (isinstance(current, exp.Literal) and current.is_int and int(current.this) <= cap):
        tree = tree.limit(cap)
        changes.append(f"LIMIT {cap} applied" + (" (replacing a larger or missing limit)" if limit else ""))

    # Execute exactly what we checked: regenerate SQL from the tree
    return CheckedQuery(sql=tree.sql(dialect="postgres"), changes=changes)


class ToolBudget:
    """Caps how many tools the agent may call for one question.

    When the budget runs out we don't crash: the LLM gets an error telling it to answer
    with what it has, which almost always produces a useful final answer.
    """

    def __init__(self, run_tool: Callable[[str, dict[str, Any]], Any], limit: int | None = None):
        self._run_tool = run_tool
        self.limit = limit if limit is not None else settings.max_tool_calls
        self.used = 0

    def run(self, name: str, args: dict[str, Any]) -> Any:
        if self.used >= self.limit:
            print(f"  BLOCKED: tool budget of {self.limit} calls used up")
            return {"error": f"Tool call limit reached ({self.limit}). "
                             "Do not call more tools. Answer now with the data you already have."}
        self.used += 1
        return self._run_tool(name, args)
