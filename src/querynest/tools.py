"""Tools the LLM can ask us to run.

The LLM never runs these itself. It only asks: "please call describe_table with
table_name=sales.invoices". Our code looks the tool up in TOOLS, checks the user may use it,
validates the input, runs it with the user's permissions, and sends the result back.

User-aware tools (M10): every tool receives a ToolContext with the calling user, so queries
run under that user's role, and results/charts belong to that user.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Callable, Literal

import psycopg
from pydantic import BaseModel, Field, ValidationError

from querynest import charts, pivots, results
from querynest.appdb import Report, session
from querynest.config import settings
from querynest.db import fetch_all
from querynest.guardrails import GuardrailError, validate_sql
from querynest.llm.types import ToolSpec
from querynest.permissions import UserContext

log = logging.getLogger("querynest.tools")
UI_PREVIEW_ROWS = 20


@dataclass
class ToolContext:
    user: UserContext
    emit: Callable[[dict], None] = lambda event: None  # sends a block (result, chart...) to the UI
    state: dict[str, Any] = field(default_factory=dict)  # per-question memory: charts, pivots, SQL
    _columns: dict[str, list[str]] = field(default_factory=dict)

    def columns_of(self, table: str) -> list[str]:
        """Columns of `table` this user may see (cached per question)."""
        if table not in self._columns:
            schema, name = table.split(".")
            rows = fetch_all("SELECT column_name FROM information_schema.columns WHERE table_schema=%s "
                             "AND table_name=%s ORDER BY ordinal_position", [schema, name], self.user)["rows"]
            self._columns[table] = [r["column_name"] for r in rows]
        return self._columns[table]


# ---------------------------------------------------------------------------
# list_tables / describe_table
# information_schema only shows tables/columns the current role has rights on, so under
# SET ROLE qn_<role> these naturally list only what this user may see.
# ---------------------------------------------------------------------------
class ListTablesInput(BaseModel):
    pass


def list_tables(args: ListTablesInput, ctx: ToolContext) -> dict[str, Any]:
    rows = fetch_all(
        """SELECT table_schema || '.' || table_name AS table_name,
                  obj_description((quote_ident(table_schema) || '.' || quote_ident(table_name))::regclass) AS description
           FROM information_schema.tables WHERE table_schema = ANY(%s) ORDER BY 1""",
        [settings.allowed_schemas], ctx.user)["rows"]
    return {"tables": [r for r in rows if r["table_name"] in ctx.user.role_def.tables]}


class DescribeTableInput(BaseModel):
    table_name: str = Field(description="Table name from list_tables, e.g. 'sales.invoices'")


def describe_table(args: DescribeTableInput, ctx: ToolContext) -> dict[str, Any]:
    known = {t["table_name"] for t in list_tables(ListTablesInput(), ctx)["tables"]}
    name = args.table_name if "." in args.table_name else f"{settings.allowed_schemas[0]}.{args.table_name}"
    if name not in known:
        return {"error": f"Unknown table '{args.table_name}'. Available: {sorted(known)}"}
    schema, table = name.split(".")
    columns = fetch_all(
        """SELECT column_name AS name, data_type AS type,
                  col_description((quote_ident(%s) || '.' || quote_ident(%s))::regclass, ordinal_position) AS description
           FROM information_schema.columns WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position""",
        [schema, table, schema, table], ctx.user)["rows"]
    result: dict[str, Any] = {"table": name, "columns": columns}
    if settings.describe_sample_rows > 0 and columns:
        col_list = ", ".join(f'"{c["name"]}"' for c in columns)  # names come from the catalog
        result["sample_rows"] = fetch_all(f'SELECT {col_list} FROM "{schema}"."{table}" LIMIT %s',
                                          [settings.describe_sample_rows], ctx.user)["rows"]
    if (restriction := ctx.user.row_filter_value(name)) is not None:
        result["note"] = f"You only see rows where {restriction[0]} is your own value."
    return result


# ---------------------------------------------------------------------------
# run_sql_query: guardrails -> Postgres (as the user's role) -> stored result
# ---------------------------------------------------------------------------
class RunSqlQueryInput(BaseModel):
    sql: str = Field(description="One PostgreSQL SELECT query. Use schema-qualified table names.")
    title: str = Field(default="", description="Short title for this result, e.g. 'Revenue by month 2025'")


def run_sql_query(args: RunSqlQueryInput, ctx: ToolContext) -> dict[str, Any]:
    log.info("SQL by %s: %s", ctx.user.username, args.sql)
    try:
        checked = validate_sql(args.sql, ctx.user, ctx.columns_of)
    except GuardrailError as e:
        log.info("BLOCKED: %s", e)
        return {"error": f"Query blocked by guardrails: {e}"}
    try:
        data = fetch_all(checked.sql, None, ctx.user, max_rows=settings.max_query_rows)
    except psycopg.errors.QueryCanceled:
        return {"error": f"Query took longer than {settings.statement_timeout_ms / 1000:g}s and was cancelled. "
                         "Filter more (WHERE), aggregate (GROUP BY), or avoid large joins."}
    except psycopg.errors.InsufficientPrivilege as e:
        return {"error": f"Permission denied by the database: {e.diag.message_primary}"}

    stored = results.save(ctx.user, checked.sql, data["columns"], data["rows"], data["truncated"],
                          args.title or "Query result")
    # Agent memory stores the model's SQL, NOT checked.sql: the executed version may contain
    # this user's row filter (e.g. salesmna = 'Salesman 12'), which must never be reused for others.
    ctx.state.setdefault("sql", []).append(args.sql)
    ctx.state.setdefault("results", []).append(stored.id)
    ctx.state["last_columns"] = stored.columns
    ctx.emit({"kind": "result", "result_id": stored.id, "title": stored.title, "sql": checked.sql,
              "columns": stored.columns, "column_types": stored.column_types,
              "rows": stored.rows[:UI_PREVIEW_ROWS], "row_count": stored.row_count, "truncated": stored.truncated})

    shown = data["rows"][: settings.llm_result_rows]
    out: dict[str, Any] = {"result_id": stored.id, "columns": data["columns"], "rows": shown,
                           "row_count": len(data["rows"]), "executed_sql": checked.sql}
    if checked.changes:
        out["guardrails"] = checked.changes
    if len(data["rows"]) > len(shown):
        out["note"] = (f"The user sees all {len(data['rows'])} rows; you see the first {len(shown)}. "
                       "Aggregate (GROUP BY, SUM, COUNT) to reason about all of them.")
    return out


# ---------------------------------------------------------------------------
# create_chart / create_pivot / create_excel_report
# ---------------------------------------------------------------------------
class CreateChartInput(BaseModel):
    result_id: str = Field(description="result_id returned by run_sql_query")
    chart_type: Literal["auto", "bar", "line", "pie"] = Field(
        default="auto", description="auto picks: time -> line, few categories -> pie, else bar")
    x: str | None = Field(default=None, description="Column for categories / x axis")
    y: list[str] | None = Field(default=None, description="Numeric column(s) to plot")
    title: str = ""


def create_chart(args: CreateChartInput, ctx: ToolContext) -> dict[str, Any]:
    try:
        spec = charts.build_chart(results.load(args.result_id, ctx.user), args.chart_type, args.x, args.y, args.title)
    except (charts.ChartError, LookupError, PermissionError) as e:
        return {"error": str(e)}
    ctx.state.setdefault("charts", []).append(spec)
    ctx.emit(spec)
    return {"chart": spec["chart_type"], "title": spec["title"], "x": spec["x"], "y": spec["y"],
            "points": len(spec["data"]), "status": "The chart is shown to the user."}


class CreatePivotInput(BaseModel):
    result_id: str = Field(description="result_id returned by run_sql_query (use a detailed, not pre-aggregated, result)")
    rows: list[str] = Field(description="Column(s) for pivot rows")
    columns: str | None = Field(default=None, description="Column whose values become pivot columns")
    values: str = Field(description="Column to aggregate")
    agg: Literal["sum", "count", "avg", "min", "max"] = "sum"
    title: str = ""


def create_pivot(args: CreatePivotInput, ctx: ToolContext) -> dict[str, Any]:
    try:
        spec = pivots.build_pivot(results.load(args.result_id, ctx.user), args.rows, args.values,
                                  args.columns, args.agg, args.title)
    except (pivots.PivotError, LookupError, PermissionError) as e:
        return {"error": str(e)}
    ctx.state.setdefault("pivots", []).append(spec)
    ctx.emit(spec)
    preview = [dict(zip(spec["header"], row)) for row in spec["rows"][:15]]
    return {"title": spec["title"], "header": spec["header"], "rows_preview": preview,
            "total_rows": len(spec["rows"]), "status": "The pivot table is shown to the user."}


class CreateExcelReportInput(BaseModel):
    title: str = Field(description="Report title")
    result_ids: list[str] = Field(description="Results to include (each becomes a sheet). Charts and pivots "
                                              "made from them in this conversation are included automatically.")


def create_excel_report(args: CreateExcelReportInput, ctx: ToolContext) -> dict[str, Any]:
    for rid in args.result_ids:
        try:
            results.load(rid, ctx.user)
        except (LookupError, PermissionError) as e:
            return {"error": str(e)}
    sections = [{"result_id": rid,
                 "charts": [c for c in ctx.state.get("charts", []) if c["result_id"] == rid],
                 "pivots": [p for p in ctx.state.get("pivots", []) if p["result_id"] == rid]}
                for rid in args.result_ids]
    report = Report(user_id=ctx.user.user_id, title=args.title[:200], spec={"sections": sections})
    with session() as s:
        s.add(report)
        s.commit()
    block = {"kind": "report", "report_id": report.id, "title": report.title,
             "url": f"/api/reports/{report.id}/excel", "sheets": len(sections)}
    ctx.emit(block)
    return {"report_id": report.id, "status": "The Excel download button is shown to the user."}


class CreateTemplateReportInput(BaseModel):
    template: str = Field(description="Template name from the list of Excel templates in the message")


def create_template_report(args: CreateTemplateReportInput, ctx: ToolContext) -> dict[str, Any]:
    """An Excel file with a REAL PivotTable, from an admin-defined template (report_templates.py).
    The query runs when the file is downloaded, with the downloader's permissions, so the data is
    always current and a salesman's file only ever contains his own rows."""
    from querynest.report_templates import load_templates

    template = load_templates().get(args.template)
    if template is None or not template.ready:
        ready = [t.name for t in load_templates().values() if t.ready]
        return {"error": f"Unknown or unprepared template '{args.template}'. Available: {ready}"}
    try:  # check now that this user may run the template's query at all
        validate_sql(template.sql, ctx.user, ctx.columns_of, max_rows=settings.report_max_rows)
    except GuardrailError as e:
        return {"error": f"This template isn't available to you: {e}"}
    report = Report(user_id=ctx.user.user_id, title=template.title, spec={"template": template.name})
    with session() as s:
        s.add(report)
        s.commit()
    ctx.emit({"kind": "report", "report_id": report.id, "title": report.title, "template": template.name,
              "url": f"/api/reports/{report.id}/excel", "sheets": 1})
    return {"report_id": report.id, "status": "The Excel download (with a real PivotTable) is shown to the user."}


# ---------------------------------------------------------------------------
# Registry. The description is how the LLM decides WHEN to use a tool, so it matters a lot.
# ---------------------------------------------------------------------------
class Tool(BaseModel):
    name: str
    description: str
    input_model: type[BaseModel]
    run: Callable[[Any, ToolContext], Any]


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool(name="list_tables", input_model=ListTablesInput, run=list_tables,
         description="List the database tables you can query, with a short description of each. "
                     "Call this first when you don't yet know which tables exist."),
    Tool(name="describe_table", input_model=DescribeTableInput, run=describe_table,
         description="Show a table's columns (name, type, meaning) and a few sample rows. "
                     "Call this before writing SQL against a table, so you use the right columns."),
    Tool(name="run_sql_query", input_model=RunSqlQueryInput, run=run_sql_query,
         description=f"Run one read-only PostgreSQL SELECT and get the rows back (you see at most "
                     f"{settings.llm_result_rows}; the user sees all as a table). Returns a result_id for charts, "
                     "pivots and Excel. Only single SELECT statements on the listed tables are allowed."),
    Tool(name="create_chart", input_model=CreateChartInput, run=create_chart,
         description="Show a chart (bar, line, pie, or auto) of a query result. Use when a visual helps: "
                     "trends over time, shares of a total, comparisons."),
    Tool(name="create_pivot", input_model=CreatePivotInput, run=create_pivot,
         description="Show a pivot table (rows x columns x aggregated value, with totals) from a query result. "
                     "Query detailed rows first (e.g. one row per branch and month), then pivot them."),
    Tool(name="create_excel_report", input_model=CreateExcelReportInput, run=create_excel_report,
         description="Create a downloadable Excel report from one or more results, including their charts and "
                     "pivots. Use when the user asks for Excel, a report, a download or an export."),
    Tool(name="create_template_report", input_model=CreateTemplateReportInput, run=create_template_report,
         description="Create a standard Excel report with a REAL, interactive Excel PivotTable from a named "
                     "template (listed in the message). Use when the user wants a pivot report in Excel that "
                     "matches a template."),
]}


def tool_specs(user: UserContext) -> list[ToolSpec]:
    """Only the tools this user's role may use (the LLM never even sees the others)."""
    return [ToolSpec(t.name, t.description, t.input_model.model_json_schema())
            for t in TOOLS.values() if user.can_use_tool(t.name)]


def run_tool(name: str, raw_args: dict[str, Any], ctx: ToolContext) -> Any:
    """Run a tool by name. Never raises: problems go back to the LLM as {"error": ...}
    so it can correct itself (e.g. fix a SQL typo) instead of crashing our program."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"Unknown tool '{name}'. Available: {list(TOOLS)}"}
    if not ctx.user.can_use_tool(name):
        return {"error": f"Your role may not use {name}."}
    if "__invalid_json__" in raw_args:
        return {"error": "Your tool arguments were not valid JSON. Send them again."}
    try:
        args = tool.input_model.model_validate(raw_args)
    except ValidationError as e:
        return {"error": f"Invalid input for {name}: {e.errors(include_url=False)}"}
    try:
        return tool.run(args, ctx)
    except Exception as e:
        log.exception("tool %s failed", name)
        return {"error": f"{name} failed: {e}"}


def to_json(value: Any) -> str:
    """JSON for the LLM. Postgres returns Decimal and date, which plain json.dumps can't handle."""

    def convert(v: Any) -> Any:
        if isinstance(v, Decimal):
            return str(v)  # keep the exact value, e.g. "190.625"
        if isinstance(v, date):
            return v.isoformat()
        raise TypeError(f"Not JSON serializable: {type(v).__name__}")

    return json.dumps(value, default=convert, ensure_ascii=False)
