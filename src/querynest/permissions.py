"""Who may see what (M10). The single source of truth, enforced in TWO places:

1. Postgres (setup_db.py turns this into roles, column GRANTs and row-level-security policies).
   Every agent query runs under `SET LOCAL ROLE qn_<role>`, so Postgres itself hides tables,
   columns and rows the user may not see, even if all our Python checks had a bug.
2. The app (guardrails.py): blocks denied tables/columns with a clear message for the LLM, and
   rewrites queries to apply row filters, so the LLM's answer matches what the user may see.

To change permissions: edit ROLES below, then run `uv run setup-db` to sync Postgres.
"""

from dataclasses import dataclass, field

ALL_TOOLS = None  # sentinel: role may use every tool


@dataclass(frozen=True)
class TableRule:
    denied_columns: frozenset[str] = frozenset()
    # (column, user attribute): the user only sees rows where column = their attribute value.
    # e.g. ("salesmna", "salesman") -> a salesman sees only his own sales.
    row_filter: tuple[str, str] | None = None


@dataclass(frozen=True)
class RoleDef:
    description: str
    tables: dict[str, TableRule]  # "schema.table" -> rule. Tables not listed are invisible.
    tools: frozenset[str] | None = ALL_TOOLS
    is_admin: bool = False  # may manage users, knowledge, and read the audit log


FULL_SALES = {"sales.invoices": TableRule(), "sales.account_entries": TableRule()}

ROLES: dict[str, RoleDef] = {
    "admin": RoleDef("Everything, plus administration", FULL_SALES, is_admin=True),
    "manager": RoleDef("All sales and accounting data", FULL_SALES),
    "finance": RoleDef("All sales and accounting data", FULL_SALES),
    "sales": RoleDef(
        "Own invoices only; no cost, margin or incentive columns; no ledger; no Excel export",
        {
            "sales.invoices": TableRule(
                denied_columns=frozenset({"costvalue", "slmincentive", "srmincentive"}),
                row_filter=("salesmna", "salesman"),
            ),
        },
        tools=frozenset({"list_tables", "describe_table", "run_sql_query", "create_chart", "create_pivot"}),
    ),
}


def pg_role(role: str) -> str:
    """Postgres group role for an app role, e.g. 'sales' -> 'qn_sales'."""
    return f"qn_{role}"


@dataclass
class UserContext:
    """Who is asking. Passed to every tool and every query (user-aware tools, as in Vanna)."""

    username: str
    role: str
    user_id: int | None = None
    attributes: dict[str, str] = field(default_factory=dict)  # e.g. {"salesman": "Salesman 12"}
    conversation_id: str | None = None

    @property
    def role_def(self) -> RoleDef:
        return ROLES[self.role]

    def can_use_tool(self, tool: str) -> bool:
        tools = self.role_def.tools
        return tools is ALL_TOOLS or tool in tools

    def row_filter_value(self, table: str) -> tuple[str, str | None] | None:
        """(column, value) this user is restricted to on `table`, or None if unrestricted.
        value None = the required attribute is missing: the user sees no rows (fail closed)."""
        rule = self.role_def.tables.get(table)
        if not rule or not rule.row_filter:
            return None
        column, attribute = rule.row_filter
        return column, self.attributes.get(attribute) or None


# Used by the CLI and MCP server when no logged-in user exists
def system_user(role: str = "manager") -> UserContext:
    return UserContext(username=f"system:{role}", role=role)
