"""MCP server (M11): lets other AI apps (Claude Desktop, IDE assistants, n8n...) use QueryNest.

MCP (Model Context Protocol) is a standard way to offer tools to AI applications. The client
starts this program and talks to it over stdin/stdout. Every call runs with the permissions
of MCP_ROLE (default: manager) and passes through the same guardrails as the chat.

Example client config (Claude Desktop, claude_desktop_config.json):
  {"mcpServers": {"querynest": {"command": "uv",
     "args": ["run", "--project", "D:/AI-Learnings/Agent-Backend", "querynest-mcp"]}}}
"""

from mcp.server.mcpserver import MCPServer

from querynest.config import settings
from querynest.permissions import system_user
from querynest.tools import ToolContext, run_tool, to_json

server = MCPServer(
    name="QueryNest",
    instructions="Query the company's sales database safely (read-only, guarded SQL). "
                 "Call list_tables, then describe_table, then run_sql_query; or just use ask_querynest.",
)


def _ctx() -> ToolContext:
    return ToolContext(user=system_user(settings.mcp_role))


@server.tool()
def list_tables() -> str:
    """List the tables you can query, with descriptions."""
    return to_json(run_tool("list_tables", {}, _ctx()))


@server.tool()
def describe_table(table_name: str) -> str:
    """Columns (name, type, meaning) and sample rows of a table, e.g. 'sales.invoices'."""
    return to_json(run_tool("describe_table", {"table_name": table_name}, _ctx()))


@server.tool()
def run_sql_query(sql: str) -> str:
    """Run one read-only PostgreSQL SELECT (schema-qualified tables). Returns up to 50 rows."""
    return to_json(run_tool("run_sql_query", {"sql": sql, "title": "MCP query"}, _ctx()))


@server.tool()
def ask_querynest(question: str) -> str:
    """Ask a question in plain language; QueryNest's own agent finds the data and answers."""
    from querynest.agent import stream_agent
    from querynest.llm import LLMRouter

    text, error = "", None
    for event in stream_agent(question, system_user(settings.mcp_role), LLMRouter()):
        if event["type"] == "done":
            text = event["text"]
        elif event["type"] == "error":
            error = event["message"]
    return text or f"Error: {error}"


def main() -> None:
    server.run()  # stdio transport


if __name__ == "__main__":
    main()
