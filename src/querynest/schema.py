"""Table pre-selection (idea from SQLBot): put the relevant tables' schema straight into the prompt.

Without it, the agent spends two LLM round-trips on list_tables + describe_table before it can
write SQL. With it, the model sees the most relevant tables up front and usually writes the SQL
in its first call. For databases with hundreds of tables this is also what keeps the prompt
small: only the top SCHEMA_TOP_K tables are included, ranked for THIS question.

Ranking = TF-IDF similarity between the question and each table's text (name, description,
column names and comments), plus a boost for tables that matching glossary terms point to.
Only tables and columns the user may see are considered (information_schema is filtered by the
user's Postgres role), so the prompt never mentions hidden columns.
"""

import math
import time
from collections import Counter
from dataclasses import dataclass

from querynest.config import settings
from querynest.db import fetch_all
from querynest.knowledge import Knowledge, cosine, tokens
from querynest.permissions import UserContext

GLOSSARY_BOOST = 0.5


@dataclass
class Column:
    name: str
    type: str
    comment: str


@dataclass
class TableInfo:
    name: str  # "schema.table"
    description: str
    columns: list[Column]

    def text(self) -> str:
        cols = " ".join(f"{c.name} {c.comment}" for c in self.columns)
        return f"{self.name.replace('.', ' ')} {self.description} {cols}"


_cache: dict[str, tuple[float, list[TableInfo]]] = {}


def visible_tables(user: UserContext) -> list[TableInfo]:
    """All tables (with columns) this user's role may read. Cached per role and row identity."""
    key = f"{user.role}:{sorted(user.attributes.items())}"
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < settings.schema_cache_s:
        return hit[1]
    rows = fetch_all(
        """SELECT c.table_schema || '.' || c.table_name AS table_name,
                  obj_description((quote_ident(c.table_schema) || '.' || quote_ident(c.table_name))::regclass) AS table_comment,
                  c.column_name, c.data_type,
                  col_description((quote_ident(c.table_schema) || '.' || quote_ident(c.table_name))::regclass,
                                  c.ordinal_position) AS column_comment
           FROM information_schema.columns c
           WHERE c.table_schema = ANY(%s)
           ORDER BY c.table_schema, c.table_name, c.ordinal_position""",
        [settings.allowed_schemas], user)["rows"]
    tables: dict[str, TableInfo] = {}
    for r in rows:
        if r["table_name"] not in user.role_def.tables:
            continue
        t = tables.setdefault(r["table_name"], TableInfo(r["table_name"], r["table_comment"] or "", []))
        t.columns.append(Column(r["column_name"], r["data_type"], r["column_comment"] or ""))
    result = list(tables.values())
    _cache[key] = (time.monotonic(), result)
    return result


def select_tables(question: str, user: UserContext, knowledge: Knowledge | None = None,
                  k: int | None = None) -> list[TableInfo]:
    """The k tables most relevant to the question (all of them if there are only k or fewer)."""
    k = k or settings.schema_top_k
    tables = visible_tables(user)
    if len(tables) <= k:
        return tables
    docs = [Counter(tokens(t.text())) for t in tables]
    df = Counter(term for d in docs for term in d)
    idf = {term: math.log((1 + len(docs)) / (1 + n)) + 1 for term, n in df.items()}
    q = Counter(tokens(question))
    hints = " ".join(t.sql_hint for t in (knowledge.terms if knowledge else [])).lower()
    scored = []
    for table, doc in zip(tables, docs):
        score = cosine(q, doc, idf) + (GLOSSARY_BOOST if table.name.lower() in hints else 0)
        scored.append((score, table))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [t for _, t in scored[:k]]


def schema_prompt(tables: list[TableInfo]) -> str:
    """Compact schema text (in the spirit of SQLBot's "M-Schema"): one line per column."""
    parts = ["Relevant tables (you normally don't need list_tables/describe_table for these):"]
    for t in tables:
        parts.append(f"# {t.name}: {t.description}")
        for c in t.columns:
            parts.append(f"  - {c.name} ({c.type})" + (f": {c.comment}" if c.comment else ""))
    return "\n".join(parts)


def clear_cache() -> None:
    _cache.clear()
