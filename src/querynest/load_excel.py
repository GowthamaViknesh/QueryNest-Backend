"""Load the Excel files in Agent-Backend/data/ into Postgres tables.

Run:  uv run load-excel

Safe to run again: each table is dropped and recreated, all inside ONE transaction.
If anything fails halfway, Postgres rolls everything back, so you never end up
with a half-loaded table.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from psycopg import sql

from querynest.config import settings

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SCHEMA = "sales"

MONEY = "numeric(14,3)"  # Omani Rial has 3 decimals; numeric keeps totals exact (float would not)

# ---------------------------------------------------------------------------
# One entry per table: which Excel file, and each column's Postgres type + comment.
# Column names are the Excel headers, lowercased.
# The comments are what the LLM will read in describe_table, so clear wording
# here directly improves the SQL it writes. Edit them freely and re-run.
# ---------------------------------------------------------------------------
TABLES: dict[str, dict[str, Any]] = {
    "invoices": {
        "file": "salesall.xlsx",
        "comment": "Credit sales invoice lines (one row per invoice line). Amounts in OMR.",
        "columns": {
            "salinvtype":     ("text",    "Sales invoice type (always 'Credit Sales')"),
            "invtype":        ("text",    "Document type (always 'Invoice')"),
            "divisionmastid": ("bigint",  "Division internal ID"),
            "branchmastid":   ("bigint",  "Branch internal ID"),
            "divcode":        ("text",    "Division code, e.g. TE"),
            "branchcode":     ("text",    "Branch code, e.g. TEDO"),
            "docid":          ("text",    "Invoice number, e.g. TE/SI/2017000010. Joins to account_entries.mvchno"),
            "docdt":          ("date",    "Invoice date"),
            "refno":          ("text",    "Customer reference / PO number (often empty)"),
            "customername":   ("text",    "Customer name"),
            "brandname":      ("text",    "Product brand"),
            "mname":          ("text",    "Sales category, e.g. 'EQUIPMENT SALES', 'STATIONERY SALES'"),
            "salesamt":       (MONEY,     "Sales amount (net)"),
            "salesmna":       ("text",    "Salesman name"),
            "tst":            ("text",    "Transaction source type (always 'tINV')"),
            "recid":          ("bigint",  "Invoice internal ID"),
            "salestype":      ("text",    "Sales type (always 'Credit')"),
            "customercode":   ("text",    "Customer code"),
            "grossamt":       (MONEY,     "Gross amount before discount"),
            "totdisamt":      (MONEY,     "Total discount amount"),
            "gdgross":        (MONEY,     "Gross amount after discount (unconfirmed)"),
            "empmastid":      ("bigint",  "Salesman employee internal ID"),
            "costvalue":      (MONEY,     "Cost value of goods sold"),
            "cmonth":         ("text",    "Month name of docdt, e.g. 'February'"),
            "cyyyy":          ("integer", "Year of docdt"),
            "cmn":            ("integer", "Month number (1-12) of docdt"),
            "customerid":     ("bigint",  "Customer internal ID"),
            "totqty":         (MONEY,     "Total quantity"),
            "slmincentive":   (MONEY,     "Salesman incentive (unconfirmed)"),
            "srmincentive":   (MONEY,     "Sales rep incentive (unconfirmed)"),
        },
    },
    "account_entries": {
        "file": "salesaccount.xlsx",
        "comment": "Accounting (ledger) entries posted for sales invoices: debit/credit per account. Amounts in OMR.",
        "columns": {
            "mvchno":      ("text",  "Main voucher number = invoice number. Joins to invoices.docid"),
            "vchno":       ("text",  "Sub-voucher number"),
            "vchdt":       ("date",  "Voucher date"),
            "accountname": ("text",  "Ledger account, e.g. 'BATTERY SALES', 'OUTPUT 5% OMAN VAT'"),
            "subledger":   ("text",  "Sub-ledger party, usually the customer on debtor entries (often empty)"),
            "dramount":    (MONEY,   "Debit amount"),
            "cramount":    (MONEY,   "Credit amount"),
        },
    },
}


def to_db_value(value: Any, pg_type: str) -> Any:
    """Convert one pandas/numpy value into a plain Python value psycopg understands."""
    if pd.isna(value):
        return None  # empty Excel cell -> SQL NULL
    if pg_type == "date":
        return value.date() if isinstance(value, (pd.Timestamp, datetime)) else date.fromisoformat(str(value)[:10])
    if pg_type == "text":
        text = str(value).strip()  # Excel pads some values, e.g. 'February '
        return text or None
    if pg_type in ("bigint", "integer"):
        return int(value)
    return round(float(value), 3)  # numeric: round away float noise like 20.000000000000004


def read_excel(file: Path, columns: dict[str, tuple[str, str]]) -> pd.DataFrame:
    df = pd.read_excel(file)
    df.columns = [c.strip().lower() for c in df.columns]

    # Fail loudly if the Excel headers don't match what we expect
    missing = set(columns) - set(df.columns)
    extra = set(df.columns) - set(columns)
    if missing or extra:
        raise ValueError(f"{file.name}: missing columns {sorted(missing)}, unexpected columns {sorted(extra)}")
    return df[list(columns)]  # same order as the table definition


def fake_labels(values: pd.Series, prefix: str, width: int) -> dict[str, str]:
    """Map each distinct real value to a fake label: 'ACME LLC' -> 'Customer 0001'.
    Sorted, so the same data always gets the same labels on every run."""
    distinct = sorted({str(v).strip() for v in values.dropna()} - {""})
    return {real: f"{prefix}{i:0{width}d}" for i, real in enumerate(distinct, start=1)}


def anonymize(frames: dict[str, pd.DataFrame]) -> None:
    """Replace identifying values with consistent fake labels (changes the DataFrames in place).

    Consistent = the same real customer gets the same fake name everywhere, in BOTH tables,
    so "top 10 customers" and joins by name still give correct answers.
    """
    invoices, entries = frames["invoices"], frames["account_entries"]

    def replace(df: pd.DataFrame, col: str, mapping: dict[str, str]) -> None:
        df[col] = df[col].map(lambda v: mapping.get(str(v).strip()) if pd.notna(v) else None)

    # Customer names live in invoices.customername AND account_entries.subledger: one shared mapping
    parties = fake_labels(pd.concat([invoices["customername"], entries["subledger"]]), "Customer ", 4)
    replace(invoices, "customername", parties)
    replace(entries, "subledger", parties)
    replace(invoices, "salesmna", fake_labels(invoices["salesmna"], "Salesman ", 2))
    replace(invoices, "customercode", fake_labels(invoices["customercode"], "CUST-", 4))
    invoices["refno"] = None  # free text (customer PO numbers): drop it entirely

    print(f"  Anonymized: {len(parties):,} customer/party names, "
          f"{invoices['salesmna'].nunique()} salesmen, {invoices['customercode'].nunique():,} customer codes")


def load_table(conn: psycopg.Connection, table: str, spec: dict[str, Any], df: pd.DataFrame) -> int:
    columns: dict[str, tuple[str, str]] = spec["columns"]
    full_name = sql.Identifier(SCHEMA, table)  # safely quoted "sales"."invoices"

    # 1. Recreate the table
    conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(full_name))
    conn.execute(
        sql.SQL("CREATE TABLE {} ({})").format(
            full_name,
            sql.SQL(", ").join(
                sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(pg_type))
                for col, (pg_type, _) in columns.items()
            ),
        )
    )

    # 2. Comments: the table's and each column's description, stored inside Postgres
    conn.execute(sql.SQL("COMMENT ON TABLE {} IS {}").format(full_name, sql.Literal(spec["comment"])))
    for col, (_, comment) in columns.items():
        conn.execute(
            sql.SQL("COMMENT ON COLUMN {}.{} IS {}").format(full_name, sql.Identifier(col), sql.Literal(comment))
        )

    # 3. Bulk load with COPY: streams all rows in one go, much faster than INSERT per row
    types = [pg_type for pg_type, _ in columns.values()]
    copy_stmt = sql.SQL("COPY {} ({}) FROM STDIN").format(
        full_name, sql.SQL(", ").join(map(sql.Identifier, columns))
    )
    with conn.cursor() as cur, cur.copy(copy_stmt) as copy:
        for row in df.itertuples(index=False):
            copy.write_row([to_db_value(v, t) for v, t in zip(row, types)])

    # 4. Check: rows in Postgres must equal rows in Excel
    db_count = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(full_name)).fetchone()[0]
    if db_count != len(df):
        raise RuntimeError(f"{table}: Excel has {len(df)} rows but table has {db_count}")
    return db_count


def main() -> None:
    print(f"Connecting to {settings.postgres_db} on {settings.postgres_host}:{settings.postgres_port} ...")
    # 'with' = one transaction: commit at the end if all went well, rollback on any error
    # Read every file first: anonymization needs all tables at once (shared name mapping)
    frames = {}
    for table, spec in TABLES.items():
        print(f"Reading {spec['file']} ...")
        frames[table] = read_excel(DATA_DIR / spec["file"], spec["columns"])
    if settings.anonymize_data:
        anonymize(frames)
    else:
        print("  WARNING: ANONYMIZE_DATA is off, real names will be loaded")

    with psycopg.connect(**settings.postgres_conninfo()) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(SCHEMA)))
        for table, spec in TABLES.items():
            print(f"Loading {SCHEMA}.{table} ...")
            count = load_table(conn, table, spec, frames[table])
            print(f"  {count:,} rows loaded")
        # Recreated tables lost their GRANTs and row policies: re-apply them (same transaction)
        from querynest.setup_db import roles_exist, sync_data_permissions
        if roles_exist(conn):
            sync_data_permissions(conn)
            print("  Permissions re-applied")
        else:
            print("  Run `uv run setup-db` next to create roles and permissions")
    print("Done. All tables committed.")


if __name__ == "__main__":
    main()
