"""Pivot tables (M8): rows x columns x aggregated values, with totals, computed with pandas.

Example: revenue by region (rows) by month (columns):
    create_pivot(result_id, rows=["branchcode"], columns="month", values="revenue", agg="sum")
"""

from typing import Any

import pandas as pd

from querynest.appdb import ResultSet

AGGS = {"sum": "sum", "count": "count", "avg": "mean", "min": "min", "max": "max"}
MAX_PIVOT_COLUMNS = 40
MAX_PIVOT_ROWS = 500


class PivotError(ValueError):
    pass


def label(value: Any) -> str:
    return "" if value is None else str(value)


def build_pivot(result: ResultSet, rows: list[str], values: str, columns: str | None = None,
                agg: str = "sum", title: str = "") -> dict[str, Any]:
    for col in [*rows, values, *([columns] if columns else [])]:
        if col not in result.columns:
            raise PivotError(f"Column '{col}' is not in the result. Columns: {result.columns}")
    if agg not in AGGS:
        raise PivotError(f"agg must be one of {list(AGGS)}")
    if agg != "count" and result.column_types.get(values) != "number":
        raise PivotError(f"'{values}' is not numeric; use agg='count' or pick a numeric column.")
    if not rows:
        raise PivotError("Give at least one row field.")

    df = pd.DataFrame(result.rows, columns=result.columns)
    for col in rows + ([columns] if columns else []):
        # Text labels: None -> "". Numbers and dates keep their type so they sort naturally
        # (month 2 before month 10), with missing values grouped as "".
        if result.column_types.get(col) == "text" or df[col].isna().any():
            df[col] = df[col].map(label)
    if agg != "count":
        df[values] = pd.to_numeric(df[values], errors="coerce")

    table = pd.pivot_table(df, index=rows, columns=columns, values=values, aggfunc=AGGS[agg],
                           margins=True, margins_name="Total", fill_value=0, observed=True)
    if isinstance(table, pd.Series):
        table = table.to_frame(values)
    if columns is None:
        table.columns = [values]
    if table.shape[1] > MAX_PIVOT_COLUMNS + 1:
        raise PivotError(f"'{columns}' has too many distinct values ({table.shape[1] - 1}); "
                         f"the limit is {MAX_PIVOT_COLUMNS}. Group it more coarsely first.")

    column_labels = [label(c) for c in table.columns]
    body = []
    for index, row in table.head(MAX_PIVOT_ROWS + 1).iterrows():
        keys = list(index) if isinstance(index, tuple) else [index]
        body.append([label(k) for k in keys] + [round(float(v), 3) for v in row.tolist()])
    return {
        "kind": "pivot", "title": title or f"{agg} of {values} by {', '.join(rows)}" + (f" and {columns}" if columns else ""),
        "result_id": result.id, "row_fields": rows, "column_field": columns, "value_field": values, "agg": agg,
        "header": rows + column_labels, "rows": body, "truncated": len(table) > MAX_PIVOT_ROWS + 1,
    }
