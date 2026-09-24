"""Excel reports (M6-M8) with XlsxWriter.

A workbook contains:
  - Cover: company, title, date, contents, and the SQL behind each sheet (auditability)
  - One data sheet per result: a formatted Excel Table (filter buttons, OMR number format,
    frozen header), ready for Insert -> PivotTable
  - Native Excel charts next to their data (they update if the data is edited)
  - Pivot sheets built from live formulas (SUMIFS / COUNTIFS / AVERAGEIFS / MINIFS / MAXIFS)
    over the data sheet, with totals. Python libraries can't write real PivotTable objects,
    so formulas are the closest "live" equivalent; values are pre-filled so every viewer shows
    them even before recalculating.
"""

import io
import re
from datetime import datetime
from typing import Any

import xlsxwriter
from xlsxwriter.utility import xl_col_to_name, xl_rowcol_to_cell

from querynest.appdb import ResultSet
from querynest.config import settings

EXCEL_CHART = {"bar": "column", "line": "line", "pie": "pie"}
PIVOT_FUNCS = {  # agg -> (function with criteria, function without criteria)
    "sum": ("SUMIFS", "SUM"), "count": ("COUNTIFS", "COUNTA"), "avg": ("AVERAGEIFS", "AVERAGE"),
    "min": ("MINIFS", "MIN"), "max": ("MAXIFS", "MAX"),
}


def safe_sheet_name(name: str, taken: set[str]) -> str:
    base = re.sub(r"[\[\]:*?/\\]", " ", name).strip()[:28] or "Sheet"
    candidate, n = base, 2
    while candidate.lower() in taken:
        candidate, n = f"{base[:25]} {n}", n + 1
    taken.add(candidate.lower())
    return candidate


def filename(title: str) -> str:
    return (re.sub(r"[^A-Za-z0-9 _-]", "", title).strip().replace(" ", "_")[:60] or "report") + ".xlsx"


def excel_value(value: Any, kind: str) -> Any:
    if value is None:
        return None
    if kind == "date" and isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return value
    return value


class ReportBuilder:
    def __init__(self, title: str, author: str):
        self.buffer = io.BytesIO()
        self.wb = xlsxwriter.Workbook(self.buffer, {"in_memory": True, "use_future_functions": True,
                                                    "default_date_format": "yyyy-mm-dd"})
        self.title, self.author = title, author
        self.taken: set[str] = set()
        self.contents: list[tuple[str, str, str]] = []  # (sheet, description, sql)
        color = settings.report_color
        f = self.wb.add_format
        self.fmt = {
            "title": f({"bold": True, "font_size": 18, "font_color": color}),
            "subtitle": f({"font_size": 11, "font_color": "#595959"}),
            "header": f({"bold": True, "font_color": "white", "bg_color": color, "border": 1}),
            "bold": f({"bold": True}),
            "number": f({"num_format": "#,##0.000"}),
            "date": f({"num_format": "yyyy-mm-dd"}),
            "total": f({"bold": True, "num_format": "#,##0.000", "top": 2}),
            "total_label": f({"bold": True, "top": 2}),
            "wrap": f({"text_wrap": True, "valign": "top", "font_color": "#595959"}),
        }
        self.cover = self.wb.add_worksheet(safe_sheet_name("Cover", self.taken))

    # ------------------------------------------------------------------ data + charts
    def add_result(self, result: ResultSet, charts: list[dict] | None = None) -> tuple[str, int]:
        """Data sheet as an Excel Table, plus native charts. Returns (sheet name, row count)."""
        name = safe_sheet_name(result.title, self.taken)
        ws = self.wb.add_worksheet(name)
        kinds = [result.column_types.get(c, "text") for c in result.columns]
        data = [[excel_value(r.get(c), k) for c, k in zip(result.columns, kinds)] for r in result.rows]
        columns = [{"header": c, "format": self.fmt["number"] if k == "number" else self.fmt["date"] if k == "date" else None}
                   for c, k in zip(result.columns, kinds)]
        last_row = max(len(data), 1)
        ws.add_table(0, 0, last_row, len(result.columns) - 1, {
            "data": data or None, "columns": columns, "style": "Table Style Medium 2",
            "name": f"Data_{len(self.contents) + 1}",
        })
        ws.freeze_panes(1, 0)
        ws.autofit()
        note = f"{len(data):,} rows" + (" (limited by MAX_QUERY_ROWS)" if result.truncated else "")
        self.contents.append((name, f"{result.title}: {note}", result.sql))

        for i, chart in enumerate(charts or []):
            self._add_chart(ws, name, result, chart, anchor_row=1 + i * 20, anchor_col=len(result.columns) + 1)
        return name, len(data)

    def _add_chart(self, ws, sheet: str, result: ResultSet, spec: dict, anchor_row: int, anchor_col: int) -> None:
        n = min(len(result.rows), 200)
        if n == 0:
            return
        chart = self.wb.add_chart({"type": EXCEL_CHART.get(spec["chart_type"], "column")})
        x_col = result.columns.index(spec["x"])
        for y in spec["y"]:
            y_col = result.columns.index(y)
            chart.add_series({
                "name": y,
                "categories": [sheet, 1, x_col, n, x_col],
                "values": [sheet, 1, y_col, n, y_col],
                **({"data_labels": {"percentage": True}} if spec["chart_type"] == "pie" else {}),
            })
        chart.set_title({"name": spec.get("title") or result.title})
        if spec["chart_type"] != "pie":
            chart.set_legend({"position": "bottom"} if len(spec["y"]) > 1 else {"none": True})
        chart.set_size({"width": 720, "height": 380})
        ws.insert_chart(anchor_row, anchor_col, chart)

    # ------------------------------------------------------------------ pivots
    def add_pivot(self, result: ResultSet, data_sheet: str, spec: dict) -> None:
        """Pivot sheet with live formulas referencing the data sheet."""
        ws = self.wb.add_worksheet(safe_sheet_name(f"Pivot {spec['title']}", self.taken))
        rows_n = max(len(result.rows), 1)

        def data_range(column: str) -> str:
            letter = xl_col_to_name(result.columns.index(column))
            return f"'{data_sheet}'!${letter}$2:${letter}${rows_n + 1}"

        with_crit, without_crit = PIVOT_FUNCS[spec["agg"]]
        value_rng = data_range(spec["value_field"])

        def formula(criteria: list[tuple[str, str]]) -> str:
            if not criteria:
                return f"={without_crit}({value_rng})"
            pairs = ",".join(f"{data_range(col)},{cell}" for col, cell in criteria)
            return f"={with_crit}({pairs})" if with_crit == "COUNTIFS" else f"={with_crit}({value_rng},{pairs})"

        ws.write(0, 0, spec["title"], self.fmt["title"])
        ws.write(1, 0, f"{spec['agg']} of {spec['value_field']} - live formulas over sheet '{data_sheet}'",
                 self.fmt["subtitle"])
        top = 3
        n_row_fields = len(spec["row_fields"])
        for c, head in enumerate(spec["header"]):
            ws.write(top, c, head, self.fmt["header"])
        kinds = {c: result.column_types.get(c, "text") for c in result.columns}
        col_field = spec["column_field"]

        for r, body in enumerate(spec["rows"], start=top + 1):
            is_total_row = body[0] == "Total"
            labels, values = body[:n_row_fields], body[n_row_fields:]
            for c, text in enumerate(labels):
                field = spec["row_fields"][c]
                value = excel_value(text, kinds[field]) if not is_total_row else text
                if kinds[field] == "number" and not is_total_row:
                    try:
                        value = float(text)
                    except ValueError:
                        pass
                fmt = self.fmt["total_label"] if is_total_row else (self.fmt["date"] if kinds[field] == "date" else None)
                ws.write(r, c, value, fmt)
            for j, cached in enumerate(values):
                col_index = n_row_fields + j
                header = spec["header"][col_index]
                criteria = []
                if not is_total_row:
                    criteria += [(f, xl_rowcol_to_cell(r, i, col_abs=True)) for i, f in enumerate(spec["row_fields"])]
                if col_field and header != "Total":
                    criteria.append((col_field, xl_rowcol_to_cell(top, col_index, row_abs=True)))
                ws.write_formula(r, col_index, formula(criteria),
                                 self.fmt["total"] if is_total_row or header == "Total" else self.fmt["number"], cached)
        # Column headers that are dates/numbers must be real values for the criteria to match
        if col_field and kinds.get(col_field) in ("date", "number"):
            for j, head in enumerate(spec["header"][n_row_fields:-1], start=n_row_fields):
                value = excel_value(head, kinds[col_field])
                if kinds[col_field] == "number":
                    try:
                        value = float(head)
                    except ValueError:
                        pass
                ws.write(top, j, value, self.fmt["header"])
        ws.freeze_panes(top + 1, n_row_fields)
        ws.autofit()

    # ------------------------------------------------------------------ cover + output
    def _write_cover(self) -> None:
        ws, f = self.cover, self.fmt
        ws.write(0, 0, settings.company_name, f["subtitle"])
        ws.write(1, 0, self.title, f["title"])
        ws.write(2, 0, f"Generated {datetime.now():%Y-%m-%d %H:%M} for {self.author}", f["subtitle"])
        ws.write(4, 0, "Contents", f["bold"])
        ws.write(4, 1, "Description", f["bold"])
        ws.write(4, 2, "Query (SQL)", f["bold"])
        for i, (sheet, desc, sql) in enumerate(self.contents, start=5):
            ws.write_url(i, 0, f"internal:'{sheet}'!A1", string=sheet)
            ws.write(i, 1, desc)
            ws.write(i, 2, sql, f["wrap"])
        ws.write(len(self.contents) + 7, 0, "Tip: data sheets are Excel Tables. Click inside one and choose "
                 "Insert > PivotTable for an interactive pivot.", f["subtitle"])
        ws.set_column(0, 0, 28)
        ws.set_column(1, 1, 45)
        ws.set_column(2, 2, 90)

    def build(self) -> bytes:
        self._write_cover()
        self.wb.close()
        return self.buffer.getvalue()


def build_report(title: str, author: str, sections: list[dict]) -> bytes:
    """sections: [{"result": ResultSet, "charts": [spec...], "pivots": [spec...]}, ...]"""
    builder = ReportBuilder(title, author)
    for section in sections:
        sheet, _ = builder.add_result(section["result"], section.get("charts"))
        for pivot in section.get("pivots", []):
            builder.add_pivot(section["result"], sheet, pivot)
    return builder.build()
