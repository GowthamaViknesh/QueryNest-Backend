"""Charts (M7): turn a stored result into a chart spec the UI (Recharts) and Excel can draw.

The LLM only chooses WHAT to plot (which columns, which chart type or "auto"). The data points
come straight from the stored result, so numbers in the chart can't be mistyped by the model.
"""

from typing import Any

from querynest.appdb import ResultSet

CHART_TYPES = ("auto", "bar", "line", "pie")
MAX_POINTS = 200
PIE_MAX_SLICES = 8


class ChartError(ValueError):
    pass


def auto_type(result: ResultSet, x: str, y: list[str]) -> str:
    """Pick a sensible chart when the LLM says 'auto':
    time on the x axis -> line; a few categories with one positive measure -> pie; else bar."""
    if result.column_types.get(x) == "date" or x.lower() in {"month", "year", "date", "day", "week", "quarter"}:
        return "line"
    values = [r.get(y[0]) for r in result.rows]
    if len(y) == 1 and 1 < len(result.rows) <= PIE_MAX_SLICES and all(isinstance(v, (int, float)) and v >= 0 for v in values):
        return "pie"
    return "bar"


def build_chart(result: ResultSet, chart_type: str = "auto", x: str | None = None,
                y: list[str] | None = None, title: str = "") -> dict[str, Any]:
    numeric = [c for c in result.columns if result.column_types.get(c) == "number"]
    others = [c for c in result.columns if c not in numeric]
    x = x or (others[0] if others else result.columns[0])
    y = y or [c for c in numeric if c != x][:3]
    if x not in result.columns:
        raise ChartError(f"Column '{x}' is not in the result. Columns: {result.columns}")
    for col in y:
        if col not in result.columns:
            raise ChartError(f"Column '{col}' is not in the result. Columns: {result.columns}")
        if result.column_types.get(col) != "number":
            raise ChartError(f"Column '{col}' is not numeric, so it can't be plotted.")
    if not y:
        raise ChartError("The result has no numeric column to plot.")
    if chart_type not in CHART_TYPES:
        raise ChartError(f"chart_type must be one of {CHART_TYPES}")
    if chart_type == "auto":
        chart_type = auto_type(result, x, y)
    if chart_type == "pie" and len(y) > 1:
        y = y[:1]  # a pie shows one measure

    # One point per x value: if the result has several rows per x (e.g. category x quarter),
    # sum them, so a bar chart never shows the same category twice.
    points: dict = {}
    for r in result.rows:
        point = points.setdefault(r.get(x), {x: r.get(x), **{c: 0.0 for c in y}})
        for c in y:
            point[c] += r.get(c) or 0
    data = list(points.values())
    summed = len(data) < len(result.rows)
    return {
        "kind": "chart", "chart_type": chart_type, "title": title or result.title,
        "x": x, "y": y, "data": data[:MAX_POINTS], "result_id": result.id,
        "truncated": len(data) > MAX_POINTS, "summed": summed,
    }
