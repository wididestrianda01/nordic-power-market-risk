"""Report figures: render the eight headline figures from pipeline outputs.

Each figure reads its source table if present and skips with a warning if the
stage that produces it has not run yet. This makes `nordic-risk figures` incremental:
running it after `settle` renders the P&L figures, and earlier runs render
only what the data supports.

Output lands in `docs/figures/` next to the report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")  # headless; no display needed in the batch/CLI shape

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from nordic_power_risk.config import PipelineConfig  # noqa: E402
from nordic_power_risk.ingest.duckdb_io import get_connection  # noqa: E402
from nordic_power_risk.risk.run import decision_log_path  # noqa: E402

FIGURES_DIR = Path("docs/figures")

_REPORT_LABELS = {
    "no_trade": "No trade",
    "heuristic": "Heuristic",
    "optimized": "Paper policy",
    "perfect_foresight": "Perfect foresight",
}


def _fig_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def _table(conn: Any, name: str) -> pd.DataFrame | None:
    try:
        return cast(pd.DataFrame, conn.execute(f"SELECT * FROM {name}").fetchdf())
    except Exception:
        return None


def _save(fig: Figure, name: str) -> Path | None:
    path = _fig_dir() / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _hero_pnl(conn: Any) -> Path | None:
    """Figure 1: total paper P&L by policy from the comparison table."""
    df = _table(conn, "comparison")
    if df is None or df.empty:
        return None
    order = ["no_trade", "heuristic", "optimized", "perfect_foresight"]
    df = df[df["policy"].isin(order)].set_index("policy").reindex(order).dropna()
    labels = [_REPORT_LABELS.get(p, p) for p in df.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#c0c0c0", "#c0c0c0", "#2a7de1", "#d0d0d0"]
    ax.bar(labels, df["total_pnl_eur"], color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Total paper P&L (EUR)")
    ax.set_title("Cumulative paper P&L net of declared costs")
    fig.autofmt_xdate()
    return _save(fig, "hero_cumulative_pnl.png")


def _quantile_fan(conn: Any) -> Path | None:
    """Figure 2: day-ahead quantile forecast fan with realized price."""
    fc = _table(conn, "forecast_day_ahead")
    if fc is None or fc.empty:
        return None
    fc = fc.sort_values("event_time")
    tail = fc.tail(168)  # one representative week
    quantiles = [c for c in fc.columns if c.startswith("q")]
    q_lo = "q0_1" if "q0_1" in quantiles else quantiles[0]
    q_hi = "q0_9" if "q0_9" in quantiles else quantiles[-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(tail["event_time"], tail["q0_5"], color="#2a7de1", label="Median")
    ax.fill_between(
        tail["event_time"], tail[q_lo], tail[q_hi], color="#2a7de1", alpha=0.25, label="10-90%"
    )
    realized = _table(conn, "fact_day_ahead_price")
    if realized is not None and not realized.empty:
        merged = tail.merge(realized, left_on="event_time", right_on="event_time", how="left")
        value_col = "price_eur_mwh" if "price_eur_mwh" in merged else merged.columns[-1]
        ax.plot(
            merged["event_time"], merged[value_col], color="black", linewidth=1.0, label="Realized"
        )
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title("Day-ahead quantile forecast, representative week")
    ax.legend()
    fig.autofmt_xdate()
    return _save(fig, "quantile_forecast_fan.png")


def _attribution_waterfall(conn: Any) -> Path | None:
    """Figure 5: P&L attribution waterfall from the attribution table."""
    df = _table(conn, "attribution")
    if df is None or df.empty:
        return None
    order = ["forecast_error", "constraint_cost", "degradation"]
    df = df[df["component"].isin(order)].set_index("component").reindex(order).dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = {
        "forecast_error": "Forecast error",
        "constraint_cost": "Constraint cost",
        "degradation": "Degradation",
    }
    names = [labels.get(c, c) for c in df.index]
    ax.bar(names, df["value_eur"], color="#2a7de1")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Gap to perfect foresight (EUR)")
    ax.set_title("Attribution of the perfect-foresight gap")
    fig.autofmt_xdate()
    return _save(fig, "pnl_attribution_waterfall.png")


def _reserve_value(conn: Any) -> Path | None:
    """Figure 7: conditional reserve-capacity value by product."""
    df = _table(conn, "dispatch_reserve")
    if df is None or df.empty or "capacity_value_eur" not in df.columns:
        return None
    grouped = df.groupby("product")["capacity_value_eur"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(grouped.index, grouped.to_list(), color="#2a7de1")
    ax.set_xlabel("Conditional capacity value (EUR)")
    ax.set_title("Reserve value by product (conditional on acceptance)")
    return _save(fig, "reserve_value_by_product.png")


def _dispatch_negative_price(conn: Any) -> Path | None:
    """Figure 4: SoC and charge/discharge on a negative-price day."""
    dispatch = _table(conn, "dispatch_energy")
    if dispatch is None or dispatch.empty:
        return None
    price = _table(conn, "fact_day_ahead_price")
    if price is None or price.empty:
        return None
    dispatch = dispatch.sort_values("delivery_time")
    # Pick the delivery day with the lowest realized price.
    price_col = "price_eur_mwh" if "price_eur_mwh" in price else price.columns[-1]
    if "delivery_time" not in price.columns:
        return None
    price = price.sort_values(price_col)
    target_day = pd.to_datetime(price.iloc[0]["delivery_time"]).date()
    day = dispatch[pd.to_datetime(dispatch["delivery_time"]).dt.date == target_day]
    if day.empty:
        return None
    net = day.get("charge_mw", pd.Series(0, index=day.index)) - day.get(
        "discharge_mw", pd.Series(0, index=day.index)
    )
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(day["delivery_time"], net, color="#2a7de1", label="Net charge (MW)")
    ax1.set_ylabel("Net charge (MW)")
    soc_col = next((c for c in ("soc_mwh", "soc") if c in day.columns), None)
    if soc_col:
        ax2 = ax1.twinx()
        ax2.plot(day["delivery_time"], day[soc_col], color="#e1782a", label="SoC (MWh)")
        ax2.set_ylabel("SoC (MWh)")
    ax1.set_title("Dispatch on the most negative-price day")
    fig.autofmt_xdate()
    return _save(fig, "dispatch_negative_price.png")


def _risk_gate(config: PipelineConfig) -> Path | None:
    """Figure 6: daily loss against the CVaR/drawdown limits from the decision log."""
    path = decision_log_path(config)
    if not path.exists():
        return None
    records = pd.read_json(path, lines=True)
    if records.empty or "realized_daily_loss_eur" not in records:
        return None
    records = records.dropna(subset=["realized_daily_loss_eur"])
    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(records))
    ax.plot(x, records["realized_daily_loss_eur"], color="black", linewidth=1.0, label="Daily P&L")
    if "loss_limit_99_eur" in records:
        ax.plot(
            x, records["loss_limit_99_eur"], color="#d02a2a", linestyle="--", label="CVaR-99 limit"
        )
    breach_col = records["breach"] if "breach" in records else pd.Series(False, index=records.index)
    breaches = records[breach_col.fillna(False).astype(bool)]
    if not breaches.empty:
        ax.scatter(
            breaches.index,
            breaches["realized_daily_loss_eur"],
            color="#d02a2a",
            marker="x",
            s=60,
            label="Breach",
        )
    ax.set_ylabel("Daily P&L (EUR)")
    ax.set_title("Risk gate: daily P&L against the CVaR-99 limit")
    ax.legend()
    return _save(fig, "risk_gate_cvar_drawdown.png")


def _pinball_ladder(conn: Any) -> Path | None:
    """Figure 3: pinball loss per rung across the benchmark ladder.

    Rendered from MLflow run metrics for the primary experiment. Skipped when
    MLflow has not recorded the ladder yet.
    """
    try:
        import mlflow
    except Exception:
        return None
    df = _table(conn, "forecast_day_ahead")
    if df is None or df.empty:
        return None
    # The ladder comparison is recorded in MLflow; surface it if queryable.
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("day-ahead-ladder")
        if experiment is None:
            return None
        runs = client.search_runs(
            experiment.experiment_id, order_by=["attributes.start_time DESC"], max_results=1
        )
        if not runs:
            return None
        metrics = runs[0].data.metrics
    except Exception:
        return None
    rungs = [k for k in ("naive", "seasonal_naive", "lear", "dnn") if f"pinball_{k}" in metrics]
    if not rungs:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [r.replace("_", " ").title() for r in rungs]
    ax.bar(labels, [metrics[f"pinball_{r}"] for r in rungs], color="#2a7de1")
    ax.set_ylabel("Pinball loss")
    ax.set_title("Forecast ladder: pinball loss by rung")
    return _save(fig, "pinball_ladder.png")


def _bridge_its(config: PipelineConfig) -> Path | None:
    """Figure 8: pre/post 15-minute imbalance-forecast difficulty.

    Rendered from the bridge chapter artifact when Phase 6 has run. Skipped
    otherwise; the chapter path is the source of truth for the exhibit.
    """
    chapter = config.duckdb_path.parent.parent / "docs" / "bridge" / "chapter.md"
    if not chapter.exists():
        return None
    # The bridge writes a numeric summary alongside the chapter; read it if present.
    summary = chapter.with_suffix(".json")
    if not summary.exists():
        return None
    try:
        import json

        data = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    pre = data.get("pre_pinball")
    post = data.get("post_pinball")
    if pre is None or post is None:
        return None
    ax.bar(["Pre (hourly)", "Post (15-min)"], [pre, post], color=["#c0c0c0", "#2a7de1"])
    ax.set_ylabel("Pinball loss")
    ax.set_title("Imbalance-forecast difficulty across the 15-minute transition")
    return _save(fig, "bridge_its.png")


def render_all(config: PipelineConfig) -> list[Path]:
    """Render every figure whose source output exists; return the written paths."""
    written: list[Path] = []
    conn = get_connection(config.duckdb_path)
    try:
        renderers: list[tuple[str, Any]] = [
            ("hero_cumulative_pnl.png", _hero_pnl(conn)),
            ("quantile_forecast_fan.png", _quantile_fan(conn)),
            ("pnl_attribution_waterfall.png", _attribution_waterfall(conn)),
            ("reserve_value_by_product.png", _reserve_value(conn)),
            ("dispatch_negative_price.png", _dispatch_negative_price(conn)),
            ("pinball_ladder.png", _pinball_ladder(conn)),
        ]
    finally:
        conn.close()
    renderers.extend(
        [
            ("risk_gate_cvar_drawdown.png", _risk_gate(config)),
            ("bridge_its.png", _bridge_its(config)),
        ]
    )
    for name, path in renderers:
        if path is None:
            print(f"skip {name}: source output not present")
        else:
            written.append(path)
            print(f"wrote {path}")
    return written


__all__ = ["render_all"]
