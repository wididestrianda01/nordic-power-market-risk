"""Monitoring report (Phase 5): drift + health metrics from frozen pipeline artifacts.

Reads the persisted forecast/feature/fact tables and the append-only decision log,
then emits a JSON summary plus an Evidently HTML drift report. Missing inputs are
reported as absent rather than raised — monitoring observes, it does not gate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, time
from typing import Any

import duckdb
import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

from nordic_power_risk.config import PipelineConfig
from nordic_power_risk.ingest.duckdb_io import get_connection
from nordic_power_risk.risk.run import decision_log_path

MONITORED_TABLES = ("forecast_day_ahead", "feature_day_ahead", "fact_day_ahead_price")
_DRIFT_COLUMNS = (
    "price_eur_mwh",
    "price_lag_24h",
    "price_lag_168h",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
)


@dataclass(frozen=True)
class MonitoringResult:
    missingness: dict[str, float]
    forecast_mae: float | None
    interval_coverage_80: float | None
    realized_pnl_eur: float | None
    max_drawdown_eur: float | None
    breach_count: int
    optimizer_failures: int
    data_latency_hours: float | None
    drift_share: float | None
    summary_path: str
    drift_report_path: str


def _read_df(conn: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame | None:
    try:
        return conn.execute(f"SELECT * FROM {table}").fetchdf()
    except Exception:
        return None


def _missingness(df: pd.DataFrame | None) -> float:
    if df is None or df.empty:
        return 1.0
    return float(df.isna().mean().mean())


def _decision_records(config: PipelineConfig) -> list[dict[str, Any]]:
    path = decision_log_path(config)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _forecast_metrics(
    forecast: pd.DataFrame | None, price: pd.DataFrame | None
) -> tuple[float | None, float | None]:
    if forecast is None or price is None or forecast.empty or price.empty:
        return None, None
    required = {"q0_5", "q0_1", "q0_9"}
    if not required.issubset(forecast.columns) or "price_eur_mwh" not in price.columns:
        return None, None
    merged = forecast[["event_time", "q0_5", "q0_1", "q0_9"]].merge(
        price[["event_time", "price_eur_mwh"]], on="event_time", how="inner"
    )
    if merged.empty:
        return None, None
    mae = float((merged["q0_5"] - merged["price_eur_mwh"]).abs().mean())
    covered = (merged["price_eur_mwh"] >= merged["q0_1"]) & (
        merged["price_eur_mwh"] <= merged["q0_9"]
    )
    return mae, float(covered.mean())


def _drift_report(reference: pd.DataFrame, current: pd.DataFrame, html_path: str) -> float | None:
    columns = [c for c in _DRIFT_COLUMNS if c in reference.columns and c in current.columns]
    if len(columns) < 2 or reference.empty or current.empty:
        return None
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[columns], current_data=current[columns])
    report.save_html(html_path)
    for metric in report.as_dict().get("metrics", []):
        result = metric.get("result", {})
        if "share_of_drifted_columns" in result:
            return float(result["share_of_drifted_columns"])
    return None


def _drift_dataframes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if df.empty or "event_time" not in df.columns:
        return None
    ordered = df.sort_values("event_time")
    midpoint = ordered["event_time"].iloc[len(ordered) // 2]
    return ordered[ordered["event_time"] <= midpoint], ordered[ordered["event_time"] > midpoint]


def run_monitoring(config: PipelineConfig) -> MonitoringResult:
    conn = get_connection(config.duckdb_path)
    try:
        forecast = _read_df(conn, "forecast_day_ahead")
        features = _read_df(conn, "feature_day_ahead")
        price = _read_df(conn, "fact_day_ahead_price")
        settlement = _read_df(conn, "settlement")
    finally:
        conn.close()

    missingness = {
        "forecast_day_ahead": _missingness(forecast),
        "feature_day_ahead": _missingness(features),
        "fact_day_ahead_price": _missingness(price),
    }

    mae, coverage = _forecast_metrics(forecast, price)

    realized_pnl = None
    if settlement is not None and not settlement.empty and "value_eur" in settlement.columns:
        realized_pnl = float(settlement["value_eur"].sum())

    records = _decision_records(config)
    drawdowns = [float(r["drawdown_eur"]) for r in records if r.get("drawdown_eur") is not None]
    breach_count = sum(1 for r in records if r.get("breach") is True)
    optimizer_failures = sum(1 for r in records if r.get("fallback_reason") is not None)

    data_latency = None
    if price is not None and not price.empty and "event_time" in price.columns:
        latest = pd.to_datetime(price["event_time"]).max()
        end = datetime.combine(config.windows["primary"].end, time(23, 59, 59))
        data_latency = max(0.0, (end - latest).total_seconds() / 3600.0)

    drift_share = None
    drift_report_path = config.duckdb_path.parent / "drift_report.html"
    if features is not None:
        split = _drift_dataframes(features)
        if split is not None:
            drift_share = _drift_report(split[0], split[1], str(drift_report_path))

    summary_path = config.duckdb_path.parent / "monitoring.json"
    result = MonitoringResult(
        missingness=missingness,
        forecast_mae=mae,
        interval_coverage_80=coverage,
        realized_pnl_eur=realized_pnl,
        max_drawdown_eur=max(drawdowns) if drawdowns else None,
        breach_count=breach_count,
        optimizer_failures=optimizer_failures,
        data_latency_hours=data_latency,
        drift_share=drift_share,
        summary_path=str(summary_path),
        drift_report_path=str(drift_report_path),
    )
    summary_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    return result


__all__ = ["MONITORED_TABLES", "MonitoringResult", "run_monitoring"]
