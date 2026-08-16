"""Causal daily risk evaluation and append-only decision records."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from typing import Any

import duckdb

from nordic_power_risk.config import REPO_ROOT, PipelineConfig
from nordic_power_risk.energy import soc_before
from nordic_power_risk.facts.rules import delivery_day, delivery_day_hour_count
from nordic_power_risk.ingest.duckdb_io import (
    coerce_datetime,
    coerce_float,
    read_table,
)
from nordic_power_risk.risk.controls import (
    RiskState,
    empirical_var_cvar,
    historical_loss_limit,
    scenario_losses,
)

_REQUIRED_QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)


@dataclass(frozen=True)
class RiskScheduleInterval:
    issue_time: datetime
    delivery_time: datetime
    duration_hours: float
    charge_mw: float
    discharge_mw: float
    soc_mwh: float
    degradation_cost_eur: float


@dataclass(frozen=True)
class RiskGateOutcome:
    blocked: bool
    breach: bool
    fallback_reason: str | None
    var_95_eur: float | None
    cvar_95_eur: float | None
    var_99_eur: float | None
    cvar_99_eur: float | None
    loss_limit_99_eur: float | None
    drawdown_eur: float


@dataclass(frozen=True)
class RiskStatus:
    gate_state: str
    record_count: int
    last_delivery_time: datetime | None
    fallback_reason: str | None
    drawdown_eur: float | None


@dataclass(frozen=True)
class _PendingObservation:
    available_at: datetime
    realized_loss_eur: float
    loss_limit_eur: float


def _decision_record(
    *,
    decision_timestamp: str,
    delivery_time: str | None,
    forecast_quantiles: dict[str, float],
    action: str,
    charge_mw: float,
    discharge_mw: float,
    soc_mwh: float | None,
    outcome: RiskGateOutcome,
    realized_daily_loss_eur: float | None,
    model_version: str,
    git_version: str,
) -> dict[str, object]:
    """One append-only decision-log line; the single source of the record schema."""
    return {
        "decision_timestamp": decision_timestamp,
        "delivery_time": delivery_time,
        "forecast_quantiles": forecast_quantiles,
        "action": action,
        "charge_mw": charge_mw,
        "discharge_mw": discharge_mw,
        "soc_mwh": soc_mwh,
        "var_95_eur": outcome.var_95_eur,
        "cvar_95_eur": outcome.cvar_95_eur,
        "var_99_eur": outcome.var_99_eur,
        "cvar_99_eur": outcome.cvar_99_eur,
        "loss_limit_99_eur": outcome.loss_limit_99_eur,
        "realized_daily_loss_eur": realized_daily_loss_eur,
        "drawdown_eur": outcome.drawdown_eur,
        "breach": outcome.breach,
        "fallback_reason": outcome.fallback_reason,
        "model_version": model_version,
        "git_version": git_version,
    }


def _quantile(column: str) -> float | None:
    if not column.startswith("q"):
        return None
    try:
        return float(column[1:].replace("_", "."))
    except ValueError:
        return None


def _resample_path(values: list[float], interval_count: int) -> list[float]:
    if len(values) == interval_count:
        return values
    if interval_count == 1:
        return [values[0]]
    scale = (len(values) - 1) / (interval_count - 1)
    result = []
    for index in range(interval_count):
        position = index * scale
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        result.append(values[lower] * (1.0 - weight) + values[upper] * weight)
    return result


def _price_paths_by_day(rows: list[dict[str, Any]], interval_count: int) -> list[list[float]]:
    grouped: dict[date, list[tuple[datetime, float]]] = {}
    for row in rows:
        event_time = coerce_datetime(row["event_time"])
        price = coerce_float(row["price_eur_mwh"])
        if not isfinite(price):
            continue
        grouped.setdefault(delivery_day(event_time), []).append((event_time, price))
    paths = []
    for day, day_rows in grouped.items():
        values = [price for _, price in sorted(day_rows)]
        if len(values) != delivery_day_hour_count(day) and interval_count in (23, 24, 25):
            continue
        paths.append(_resample_path(values, interval_count))
    return paths


def _model_version() -> str:
    try:
        return version("nordic-power-risk")
    except PackageNotFoundError:
        return "0+unknown"


def _git_version() -> str:
    configured = os.getenv("GIT_COMMIT")
    if configured:
        return configured
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def decision_log_path(config: PipelineConfig) -> Path:
    return config.duckdb_path.parent / "decision_log.jsonl"


class RiskEvaluator:
    """Evaluate one delivery day at a time without observing future realized prices."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        model_version: str | None = None,
        git_version: str | None = None,
    ) -> None:
        self.config = config
        self.model_version = model_version or _model_version()
        self.git_version = git_version or _git_version()
        self.state = RiskState()
        self.records: list[dict[str, object]] = []
        self._pending: list[_PendingObservation] = []
        self._input_error: str | None = None
        try:
            self._forecast_rows = read_table(config, "forecast_day_ahead")
            self._forecast_columns = list(self._forecast_rows[0]) if self._forecast_rows else []
            self._fact_rows = read_table(config, "fact_day_ahead_price")
        except duckdb.Error as exc:
            self._forecast_columns = []
            self._forecast_rows = []
            self._fact_rows = []
            self._input_error = f"missing_input:{exc.__class__.__name__}"

    def record_fallback(
        self, intervals: list[RiskScheduleInterval], reason: str
    ) -> RiskGateOutcome:
        if not intervals:
            raise ValueError("a fallback record requires at least one interval")
        return self._blocked(intervals, reason)

    def record_missing_input(self, reason: str) -> None:
        """Append one flat record for a run with no dispatchable input."""
        outcome = RiskGateOutcome(
            blocked=True,
            breach=False,
            fallback_reason=reason,
            var_95_eur=None,
            cvar_95_eur=None,
            var_99_eur=None,
            cvar_99_eur=None,
            loss_limit_99_eur=None,
            drawdown_eur=0.0,
        )
        self.records.append(
            _decision_record(
                decision_timestamp=datetime.now(UTC).isoformat(),
                delivery_time=None,
                forecast_quantiles={},
                action="flat",
                charge_mw=0.0,
                discharge_mw=0.0,
                soc_mwh=None,
                outcome=outcome,
                realized_daily_loss_eur=None,
                model_version=self.model_version,
                git_version=self.git_version,
            )
        )

    def advance(self, issue_time: datetime) -> None:
        ready = [item for item in self._pending if item.available_at <= issue_time]
        self._pending = [item for item in self._pending if item.available_at > issue_time]
        for item in sorted(ready, key=lambda value: value.available_at):
            self.state.observe(
                realized_loss_eur=item.realized_loss_eur,
                loss_limit_eur=item.loss_limit_eur,
                observed_on=delivery_day(issue_time),
            )

    def _blocked(
        self,
        intervals: list[RiskScheduleInterval],
        reason: str,
        *,
        breach: bool = False,
        metrics: tuple[float, float, float, float] | None = None,
        loss_limit: float | None = None,
    ) -> RiskGateOutcome:
        if breach:
            self.state.start_cooldown(delivery_day(intervals[0].delivery_time))
        outcome = RiskGateOutcome(
            blocked=True,
            breach=breach,
            fallback_reason=reason,
            var_95_eur=metrics[0] if metrics else None,
            cvar_95_eur=metrics[1] if metrics else None,
            var_99_eur=metrics[2] if metrics else None,
            cvar_99_eur=metrics[3] if metrics else None,
            loss_limit_99_eur=loss_limit,
            drawdown_eur=self.state.drawdown_eur,
        )
        self._record(intervals, outcome, {})
        return outcome

    def evaluate(self, intervals: list[RiskScheduleInterval]) -> RiskGateOutcome:
        if not intervals:
            raise ValueError("a risk decision requires at least one interval")
        issue_times = {item.issue_time for item in intervals}
        delivery_days = {delivery_day(item.delivery_time) for item in intervals}
        if len(issue_times) != 1 or len(delivery_days) != 1:
            raise ValueError("a risk decision must contain one issue time and delivery day")
        if self._input_error is not None:
            return self._blocked(intervals, self._input_error)

        issue_time = intervals[0].issue_time
        delivery_date = delivery_day(intervals[0].delivery_time)
        self.advance(issue_time)
        quantile_columns = {
            quantile: column
            for column in self._forecast_columns
            if (quantile := _quantile(column)) is not None
        }
        for required in _REQUIRED_QUANTILES:
            if required not in quantile_columns:
                return self._blocked(intervals, f"missing_quantile_{required}")

        forecast_by_time: dict[datetime, dict[str, Any]] = {}
        duplicate = False
        for row in self._forecast_rows:
            event_time = coerce_datetime(row["event_time"])
            if event_time in forecast_by_time:
                duplicate = True
            forecast_by_time[event_time] = row
        if duplicate:
            return self._blocked(intervals, "duplicate_forecast_interval")
        try:
            forecast_rows = [forecast_by_time[item.delivery_time] for item in intervals]
        except KeyError:
            return self._blocked(intervals, "missing_forecast_interval")

        month_start = delivery_date.replace(day=1)
        training_end = month_start
        primary = self.config.windows.get("primary")
        if primary is None:
            return self._blocked(intervals, "missing_primary_window")
        training_rows = [
            row
            for row in self._fact_rows
            if primary.start <= delivery_day(coerce_datetime(row["event_time"])) < training_end
        ]
        training_paths = _price_paths_by_day(training_rows, len(intervals))
        if not training_paths:
            return self._blocked(intervals, "missing_training_history")

        position = [item.discharge_mw - item.charge_mw for item in intervals]
        durations = [item.duration_hours for item in intervals]
        degradation = sum(item.degradation_cost_eur for item in intervals)
        loss_limit = max(
            0.0,
            historical_loss_limit(position, durations, degradation, training_paths),
        )
        price_paths = [
            [coerce_float(row[column]) for row in forecast_rows]
            for _, column in sorted(quantile_columns.items())
        ]
        if any(not all(isfinite(value) for value in path) for path in price_paths):
            return self._blocked(intervals, "nonfinite_forecast_quantile")
        losses = scenario_losses(position, durations, degradation, price_paths)
        var_95, cvar_95 = empirical_var_cvar(losses, 0.95)
        var_99, cvar_99 = empirical_var_cvar(losses, 0.99)
        metrics = (var_95, cvar_95, var_99, cvar_99)

        training_prices = [value for path in training_paths for value in path]
        lower = min(training_prices) * 1.2
        upper = max(training_prices) * 1.2
        medians = [coerce_float(row[quantile_columns[0.5]]) for row in forecast_rows]
        if any(not isfinite(value) or value < lower or value > upper for value in medians):
            return self._blocked(intervals, "bid_sanity", metrics=metrics, loss_limit=loss_limit)

        active_reason = self.state.gate_reason(delivery_day, loss_limit)
        if active_reason is not None:
            return self._blocked(intervals, active_reason, metrics=metrics, loss_limit=loss_limit)
        if cvar_99 > loss_limit:
            return self._blocked(
                intervals,
                "cvar_99_limit",
                breach=True,
                metrics=metrics,
                loss_limit=loss_limit,
            )

        fact_by_time = {
            coerce_datetime(row["event_time"]): coerce_float(row["price_eur_mwh"])
            for row in self._fact_rows
        }
        try:
            realized_path = [fact_by_time[item.delivery_time] for item in intervals]
        except KeyError:
            return self._blocked(
                intervals,
                "missing_realized_prices",
                metrics=metrics,
                loss_limit=loss_limit,
            )
        if not all(isfinite(value) for value in realized_path):
            return self._blocked(
                intervals,
                "nonfinite_realized_prices",
                metrics=metrics,
                loss_limit=loss_limit,
            )
        realized_loss = scenario_losses(position, durations, degradation, [realized_path])[0]
        available_at = max(
            item.delivery_time + timedelta(hours=item.duration_hours) for item in intervals
        )
        self._pending.append(_PendingObservation(available_at, realized_loss, loss_limit))
        outcome = RiskGateOutcome(
            blocked=False,
            breach=False,
            fallback_reason=None,
            var_95_eur=var_95,
            cvar_95_eur=cvar_95,
            var_99_eur=var_99,
            cvar_99_eur=cvar_99,
            loss_limit_99_eur=loss_limit,
            drawdown_eur=self.state.drawdown_eur,
        )
        self._record(intervals, outcome, forecast_by_time)
        return outcome

    def _record(
        self,
        intervals: list[RiskScheduleInterval],
        outcome: RiskGateOutcome,
        forecast_by_time: dict[datetime, dict[str, Any]],
    ) -> None:
        flat_soc = None
        if outcome.blocked:
            first = intervals[0]
            flat_soc = soc_before(
                first.soc_mwh,
                first.charge_mw,
                first.discharge_mw,
                first.duration_hours,
                self.config.optimizer.one_way_efficiency,
            )
        quantile_columns = {
            quantile: column
            for column in self._forecast_columns
            if (quantile := _quantile(column)) is not None
        }
        for interval in intervals:
            row = forecast_by_time.get(interval.delivery_time, {})
            quantiles = {
                str(quantile): float(row[column])
                for quantile, column in sorted(quantile_columns.items())
                if column in row and row[column] is not None
            }
            action = "flat"
            if not outcome.blocked:
                if interval.charge_mw > 0.0:
                    action = "charge"
                elif interval.discharge_mw > 0.0:
                    action = "discharge"
            self.records.append(
                _decision_record(
                    decision_timestamp=interval.issue_time.isoformat(),
                    delivery_time=interval.delivery_time.isoformat(),
                    forecast_quantiles=quantiles,
                    action=action,
                    charge_mw=0.0 if outcome.blocked else interval.charge_mw,
                    discharge_mw=0.0 if outcome.blocked else interval.discharge_mw,
                    soc_mwh=flat_soc if flat_soc is not None else interval.soc_mwh,
                    outcome=outcome,
                    realized_daily_loss_eur=self.state.last_realized_loss_eur,
                    model_version=self.model_version,
                    git_version=self.git_version,
                )
            )


def append_decision_log(config: PipelineConfig, records: list[dict[str, object]]) -> Path:
    path = decision_log_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def read_risk_status(config: PipelineConfig) -> RiskStatus:
    path = decision_log_path(config)
    if not path.exists():
        return RiskStatus("unknown", 0, None, "decision_log_missing", None)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        return RiskStatus("unknown", 0, None, "decision_log_empty", None)
    latest = records[-1]
    fallback = latest.get("fallback_reason")
    return RiskStatus(
        gate_state="blocked" if fallback else "open",
        record_count=len(records),
        last_delivery_time=(
            coerce_datetime(latest["delivery_time"])
            if latest.get("delivery_time") is not None
            else None
        ),
        fallback_reason=str(fallback) if fallback else None,
        drawdown_eur=float(latest["drawdown_eur"]),
    )


__all__ = [
    "RiskEvaluator",
    "RiskGateOutcome",
    "RiskScheduleInterval",
    "RiskStatus",
    "append_decision_log",
    "decision_log_path",
    "read_risk_status",
]
