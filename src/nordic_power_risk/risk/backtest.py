"""Risk-gated wrapper around the complete causal dispatch pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nordic_power_risk.config import PipelineConfig
from nordic_power_risk.optimize.dispatch import DispatchForecast, DispatchInterval
from nordic_power_risk.optimize.run import (
    DispatchRunResult,
    MissingForecastInputError,
    _load_promoted_forecasts,
    run_energy_dispatch,
    run_flat_dispatch,
)
from nordic_power_risk.risk.run import (
    RiskEvaluator,
    RiskGateOutcome,
    RiskScheduleInterval,
    append_decision_log,
)


@dataclass(frozen=True)
class RiskBacktestResult:
    dispatch: DispatchRunResult
    decision_count: int
    blocked_decisions: int
    gate_state: str
    fallback_reason: str | None
    decision_log: Path


def _risk_schedule(
    intervals: tuple[DispatchInterval, ...],
) -> list[RiskScheduleInterval]:
    return [
        RiskScheduleInterval(
            issue_time=item.issue_time,
            delivery_time=item.delivery_time,
            duration_hours=item.duration_hours,
            charge_mw=item.charge_mw,
            discharge_mw=item.discharge_mw,
            soc_mwh=item.soc_mwh,
            degradation_cost_eur=item.degradation_cost_eur,
        )
        for item in intervals
    ]


def _flat_risk_schedule(
    forecasts: list[DispatchForecast], config: PipelineConfig
) -> list[RiskScheduleInterval]:
    return [
        RiskScheduleInterval(
            issue_time=item.issue_time,
            delivery_time=item.delivery_time,
            duration_hours=item.duration_hours,
            charge_mw=0.0,
            discharge_mw=0.0,
            soc_mwh=config.optimizer.initial_soc_mwh,
            degradation_cost_eur=0.0,
        )
        for item in sorted(forecasts, key=lambda value: value.delivery_time)
    ]


def _result(
    config: PipelineConfig,
    dispatch: DispatchRunResult,
    evaluator: RiskEvaluator,
    outcomes: list[RiskGateOutcome],
) -> RiskBacktestResult:
    path = append_decision_log(config, evaluator.records)
    record_blocked = sum(record["fallback_reason"] is not None for record in evaluator.records)
    blocked = sum(outcome.blocked for outcome in outcomes)
    decision_count = len(outcomes)
    if not outcomes and evaluator.records:
        decision_count = 1
        blocked = int(record_blocked > 0)
    latest_blocked = outcomes[-1].blocked if outcomes else blocked > 0
    fallback = (
        outcomes[-1].fallback_reason
        if outcomes
        else next(
            (
                str(record["fallback_reason"])
                for record in reversed(evaluator.records)
                if record["fallback_reason"] is not None
            ),
            None,
        )
    )
    return RiskBacktestResult(
        dispatch=dispatch,
        decision_count=decision_count,
        blocked_decisions=blocked,
        gate_state="blocked" if latest_blocked else "open",
        fallback_reason=fallback,
        decision_log=path,
    )


def run_risk_backtest(config: PipelineConfig) -> RiskBacktestResult:
    """Run dispatch with pre-decision tail, drawdown, cooldown, and sanity gates."""
    evaluator = RiskEvaluator(config)
    outcomes: list[RiskGateOutcome] = []
    try:
        forecasts, _ = _load_promoted_forecasts(config)
    except MissingForecastInputError as exc:
        message = str(exc)
        dispatch = run_flat_dispatch(config, [], solver_status="missing_input")
        evaluator.record_missing_input(f"missing_input:{message}")
        return _result(config, dispatch, evaluator, outcomes)

    def energy_gate(intervals: tuple[DispatchInterval, ...]) -> bool:
        schedule = _risk_schedule(intervals)
        try:
            outcome = evaluator.evaluate(schedule)
        except ValueError as exc:
            outcome = evaluator.record_fallback(schedule, f"risk_input:{exc}")
        outcomes.append(outcome)
        return outcome.blocked

    try:
        dispatch = run_energy_dispatch(config, energy_gate=energy_gate)
    except RuntimeError as exc:
        reason = f"optimizer_failure:{exc}"
        dispatch = run_flat_dispatch(config, forecasts, solver_status="optimizer_failure")
        schedule = _flat_risk_schedule(forecasts, config)
        if schedule:
            outcomes.append(evaluator.record_fallback(schedule, reason))
        else:
            evaluator.record_missing_input(reason)
    return _result(config, dispatch, evaluator, outcomes)


__all__ = ["RiskBacktestResult", "run_risk_backtest"]
