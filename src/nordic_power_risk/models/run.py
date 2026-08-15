"""Run naive/seasonal-naive/LEAR benchmark ladder over T08 rolling-origin folds
(Phase 2 tickets 01, 02).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd

from nordic_power_risk.config import PipelineConfig
from nordic_power_risk.features.split import rolling_origin_folds
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.models.baselines import (
    naive_forecast,
    quantile_forecast,
    residual_quantiles,
    seasonal_naive_forecast,
)
from nordic_power_risk.models.dnn import dnn_forecast
from nordic_power_risk.models.lear import lear_forecast
from nordic_power_risk.models.metrics import (
    diebold_mariano_test,
    interval_coverage,
    mean_pinball_per_row,
    pit_values,
    winkler_score,
)

FoldForecaster = Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.Series, pd.Series]]


def _baseline_forecaster(point_fn: Callable[[pd.DataFrame], pd.Series]) -> FoldForecaster:
    def forecaster(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        return point_fn(train), point_fn(test)

    return forecaster


RUNGS: dict[str, FoldForecaster] = {
    "naive": _baseline_forecaster(naive_forecast),
    "seasonal_naive": _baseline_forecaster(seasonal_naive_forecast),
    "lear": lear_forecast,
    "dnn": dnn_forecast,
}
COVERAGE_LOWER_Q = 0.1
COVERAGE_UPPER_Q = 0.9
COVERAGE_ALPHA = 0.2  # 1 - (upper - lower)
DM_SIGNIFICANCE_ALPHA = 0.05


@dataclass(frozen=True)
class RungResult:
    rung: str
    n_obs: int
    pinball_loss: float
    crps: float
    coverage_80: float
    winkler_80: float
    pit_mean: float
    dm_stat: float | None
    dm_pvalue: float | None
    dm_stat_vs_seasonal_naive: float | None
    dm_pvalue_vs_seasonal_naive: float | None


def run_benchmark_ladder(config: PipelineConfig) -> list[RungResult]:
    conn = get_connection(config.duckdb_path)
    try:
        df = conn.execute("SELECT * FROM feature_day_ahead").fetchdf()
    finally:
        conn.close()

    primary = config.windows["primary"]
    folds = rolling_origin_folds(primary.start, primary.end)
    event_date = df["event_time"].dt.date

    row_losses: dict[str, list[np.ndarray]] = {name: [] for name in RUNGS}
    coverage_vals: dict[str, list[float]] = {name: [] for name in RUNGS}
    winkler_vals: dict[str, list[float]] = {name: [] for name in RUNGS}
    pit_vals: dict[str, list[np.ndarray]] = {name: [] for name in RUNGS}
    n_obs: dict[str, int] = {name: 0 for name in RUNGS}
    forecast_records: dict[str, list[dict[str, object]]] = {name: [] for name in RUNGS}

    for fold in folds:
        train = df[(event_date >= fold.train_start) & (event_date < fold.train_end)]
        test = df[(event_date >= fold.test_start) & (event_date < fold.test_end)]
        for name, forecaster in RUNGS.items():
            train_point, test_point = forecaster(train, test)
            valid = test_point.notna() & test["price_eur_mwh"].notna()
            if not valid.any() or train_point.notna().sum() == 0:
                continue

            resid_q = residual_quantiles(train["price_eur_mwh"], train_point)
            q_forecasts = quantile_forecast(test_point[valid], resid_q)
            q_forecasts_arr = {q: v.to_numpy() for q, v in q_forecasts.items()}
            y_true = test.loc[valid, "price_eur_mwh"].to_numpy()

            row_losses[name].append(mean_pinball_per_row(y_true, q_forecasts_arr))
            coverage_vals[name].append(
                interval_coverage(
                    y_true, q_forecasts_arr[COVERAGE_LOWER_Q], q_forecasts_arr[COVERAGE_UPPER_Q]
                )
            )
            winkler_vals[name].append(
                winkler_score(
                    y_true,
                    q_forecasts_arr[COVERAGE_LOWER_Q],
                    q_forecasts_arr[COVERAGE_UPPER_Q],
                    COVERAGE_ALPHA,
                )
            )
            pit_vals[name].append(pit_values(y_true, q_forecasts_arr))
            n_obs[name] += int(valid.sum())

            event_times = test.loc[valid, "event_time"]
            for row_pos, event_time in enumerate(event_times):
                record: dict[str, object] = {"event_time": event_time}
                for quantile, values in q_forecasts_arr.items():
                    record[f"q{str(quantile).replace('.', '_')}"] = float(values[row_pos])
                forecast_records[name].append(record)

    concatenated = {name: np.concatenate(row_losses[name]) for name in RUNGS if row_losses[name]}
    naive_loss_series = concatenated.get("naive")
    seasonal_naive_loss_series = concatenated.get("seasonal_naive")

    results = []
    for name in RUNGS:
        if name not in concatenated:
            continue
        loss_series = concatenated[name]
        dm_stat: float | None = None
        dm_pvalue: float | None = None
        if name != "naive" and naive_loss_series is not None:
            dm_stat, dm_pvalue = diebold_mariano_test(loss_series, naive_loss_series)
        dm_stat_sn: float | None = None
        dm_pvalue_sn: float | None = None
        if name not in ("naive", "seasonal_naive") and seasonal_naive_loss_series is not None:
            dm_stat_sn, dm_pvalue_sn = diebold_mariano_test(loss_series, seasonal_naive_loss_series)
        results.append(
            RungResult(
                rung=name,
                n_obs=n_obs[name],
                pinball_loss=float(np.mean(loss_series)),
                crps=2.0 * float(np.mean(loss_series)),
                coverage_80=float(np.mean(coverage_vals[name])),
                winkler_80=float(np.mean(winkler_vals[name])),
                pit_mean=float(np.mean(np.concatenate(pit_vals[name]))),
                dm_stat=dm_stat,
                dm_pvalue=dm_pvalue,
                dm_stat_vs_seasonal_naive=dm_stat_sn,
                dm_pvalue_vs_seasonal_naive=dm_pvalue_sn,
            )
        )

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)
    for result in results:
        with mlflow.start_run(run_name=result.rung):
            mlflow.log_metrics(
                {
                    "n_obs": result.n_obs,
                    "pinball_loss": result.pinball_loss,
                    "crps": result.crps,
                    "coverage_80": result.coverage_80,
                    "winkler_80": result.winkler_80,
                    "pit_mean": result.pit_mean,
                    **({"dm_stat": result.dm_stat} if result.dm_stat is not None else {}),
                    **({"dm_pvalue": result.dm_pvalue} if result.dm_pvalue is not None else {}),
                    **(
                        {"dm_stat_vs_seasonal_naive": result.dm_stat_vs_seasonal_naive}
                        if result.dm_stat_vs_seasonal_naive is not None
                        else {}
                    ),
                    **(
                        {"dm_pvalue_vs_seasonal_naive": result.dm_pvalue_vs_seasonal_naive}
                        if result.dm_pvalue_vs_seasonal_naive is not None
                        else {}
                    ),
                }
            )

    if results:
        best = select_best_rung(results)
        if forecast_records[best.rung]:
            conn = get_connection(config.duckdb_path)
            try:
                write_table(conn, "forecast_day_ahead", forecast_records[best.rung])
            finally:
                conn.close()

    return results


def select_best_rung(
    results: list[RungResult], *, alpha: float = DM_SIGNIFICANCE_ALPHA
) -> RungResult:
    """Pick the lowest-pinball-loss rung among those not significantly worse than naive.

    A rung is eligible if it's the naive reference itself (dm_pvalue is None) or if its
    DM test is significant AND directionally better than naive (dm_stat < 0) — a plain
    p < alpha check would also admit rungs that are significantly worse.
    """
    eligible = [
        r
        for r in results
        if r.dm_pvalue is None or (r.dm_stat is not None and r.dm_pvalue < alpha and r.dm_stat < 0)
    ]
    return min(eligible, key=lambda r: r.pinball_loss)


__all__ = ["RungResult", "run_benchmark_ladder", "select_best_rung"]
