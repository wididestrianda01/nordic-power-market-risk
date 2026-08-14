"""Run naive/seasonal-naive benchmark ladder over T08 rolling-origin folds (Phase 2 ticket 01)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nordic_power_risk.config import PipelineConfig
from nordic_power_risk.features.split import rolling_origin_folds
from nordic_power_risk.ingest.duckdb_io import get_connection
from nordic_power_risk.models.baselines import (
    naive_forecast,
    quantile_forecast,
    residual_quantiles,
    seasonal_naive_forecast,
)
from nordic_power_risk.models.metrics import (
    diebold_mariano_test,
    interval_coverage,
    mean_pinball_per_row,
    pit_values,
    winkler_score,
)

RUNGS = {"naive": naive_forecast, "seasonal_naive": seasonal_naive_forecast}
COVERAGE_LOWER_Q = 0.1
COVERAGE_UPPER_Q = 0.9
COVERAGE_ALPHA = 0.2  # 1 - (upper - lower)


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

    for fold in folds:
        train = df[(event_date >= fold.train_start) & (event_date < fold.train_end)]
        test = df[(event_date >= fold.test_start) & (event_date < fold.test_end)]
        for name, point_fn in RUNGS.items():
            train_point = point_fn(train)
            test_point = point_fn(test)
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

    concatenated = {name: np.concatenate(row_losses[name]) for name in RUNGS if row_losses[name]}
    naive_loss_series = concatenated.get("naive")

    results = []
    for name in RUNGS:
        if name not in concatenated:
            continue
        loss_series = concatenated[name]
        dm_stat: float | None = None
        dm_pvalue: float | None = None
        if name != "naive" and naive_loss_series is not None:
            dm_stat, dm_pvalue = diebold_mariano_test(loss_series, naive_loss_series)
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
            )
        )
    return results


__all__ = ["RungResult", "run_benchmark_ladder"]
