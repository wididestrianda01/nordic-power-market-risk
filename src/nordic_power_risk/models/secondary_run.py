"""Secondary-target benchmark: seasonal-naive vs LightGBM quantile (Phase 2 ticket 04).

No promotion gate here (unlike the day-ahead ladder) -- ticket 04 scopes this
to reporting pinball loss and diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd

from nordic_power_risk.config import PipelineConfig
from nordic_power_risk.features.split import rolling_origin_folds
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.models.baselines import (
    quantile_forecast,
    residual_quantiles,
    seasonal_naive_forecast_for,
)
from nordic_power_risk.models.lgbm import SECONDARY_QUANTILE_GRID, lgbm_quantile_forecast
from nordic_power_risk.models.metrics import (
    crps_approx,
    diebold_mariano_test,
    interval_coverage,
    mean_pinball_over_grid,
    mean_pinball_per_row,
    pit_values,
    winkler_score,
)

SECONDARY_TARGETS: tuple[tuple[str, str], ...] = (
    ("feature_fcr_d_up", "price"),
    ("feature_fcr_d_down", "price"),
    ("feature_fcr_n", "price"),
    ("feature_imbalance", "imbalance_price_eur_mwh"),
)

COVERAGE_LOWER_Q = 0.1
COVERAGE_UPPER_Q = 0.9
COVERAGE_ALPHA = 0.2
MLFLOW_EXPERIMENT_SUFFIX = "-secondary-targets"
IMBALANCE_FORECAST_COLUMNS = {
    "issue_time": "TIMESTAMP",
    "event_time": "TIMESTAMP",
    "q0_1": "DOUBLE",
    "q0_5": "DOUBLE",
    "q0_9": "DOUBLE",
}
RESERVE_TARGETS = {
    "feature_fcr_d_up": ("FCR_D", "up"),
    "feature_fcr_d_down": ("FCR_D", "down"),
    "feature_fcr_n": ("FCR_N", "symmetric"),
}
RESERVE_FORECAST_COLUMNS = {
    "product": "VARCHAR",
    "direction": "VARCHAR",
    "issue_time": "TIMESTAMP",
    "delivery_time": "TIMESTAMP",
    "q0_1": "DOUBLE",
    "q0_5": "DOUBLE",
    "q0_9": "DOUBLE",
    "forecast_source": "VARCHAR",
}


@dataclass(frozen=True)
class SecondaryRungResult:
    target: str
    rung: str
    n_obs: int
    pinball_loss: float
    crps: float
    coverage_80: float
    winkler_80: float
    pit_mean: float
    dm_stat: float | None
    dm_pvalue: float | None


def _forecast_seasonal_naive(
    train: pd.DataFrame, test: pd.DataFrame, value_column: str
) -> dict[float, pd.Series]:
    train_point = seasonal_naive_forecast_for(train, value_column)
    test_point = seasonal_naive_forecast_for(test, value_column)
    resid_q = residual_quantiles(
        train[value_column], train_point, quantile_grid=SECONDARY_QUANTILE_GRID
    )
    return quantile_forecast(test_point, resid_q)


def _optimizer_forecast_rows(
    test: pd.DataFrame, quantile_preds: dict[float, pd.Series]
) -> list[dict[str, object]]:
    valid = pd.concat(quantile_preds.values(), axis=1).notna().all(axis=1)
    return [
        {
            "issue_time": test.at[index, "issue_time"],
            "event_time": test.at[index, "event_time"],
            "q0_1": float(quantile_preds[0.1].at[index]),
            "q0_5": float(quantile_preds[0.5].at[index]),
            "q0_9": float(quantile_preds[0.9].at[index]),
        }
        for index in test.index[valid]
    ]


def _reserve_forecast_rows(
    table: str, test: pd.DataFrame, quantile_preds: dict[float, pd.Series]
) -> list[dict[str, object]]:
    product, direction = RESERVE_TARGETS[table]
    rows = _optimizer_forecast_rows(test, quantile_preds)
    return [
        {
            "product": product,
            "direction": direction,
            "issue_time": row["issue_time"],
            "delivery_time": row["event_time"],
            "q0_1": row["q0_1"],
            "q0_5": row["q0_5"],
            "q0_9": row["q0_9"],
            "forecast_source": "lgbm",
        }
        for row in rows
    ]


def _unique_imbalance_forecasts(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return list(
        {(row["issue_time"], row["event_time"]): row for row in rows}.values()
    )


def _unique_reserve_forecasts(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return list(
        {
            (row["product"], row["direction"], row["issue_time"], row["delivery_time"]): row
            for row in rows
        }.values()
    )


def _write_optimizer_forecasts(
    conn: object,
    imbalance_rows: list[dict[str, object]],
    reserve_rows: list[dict[str, object]],
) -> None:
    write_table(
        conn, "forecast_imbalance", _unique_imbalance_forecasts(imbalance_rows),
        columns=IMBALANCE_FORECAST_COLUMNS,
    )
    write_table(
        conn, "forecast_reserve", _unique_reserve_forecasts(reserve_rows),
        columns=RESERVE_FORECAST_COLUMNS,
    )


def _persist_optimizer_forecasts(
    config: PipelineConfig,
    imbalance_rows: list[dict[str, object]],
    reserve_rows: list[dict[str, object]],
) -> None:
    conn = get_connection(config.duckdb_path)
    conn.execute("BEGIN TRANSACTION")
    try:
        _write_optimizer_forecasts(conn, imbalance_rows, reserve_rows)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
    finally:
        conn.close()


def run_secondary_benchmark(config: PipelineConfig) -> list[SecondaryRungResult]:
    conn = get_connection(config.duckdb_path)
    try:
        tables = {
            table: conn.execute(f"SELECT * FROM {table}").fetchdf()
            for table, _ in SECONDARY_TARGETS
        }
    finally:
        conn.close()

    primary = config.windows["primary"]
    folds = rolling_origin_folds(primary.start, primary.end)

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment + MLFLOW_EXPERIMENT_SUFFIX)

    results: list[SecondaryRungResult] = []
    optimizer_forecasts: list[dict[str, object]] = []
    reserve_forecasts: list[dict[str, object]] = []
    for table, value_column in SECONDARY_TARGETS:
        df = tables[table]
        event_date = df["event_time"].dt.date

        row_losses: dict[str, list[np.ndarray]] = {"seasonal_naive": [], "lgbm": []}
        agg: dict[str, dict[str, float | int]] = {
            rung: {"n_obs": 0, "pinball_sum": 0.0, "crps_sum": 0.0, "coverage_sum": 0.0,
                   "winkler_sum": 0.0, "pit_sum": 0.0}
            for rung in ("seasonal_naive", "lgbm")
        }

        for fold in folds:
            train = df[(event_date >= fold.train_start) & (event_date < fold.train_end)]
            test = df[(event_date >= fold.test_start) & (event_date < fold.test_end)]
            if train.empty or test.empty:
                continue

            forecasts = {
                "seasonal_naive": _forecast_seasonal_naive(train, test, value_column),
                "lgbm": lgbm_quantile_forecast(train, test, value_column),
            }
            if table == "feature_imbalance":
                optimizer_forecasts.extend(
                    _optimizer_forecast_rows(test, forecasts["lgbm"])
                )
            if table in RESERVE_TARGETS:
                reserve_forecasts.extend(
                    _reserve_forecast_rows(table, test, forecasts["lgbm"])
                )

            for rung, quantile_preds in forecasts.items():
                median = quantile_preds[0.5]
                valid = median.notna() & test[value_column].notna()
                if not valid.any():
                    continue

                y_true = test.loc[valid, value_column].to_numpy()
                preds = {q: series.loc[valid].to_numpy() for q, series in quantile_preds.items()}

                row_losses[rung].append(mean_pinball_per_row(y_true, preds))
                bucket = agg[rung]
                n = int(valid.sum())
                bucket["n_obs"] = int(bucket["n_obs"]) + n
                bucket["pinball_sum"] = float(bucket["pinball_sum"]) + mean_pinball_over_grid(
                    y_true, preds
                ) * n
                bucket["crps_sum"] = float(bucket["crps_sum"]) + crps_approx(y_true, preds) * n
                bucket["coverage_sum"] = float(bucket["coverage_sum"]) + interval_coverage(
                    y_true, preds[COVERAGE_LOWER_Q], preds[COVERAGE_UPPER_Q]
                ) * n
                bucket["winkler_sum"] = float(bucket["winkler_sum"]) + winkler_score(
                    y_true, preds[COVERAGE_LOWER_Q], preds[COVERAGE_UPPER_Q], COVERAGE_ALPHA
                ) * n
                bucket["pit_sum"] = float(bucket["pit_sum"]) + float(
                    np.mean(pit_values(y_true, preds))
                ) * n

        baseline_losses = (
            np.concatenate(row_losses["seasonal_naive"])
            if row_losses["seasonal_naive"]
            else np.array([])
        )
        lgbm_losses = (
            np.concatenate(row_losses["lgbm"]) if row_losses["lgbm"] else np.array([])
        )
        dm_stat, dm_pvalue = (
            diebold_mariano_test(lgbm_losses, baseline_losses)
            if len(lgbm_losses) > 0 and len(baseline_losses) > 0
            else (None, None)
        )

        for rung in ("seasonal_naive", "lgbm"):
            bucket = agg[rung]
            n_obs = int(bucket["n_obs"])
            if n_obs == 0:
                continue
            result = SecondaryRungResult(
                target=table,
                rung=rung,
                n_obs=n_obs,
                pinball_loss=float(bucket["pinball_sum"]) / n_obs,
                crps=float(bucket["crps_sum"]) / n_obs,
                coverage_80=float(bucket["coverage_sum"]) / n_obs,
                winkler_80=float(bucket["winkler_sum"]) / n_obs,
                pit_mean=float(bucket["pit_sum"]) / n_obs,
                dm_stat=dm_stat if rung == "lgbm" else None,
                dm_pvalue=dm_pvalue if rung == "lgbm" else None,
            )
            results.append(result)

            with mlflow.start_run(run_name=f"{table}-{rung}"):
                metrics = {
                    "n_obs": result.n_obs,
                    "pinball_loss": result.pinball_loss,
                    "crps": result.crps,
                    "coverage_80": result.coverage_80,
                    "winkler_80": result.winkler_80,
                    "pit_mean": result.pit_mean,
                }
                if result.dm_stat is not None:
                    metrics["dm_stat"] = result.dm_stat
                if result.dm_pvalue is not None:
                    metrics["dm_pvalue"] = result.dm_pvalue
                mlflow.log_metrics(metrics)

    _persist_optimizer_forecasts(config, optimizer_forecasts, reserve_forecasts)
    return results


__all__ = ["SECONDARY_TARGETS", "SecondaryRungResult", "run_secondary_benchmark"]
