"""Thesis-bridge analysis (Phase 6): minimum-detectable-effect gate + interrupted
time series comparing imbalance-price forecast difficulty across the 15-minute MTU
transition (pre: hourly-aggregated, post: native 15-minute).

Identification is descriptive/quasi-experimental, not a strong causal claim: the
transition was EU-wide and simultaneous, so no untransitioned control zone exists
and concurrent confounders in the same window cannot be ruled out.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from nordic_power_risk.bridge.mde import MdeGate, compute_mde_gate
from nordic_power_risk.config import PipelineConfig
from nordic_power_risk.facts.rules import IMBALANCE_FINAL_LAG, imbalance_forecast_issue_time
from nordic_power_risk.features.build import build_lag_calendar_features
from nordic_power_risk.features.split import rolling_origin_folds
from nordic_power_risk.ingest.duckdb_io import fetch_scalar, get_connection
from nordic_power_risk.models.baselines import (
    quantile_forecast,
    residual_quantiles,
    seasonal_naive_forecast,
    seasonal_naive_forecast_for,
)
from nordic_power_risk.models.lgbm import SECONDARY_QUANTILE_GRID, lgbm_quantile_forecast
from nordic_power_risk.models.metrics import (
    crps_approx,
    mean_pinball_over_grid,
    mean_pinball_per_row,
)

IMBALANCE_VALUE_COLUMN = "imbalance_price_eur_mwh"


@dataclass(frozen=True)
class PeriodDifficulty:
    period: str
    granularity: str
    n_obs: int
    pinball_loss_lgbm: float
    crps_lgbm: float
    pinball_loss_seasonal_naive: float


@dataclass(frozen=True)
class BridgeResult:
    transition_date: date
    sigma: float
    gate: MdeGate
    pre: PeriodDifficulty
    post: PeriodDifficulty
    effect_pinball: float
    effect_detected: bool
    framing: str
    summary_path: str
    chapter_path: str


def _read_final_imbalance_prices(config: PipelineConfig) -> pd.DataFrame:
    """Final imbalance settlement prices at native granularity (event_time, value)."""
    conn = get_connection(config.duckdb_path)
    try:
        exists = fetch_scalar(
            conn,
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'fact_imbalance_price'",
        )
        if not exists:
            raise RuntimeError(
                "fact_imbalance_price table missing; run `nordic-risk ingest` "
                "(eSett imbalance) through `facts` first"
            )
        df = conn.execute(
            f"SELECT event_time, {IMBALANCE_VALUE_COLUMN} "
            f"FROM fact_imbalance_price WHERE price_type = 'final'"
        ).fetchdf()
    finally:
        conn.close()
    if df.empty:
        raise RuntimeError(
            "fact_imbalance_price has no final rows; run `nordic-risk ingest` "
            "(eSett imbalance) through `facts` first"
        )
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df.sort_values("event_time")[["event_time", IMBALANCE_VALUE_COLUMN]]


def _aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """15-minute (or finer) rows -> hourly mean imbalance price, NaN hours dropped."""
    hourly = (
        df.set_index("event_time")
        .resample("h")
        .mean(numeric_only=True)
        .dropna()
        .reset_index()
    )
    return hourly[["event_time", IMBALANCE_VALUE_COLUMN]]


def _seasonal_naive_quantiles(
    train: pd.DataFrame, test: pd.DataFrame, value_column: str
) -> dict[float, pd.Series]:
    train_point = seasonal_naive_forecast_for(train, value_column)
    test_point = seasonal_naive_forecast_for(test, value_column)
    resid_q = residual_quantiles(
        train[value_column], train_point, quantile_grid=SECONDARY_QUANTILE_GRID
    )
    return quantile_forecast(test_point, resid_q)


def _period_difficulty(
    features: pd.DataFrame, value_column: str, start: date, end: date, period: str
) -> PeriodDifficulty:
    """Secondary-forecast pinball/CRPS over rolling-origin folds for one period."""
    folds = rolling_origin_folds(start, end)
    event_date = features["event_time"].dt.date
    agg = {
        "lgbm": {"n": 0, "pinball_sum": 0.0, "crps_sum": 0.0},
        "seasonal_naive": {"n": 0, "pinball_sum": 0.0},
    }
    for fold in folds:
        train = features[(event_date >= fold.train_start) & (event_date < fold.train_end)]
        test = features[(event_date >= fold.test_start) & (event_date < fold.test_end)]
        if train.empty or test.empty:
            continue
        forecasts = {
            "seasonal_naive": _seasonal_naive_quantiles(train, test, value_column),
            "lgbm": lgbm_quantile_forecast(train, test, value_column),
        }
        common_valid = test[value_column].notna()
        for qpred in forecasts.values():
            common_valid &= qpred[0.5].notna()
        if not common_valid.any():
            continue
        y_true = test.loc[common_valid, value_column].to_numpy()
        for rung, qpred in forecasts.items():
            preds = {q: s.loc[common_valid].to_numpy() for q, s in qpred.items()}
            n = int(common_valid.sum())
            bucket = agg[rung]
            bucket["n"] += n
            bucket["pinball_sum"] += mean_pinball_over_grid(y_true, preds) * n
            if rung == "lgbm":
                bucket["crps_sum"] += crps_approx(y_true, preds) * n
    n_lgbm = int(agg["lgbm"]["n"])
    n_sn = int(agg["seasonal_naive"]["n"])
    if n_lgbm == 0 or n_sn == 0:
        raise RuntimeError(f"no evaluable folds for {period} period; insufficient imbalance data")
    granularity = "hourly" if period == "pre" else "15min"
    return PeriodDifficulty(
        period=period,
        granularity=granularity,
        n_obs=n_lgbm,
        pinball_loss_lgbm=agg["lgbm"]["pinball_sum"] / n_lgbm,
        crps_lgbm=agg["lgbm"]["crps_sum"] / n_lgbm,
        pinball_loss_seasonal_naive=agg["seasonal_naive"]["pinball_sum"] / n_sn,
    )


def _day_ahead_loss_sigma(config: PipelineConfig) -> float:
    """Per-observation pinball-loss std dev from the day-ahead spine's 7.5-year backtest.

    Uses the seasonal-naive rung of the day-ahead ladder with the secondary 3-point
    quantile grid so the loss scale matches the imbalance difficulty measure. Seasonal
    naive is the higher-variance rung, so this is a conservative (larger) MDE.
    """
    conn = get_connection(config.duckdb_path)
    try:
        exists = fetch_scalar(
            conn,
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'feature_day_ahead'",
        )
        if not exists:
            raise RuntimeError(
                "feature_day_ahead table missing; run `nordic-risk ingest` "
                "through `features` first"
            )
        df = conn.execute(
            "SELECT event_time, price_eur_mwh, price_lag_168h FROM feature_day_ahead"
        ).fetchdf()
    finally:
        conn.close()
    if df.empty:
        raise RuntimeError(
            "feature_day_ahead is empty; run `nordic-risk ingest` through `features` first"
        )
    df["event_time"] = pd.to_datetime(df["event_time"])
    primary = config.windows["primary"]
    folds = rolling_origin_folds(primary.start, primary.end)
    event_date = df["event_time"].dt.date
    losses: list[np.ndarray] = []
    for fold in folds:
        train = df[(event_date >= fold.train_start) & (event_date < fold.train_end)]
        test = df[(event_date >= fold.test_start) & (event_date < fold.test_end)]
        if train.empty or test.empty:
            continue
        train_point = seasonal_naive_forecast(train)
        test_point = seasonal_naive_forecast(test)
        resid_q = residual_quantiles(
            train["price_eur_mwh"], train_point, quantile_grid=SECONDARY_QUANTILE_GRID
        )
        qpred = quantile_forecast(test_point, resid_q)
        valid = test["price_eur_mwh"].notna() & test_point.notna()
        if not valid.any():
            continue
        y_true = test.loc[valid, "price_eur_mwh"].to_numpy()
        preds = {q: s.loc[valid].to_numpy() for q, s in qpred.items()}
        losses.append(mean_pinball_per_row(y_true, preds))
    if not losses:
        raise RuntimeError("no evaluable day-ahead folds for the loss-variance estimate")
    return float(np.std(np.concatenate(losses), ddof=1))


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag/calendar features for the imbalance target, issue-time gated at T-60min.

    The fact table's final-price issue_time (event_time + IMBALANCE_FINAL_LAG) is
    reconstructed here so the lag look-ahead check runs against the true
    publication time; build_lag_calendar_features then overrides the forecast
    issue_time to the T-60min decision cutoff.
    """
    frame = df.copy()
    frame["issue_time"] = frame["event_time"] + IMBALANCE_FINAL_LAG
    return build_lag_calendar_features(
        frame, IMBALANCE_VALUE_COLUMN, issue_time_fn=imbalance_forecast_issue_time
    )


def _write_chapter(result: BridgeResult, path: str) -> None:
    pre, post, gate = result.pre, result.post, result.gate
    verdict = (
        "inferential exhibit"
        if result.framing == "inferential_exhibit"
        else "methodology demonstration"
    )
    effect_word = "detectable" if result.effect_detected else "not detectable"
    text = (
        "# Thesis bridge — 15-minute MTU transition and imbalance-price forecast difficulty\n\n"
        "## Identification\n\n"
        "Interrupted time series on SE3's own imbalance-price series, pre vs post the "
        f"15-minute MTU transition ({result.transition_date.isoformat()}). The transition was "
        "EU-wide and simultaneous, so no untransitioned control zone exists for a "
        "difference-in-differences design. This chapter is descriptive/quasi-experimental, "
        "not a strong causal claim: concurrent confounders in the same window cannot be "
        "ruled out.\n\n"
        "## Measured variable\n\n"
        "Forecast difficulty of the secondary imbalance model (quantile LightGBM, 3-point "
        "grid {0.1, 0.5, 0.9}): mean pinball loss and CRPS. Pre-period difficulty is "
        "measured on hourly-aggregated imbalance prices; post-period difficulty on native "
        "15-minute prices.\n\n"
        "## Pre-declared minimum-detectable-effect gate\n\n"
        f"- sigma (day-ahead 7.5-year backtest loss std dev): {gate.sigma:.4f}\n"
        f"- n_pre: {gate.n_pre} | n_post: {gate.n_post}\n"
        f"- alpha: {gate.alpha} | power: {gate.power}\n"
        f"- MDE: {gate.mde:.4f} (absolute pinball-loss shift)\n"
        f"- baseline pre-period pinball loss: {gate.baseline_loss:.4f}\n"
        f"- relative MDE: {gate.relative_mde:.3f} "
        f"(relevance threshold {gate.relevance_fraction:.2f})\n"
        f"- framing (decided before seeing the result): **{verdict}**\n\n"
        "## Result\n\n"
        f"- pre ({pre.granularity}, n={pre.n_obs}): pinball {pre.pinball_loss_lgbm:.4f}, "
        f"CRPS {pre.crps_lgbm:.4f}\n"
        f"- post ({post.granularity}, n={post.n_obs}): pinball {post.pinball_loss_lgbm:.4f}, "
        f"CRPS {post.crps_lgbm:.4f}\n"
        f"- effect (post - pre pinball loss): {result.effect_pinball:+.4f} — "
        f"{effect_word} at the MDE\n\n"
        "## Limitations\n\n"
        "- Short post period (~10.5 months) relative to the 7.5-year pre spine.\n"
        "- No counterfactual zone (EU-wide simultaneous transition).\n"
        "- Concurrent confounders (market design, generation mix, price levels) cannot "
        "be ruled out.\n"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def run_thesis_bridge(config: PipelineConfig) -> BridgeResult:
    """Compute the MDE gate, run the pre/post ITS comparison, and write the chapter."""
    prices = _read_final_imbalance_prices(config)
    transition = config.windows["secondary"].start
    transition_ts = pd.Timestamp(transition)

    pre_prices = _aggregate_hourly(prices[prices["event_time"] < transition_ts])
    post_prices = prices[prices["event_time"] >= transition_ts]
    if pre_prices.empty:
        raise RuntimeError(f"no pre-transition imbalance prices before {transition.isoformat()}")
    if post_prices.empty:
        raise RuntimeError(f"no post-transition imbalance prices from {transition.isoformat()}")

    pre_features = _build_features(pre_prices)
    post_features = _build_features(post_prices)

    sigma = _day_ahead_loss_sigma(config)

    # MDE gate is computed *before* the post-period result is seen, using only the
    # pre-period baseline difficulty and the day-ahead loss variance.
    pre = _period_difficulty(
        pre_features,
        IMBALANCE_VALUE_COLUMN,
        config.windows["primary"].start,
        transition,
        "pre",
    )
    gate = compute_mde_gate(
        sigma=sigma,
        n_pre=len(pre_prices),
        n_post=len(post_prices),
        baseline_loss=pre.pinball_loss_lgbm,
    )

    post = _period_difficulty(
        post_features,
        IMBALANCE_VALUE_COLUMN,
        transition,
        config.windows["secondary"].end + timedelta(days=1),
        "post",
    )

    effect = post.pinball_loss_lgbm - pre.pinball_loss_lgbm
    effect_detected = abs(effect) >= gate.mde

    summary_path = config.duckdb_path.parent / "bridge_analysis.json"
    chapter_path = config.duckdb_path.parent / "bridge_chapter.md"
    result = BridgeResult(
        transition_date=transition,
        sigma=sigma,
        gate=gate,
        pre=pre,
        post=post,
        effect_pinball=effect,
        effect_detected=effect_detected,
        framing=gate.framing,
        summary_path=str(summary_path),
        chapter_path=str(chapter_path),
    )

    summary_path.write_text(
        json.dumps(asdict(result), indent=2, default=str), encoding="utf-8"
    )
    _write_chapter(result, str(chapter_path))
    return result


__all__ = [
    "BridgeResult",
    "PeriodDifficulty",
    "run_thesis_bridge",
]
