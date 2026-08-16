import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nordic_power_risk.bridge import run as bridge_module
from nordic_power_risk.bridge.run import run_thesis_bridge
from nordic_power_risk.config import PipelineConfig, Window
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table

TRANSITION = date(2025, 10, 1)


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        zone="SE3",
        windows={
            "primary": Window(start=date(2024, 6, 1), end=TRANSITION),
            "secondary": Window(start=TRANSITION, end=date(2026, 3, 31)),
        },
        duckdb_path=tmp_path / "p16.duckdb",
        manifest_path=tmp_path / "manifest.json",
    )


def _seed(config: PipelineConfig) -> None:
    rng = np.random.default_rng(7)
    pre_idx = pd.date_range(
        config.windows["primary"].start, TRANSITION, freq="15min", inclusive="left"
    )
    post_idx = pd.date_range(
        TRANSITION,
        config.windows["secondary"].end + timedelta(days=1),
        freq="15min",
        inclusive="left",
    )

    def _imbalance(ts: pd.Timestamp) -> float:
        return float(15.0 + 3.0 * np.sin(2 * np.pi * ts.hour / 24.0) + rng.normal(0, 0.3))

    imbalance_rows = [
        {
            "event_time": ts,
            "issue_time": ts + timedelta(minutes=45),
            "imbalance_price_eur_mwh": _imbalance(ts),
            "price_type": "final",
        }
        for ts in pre_idx.append(post_idx)
    ]

    hours = pd.date_range(
        config.windows["primary"].start,
        config.windows["primary"].end,
        freq="h",
        inclusive="left",
    )
    da_prices = 20.0 + 5.0 * np.sin(2 * np.pi * np.arange(len(hours)) / 24.0) + rng.normal(
        0, 1.0, len(hours)
    )
    day_ahead_rows = [
        {
            "event_time": ts,
            "price_eur_mwh": float(da_prices[i]),
            "price_lag_168h": float(da_prices[i - 168]) if i >= 168 else float("nan"),
        }
        for i, ts in enumerate(hours)
    ]

    conn = get_connection(config.duckdb_path)
    try:
        write_table(conn, "fact_imbalance_price", imbalance_rows)
        write_table(conn, "feature_day_ahead", day_ahead_rows)
    finally:
        conn.close()


def _stub_lgbm(train, test, value_column, *, quantile_grid=(0.1, 0.5, 0.9)):  # type: ignore[no-untyped-def]
    """Deterministic stand-in for LightGBM: constant quantiles, NaN where lag is missing.

    LightGBM itself is exercised in tests/models/test_secondary_run.py; the bridge
    test only needs a fast, aligned forecast to verify the orchestration.
    """
    lag = test[f"{value_column}_lag_168h"]
    valid = lag.notna()
    out = {}
    for q in quantile_grid:
        series = pd.Series(15.0 + (q - 0.5), index=test.index, dtype=float)
        series[~valid] = float("nan")
        out[q] = series
    return out


def test_run_thesis_bridge_splits_aggregates_and_applies_mde_rule(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    config = _make_config(tmp_path)
    _seed(config)
    monkeypatch.setattr(bridge_module, "lgbm_quantile_forecast", _stub_lgbm)

    result = run_thesis_bridge(config)

    # granularity: pre is hourly-aggregated, post is native 15-minute
    assert result.pre.granularity == "hourly"
    assert result.post.granularity == "15min"

    # n_pre is the 15-min pre series collapsed 4:1 to hourly; n_post is raw 15-min
    expected_pre_hours = len(
        pd.date_range(config.windows["primary"].start, TRANSITION, freq="h", inclusive="left")
    )
    expected_post_rows = len(
        pd.date_range(
            TRANSITION,
            config.windows["secondary"].end + timedelta(days=1),
            freq="15min",
            inclusive="left",
        )
    )
    assert result.gate.n_pre == expected_pre_hours
    assert result.gate.n_post == expected_post_rows

    # MDE gate is positive and self-consistent with the observed effect
    assert result.gate.mde > 0
    assert result.effect_pinball == pytest.approx(
        result.post.pinball_loss_lgbm - result.pre.pinball_loss_lgbm
    )
    assert result.effect_detected == (abs(result.effect_pinball) >= result.gate.mde)
    assert result.framing == result.gate.framing

    # both difficulty rungs computed and finite
    assert np.isfinite(result.pre.pinball_loss_seasonal_naive)
    assert np.isfinite(result.post.crps_lgbm)

    # artifacts are written and parse
    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    assert summary["framing"] == result.framing
    assert summary["gate"]["n_pre"] == expected_pre_hours
    chapter = Path(result.chapter_path).read_text(encoding="utf-8")
    assert "Thesis bridge" in chapter
    verdict = (
        "inferential exhibit"
        if result.framing == "inferential_exhibit"
        else "methodology demonstration"
    )
    assert verdict in chapter


def test_run_thesis_bridge_requires_imbalance_data(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _make_config(tmp_path)
    with pytest.raises(RuntimeError, match="fact_imbalance_price"):
        run_thesis_bridge(config)
