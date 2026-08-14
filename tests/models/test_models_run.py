from datetime import date

import numpy as np
import pandas as pd

from nordic_power_risk.config import PipelineConfig, Window
from nordic_power_risk.facts.run import build_all_facts
from nordic_power_risk.features.run import build_all_features
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.models.run import run_benchmark_ladder


def _make_config(tmp_path, start: date, end: date) -> PipelineConfig:  # type: ignore[no-untyped-def]
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=start, end=end)},
        duckdb_path=tmp_path / "nordic_power_risk.duckdb",
        manifest_path=tmp_path / "manifest.json",
    )


def _seed(config: PipelineConfig, start: date, end: date) -> None:
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    iso_rows = [t.isoformat() for t in hours]
    rng = np.random.default_rng(42)
    seasonal = 10.0 * np.sin(np.arange(len(hours)) * 2 * np.pi / 24)
    prices = 30.0 + seasonal + rng.normal(0, 1, len(hours))
    conn = get_connection(config.duckdb_path)
    try:
        write_table(
            conn,
            "raw_entsoe_day_ahead_price",
            [
                {"timestamp": t, "price_eur_mwh": float(p)}
                for t, p in zip(iso_rows, prices, strict=True)
            ],
        )
        write_table(
            conn,
            "raw_esett_imbalance_price",
            [{"timestamp": t, "imbalance_price_eur_mwh": 15.0} for t in iso_rows],
        )
        write_table(
            conn, "raw_svk_day_ahead_price", [{"timestamp": t, "value": 10.0} for t in iso_rows]
        )
        write_table(
            conn, "raw_svk_fcr_capacity", [{"timestamp": t, "value": 5.0} for t in iso_rows]
        )
        write_table(
            conn, "raw_svk_afrr_mfrr_capacity", [{"timestamp": t, "value": 3.0} for t in iso_rows]
        )
        write_table(
            conn,
            "raw_smhi_observations",
            [{"timestamp": int(t.timestamp() * 1000), "value": -2.5} for t in hours],
        )
    finally:
        conn.close()


def test_benchmark_ladder_reports_naive_and_seasonal_naive_with_dm_test(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    start, end = date(2024, 1, 1), date(2025, 6, 1)
    config = _make_config(tmp_path, start, end)
    _seed(config, start, end)
    build_all_facts(config)
    build_all_features(config)

    results = run_benchmark_ladder(config)
    by_rung = {r.rung: r for r in results}

    assert "naive" in by_rung
    assert "seasonal_naive" in by_rung
    assert by_rung["naive"].n_obs > 0
    assert by_rung["naive"].dm_stat is None  # naive is the reference rung
    assert by_rung["seasonal_naive"].dm_stat is not None
    assert by_rung["seasonal_naive"].dm_pvalue is not None
    for result in results:
        assert result.pinball_loss >= 0
        assert result.crps >= 0
        assert 0.0 <= result.coverage_80 <= 1.0
        assert 0.0 <= result.pit_mean <= 1.0

