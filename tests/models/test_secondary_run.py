from datetime import date

import numpy as np
import pandas as pd

from nordic_power_risk.config import PipelineConfig, Window
from nordic_power_risk.facts.run import build_all_facts
from nordic_power_risk.features.run import build_secondary_features
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.models.secondary_run import SECONDARY_TARGETS, run_secondary_benchmark


def _make_config(tmp_path, start: date, end: date) -> PipelineConfig:  # type: ignore[no-untyped-def]
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=start, end=end)},
        duckdb_path=tmp_path / "nordic_power_risk.duckdb",
        manifest_path=tmp_path / "manifest.json",
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )


def _fcr_rows(iso_rows: list[str], zone: str) -> list[dict]:  # type: ignore[type-arg]
    rows = []
    products = (("FCRD", "up", 5.0), ("FCRD", "down", 4.0), ("FCRN", "symmetric", 6.0))
    for product, direction, price in products:
        rows.extend(
            {
                "start_time_utc": t,
                "price": price,
                "bidding_zone": zone,
                "reserve_product": product,
                "reserve_direction": direction,
            }
            for t in iso_rows
        )
    return rows


def _seed(config: PipelineConfig, start: date, end: date) -> None:
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    iso_rows = [t.isoformat() for t in hours]
    rng = np.random.default_rng(42)
    seasonal = 2.0 * np.sin(np.arange(len(hours)) * 2 * np.pi / 24)
    imbalance_prices = 15.0 + seasonal + rng.normal(0, 0.5, len(hours))
    conn = get_connection(config.duckdb_path)
    try:
        write_table(
            conn,
            "raw_entsoe_day_ahead_price",
            [{"timestamp": t, "price_eur_mwh": 10.0} for t in iso_rows],
        )
        write_table(
            conn,
            "raw_esett_imbalance_price",
            [
                {"timestamp": t, "imbalance_price_eur_mwh": float(p)}
                for t, p in zip(iso_rows, imbalance_prices, strict=True)
            ],
        )
        write_table(
            conn, "raw_svk_day_ahead_price", [{"timestamp": t, "value": 10.0} for t in iso_rows]
        )
        write_table(conn, "raw_svk_fcr_capacity", _fcr_rows(iso_rows, config.zone))
        write_table(
            conn,
            "raw_svk_afrr_mfrr_capacity",
            [{"start_time_utc": t, "price": 3.0} for t in iso_rows],
        )
        write_table(
            conn,
            "raw_smhi_observations",
            [{"timestamp": int(t.timestamp() * 1000), "value": -2.5} for t in hours],
        )
    finally:
        conn.close()


def test_run_secondary_benchmark_reports_seasonal_naive_and_lgbm_with_dm_test(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    start, end = date(2024, 1, 1), date(2025, 6, 1)
    config = _make_config(tmp_path, start, end)
    _seed(config, start, end)
    build_all_facts(config)
    build_secondary_features(config)

    results = run_secondary_benchmark(config)
    assert results

    tables = {table for table, _ in SECONDARY_TARGETS}
    by_target_rung = {(r.target, r.rung): r for r in results}

    for table in tables:
        assert (table, "seasonal_naive") in by_target_rung
        assert (table, "lgbm") in by_target_rung

        naive = by_target_rung[(table, "seasonal_naive")]
        lgbm = by_target_rung[(table, "lgbm")]

        assert naive.dm_stat is None
        assert naive.dm_pvalue is None
        assert lgbm.dm_stat is not None
        assert lgbm.dm_pvalue is not None

        for result in (naive, lgbm):
            assert result.n_obs > 0
            assert result.pinball_loss >= 0
            assert result.crps >= 0
            assert 0.0 <= result.coverage_80 <= 1.0
            assert 0.0 <= result.pit_mean <= 1.0

    conn = get_connection(config.duckdb_path)
    try:
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
    finally:
        conn.close()
    assert "forecast_day_ahead" not in table_names
    assert not any(name.startswith("forecast_") for name in table_names)
