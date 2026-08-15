from contextlib import nullcontext
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nordic_power_risk.config import PipelineConfig, Window
from nordic_power_risk.facts.run import build_all_facts
from nordic_power_risk.facts.rules import imbalance_forecast_issue_time
from nordic_power_risk.features.run import build_secondary_features
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.models import secondary_run as secondary_module
from nordic_power_risk.features.split import Fold
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
        forecasts = conn.execute(
            "SELECT * FROM forecast_imbalance ORDER BY event_time"
        ).fetchdf()
    finally:
        conn.close()
    assert "forecast_day_ahead" not in table_names
    assert {name for name in table_names if name.startswith("forecast_")} == {
        "forecast_imbalance",
        "forecast_reserve",
    }
    assert list(forecasts.columns) == [
        "issue_time",
        "event_time",
        "q0_1",
        "q0_5",
        "q0_9",
    ]
    assert not forecasts.empty
    assert forecasts[["q0_1", "q0_5", "q0_9"]].notna().all().all()
    assert forecasts["issue_time"].tolist() == [
        imbalance_forecast_issue_time(event_time.to_pydatetime())
        for event_time in forecasts["event_time"]
    ]


def test_optimizer_forecast_persistence_keeps_exact_lgbm_quantiles_and_unique_keys(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    config = _make_config(tmp_path, date(2025, 1, 1), date(2025, 1, 2))
    event_time = datetime(2025, 1, 1, 12)
    issue_time = event_time - timedelta(minutes=60)
    index = pd.Index([10, 11])
    test = pd.DataFrame(
        {
            "issue_time": [issue_time, issue_time],
            "event_time": [event_time, event_time],
            "imbalance_price_eur_mwh": [999.0, 999.0],
            "realized_price_eur_mwh": [888.0, 888.0],
        },
        index=index,
    )
    predictions = {
        0.1: pd.Series([1.1, 11.1], index=index),
        0.5: pd.Series([5.5, 55.5], index=index),
        0.9: pd.Series([9.9, 99.9], index=index),
    }

    rows = secondary_module._optimizer_forecast_rows(test, predictions)
    secondary_module._persist_optimizer_forecasts(config, rows, [])
    conn = get_connection(config.duckdb_path)
    try:
        forecasts = conn.execute("SELECT * FROM forecast_imbalance").fetchdf()
    finally:
        conn.close()

    assert list(forecasts.columns) == [
        "issue_time",
        "event_time",
        "q0_1",
        "q0_5",
        "q0_9",
    ]
    assert len(forecasts) == 1
    assert forecasts.iloc[0][["q0_1", "q0_5", "q0_9"]].tolist() == pytest.approx(
        [11.1, 55.5, 99.9]
    )


def test_secondary_benchmark_persists_lgbm_not_seasonal_imbalance_quantiles(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    config = _make_config(tmp_path, date(2025, 1, 1), date(2025, 1, 5))
    train_time = datetime(2025, 1, 1, 12)
    test_time = datetime(2025, 1, 3, 12)
    conn = get_connection(config.duckdb_path)
    try:
        for table, value_column in SECONDARY_TARGETS:
            write_table(
                conn,
                table,
                [
                    {
                        "event_time": train_time,
                        "issue_time": train_time - timedelta(minutes=60),
                        value_column: 1.0,
                    },
                    {
                        "event_time": test_time,
                        "issue_time": test_time - timedelta(minutes=60),
                        value_column: 2.0,
                    },
                ],
            )
    finally:
        conn.close()
    fold = Fold(
        fold_id=0,
        train_start=date(2025, 1, 1),
        train_end=date(2025, 1, 2),
        test_start=date(2025, 1, 3),
        test_end=date(2025, 1, 4),
    )

    def sentinel(test: pd.DataFrame, base: float) -> dict[float, pd.Series]:
        return {
            quantile: pd.Series(base + quantile, index=test.index)
            for quantile in (0.1, 0.5, 0.9)
        }

    monkeypatch.setattr(secondary_module, "rolling_origin_folds", lambda *_: [fold])
    monkeypatch.setattr(
        secondary_module,
        "_forecast_seasonal_naive",
        lambda train, test, value_column: sentinel(test, 10.0),
    )
    lgbm_bases = iter([31.0, 32.0, 33.0, 20.0])
    monkeypatch.setattr(
        secondary_module,
        "lgbm_quantile_forecast",
        lambda train, test, value_column: sentinel(test, next(lgbm_bases)),
    )
    monkeypatch.setattr(secondary_module.mlflow, "set_tracking_uri", lambda *_: None)
    monkeypatch.setattr(secondary_module.mlflow, "set_experiment", lambda *_: None)
    monkeypatch.setattr(
        secondary_module.mlflow, "start_run", lambda **_: nullcontext()
    )
    monkeypatch.setattr(secondary_module.mlflow, "log_metrics", lambda *_: None)

    secondary_module.run_secondary_benchmark(config)
    conn = get_connection(config.duckdb_path)
    try:
        forecast = conn.execute("SELECT * FROM forecast_imbalance").fetchdf().iloc[0]
        reserves = conn.execute(
            "SELECT * FROM forecast_reserve ORDER BY product, direction"
        ).fetchdf()
    finally:
        conn.close()

    assert forecast[["q0_1", "q0_5", "q0_9"]].tolist() == pytest.approx(
        [20.1, 20.5, 20.9]
    )
    assert forecast["q0_5"] != pytest.approx(10.5)
    assert list(reserves.columns) == [
        "product",
        "direction",
        "issue_time",
        "delivery_time",
        "q0_1",
        "q0_5",
        "q0_9",
        "forecast_source",
    ]
    assert set(zip(reserves["product"], reserves["direction"], strict=True)) == {
        ("FCR_D", "up"),
        ("FCR_D", "down"),
        ("FCR_N", "symmetric"),
    }
    assert reserves["forecast_source"].eq("lgbm").all()
    expected_quantiles = {
        ("FCR_D", "up"): [31.1, 31.5, 31.9],
        ("FCR_D", "down"): [32.1, 32.5, 32.9],
        ("FCR_N", "symmetric"): [33.1, 33.5, 33.9],
    }
    for row in reserves.itertuples(index=False):
        assert [row.q0_1, row.q0_5, row.q0_9] == pytest.approx(
            expected_quantiles[(row.product, row.direction)]
        )


@pytest.mark.parametrize("preexisting", [False, True], ids=["empty", "replace"])
def test_optimizer_forecast_tables_replace_atomically(
    tmp_path, monkeypatch, preexisting: bool
) -> None:  # type: ignore[no-untyped-def]
    start, end = date(2024, 1, 1), date(2025, 6, 1)
    config = _make_config(tmp_path, start, end)
    _seed(config, start, end)
    build_all_facts(config)
    build_secondary_features(config)
    conn = get_connection(config.duckdb_path)
    try:
        if preexisting:
            write_table(conn, "forecast_imbalance", [{"sentinel": "old-imbalance"}])
            write_table(conn, "forecast_reserve", [{"sentinel": "old-reserve"}])
    finally:
        conn.close()
    real_write = secondary_module.write_table

    def fail_reserve(conn, table, rows, columns=None):  # type: ignore[no-untyped-def]
        if table == "forecast_reserve":
            raise RuntimeError("reserve forecast write failed")
        return real_write(conn, table, rows, columns=columns)

    monkeypatch.setattr(secondary_module, "write_table", fail_reserve)
    with pytest.raises(RuntimeError, match="reserve forecast write failed"):
        run_secondary_benchmark(config)

    conn = get_connection(config.duckdb_path)
    try:
        names = {
            row[0] for row in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'forecast_%'"
            ).fetchall()
        }
        if preexisting:
            assert conn.execute("SELECT * FROM forecast_imbalance").fetchall() == [("old-imbalance",)]
            assert conn.execute("SELECT * FROM forecast_reserve").fetchall() == [("old-reserve",)]
        else:
            assert names == set()
    finally:
        conn.close()
