"""Smoke tests for the report figure module."""

from datetime import date
from pathlib import Path

from nordic_power_risk.config import PipelineConfig, Window
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.report.figures import render_all


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2020, 1, 1), end=date(2020, 1, 2))},
        duckdb_path=Path(tmp_path / "db.duckdb"),
        manifest_path=Path(tmp_path / "manifest.json"),
    )


def test_render_all_writes_hero_and_skips_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path)
    conn = get_connection(config.duckdb_path)
    write_table(
        conn,
        "comparison",
        [
            {"policy": "no_trade", "total_pnl_eur": 0.0},
            {"policy": "heuristic", "total_pnl_eur": 5.0},
            {"policy": "optimized", "total_pnl_eur": 12.0},
            {"policy": "perfect_foresight", "total_pnl_eur": 20.0},
        ],
        columns={"policy": "VARCHAR", "total_pnl_eur": "DOUBLE"},
    )
    conn.close()

    written = render_all(config)

    hero = Path("docs/figures/hero_cumulative_pnl.png")
    assert hero.exists()
    assert str(hero) in [str(p) for p in written]
    # The other figures skip because their source tables are absent.
    assert not Path("docs/figures/quantile_forecast_fan.png").exists()


def test_render_all_empty_db_returns_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path)
    assert render_all(config) == []
