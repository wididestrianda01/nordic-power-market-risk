"""Single typer CLI: `nordic-risk <stage>`. Thin wrapper — logic lives in nordic_power_risk."""

from __future__ import annotations

import typer

from nordic_power_risk.config import get_config, get_settings

app = typer.Typer(no_args_is_help=True)


@app.command()
def ingest() -> None:
    """Pull ENTSO-E/eSett/SvK/SMHI into DuckDB and write the source manifest."""
    from nordic_power_risk.ingest.run import ingest_all

    config = get_config()
    settings = get_settings()
    try:
        entries = ingest_all(config, settings)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    for entry in entries:
        typer.echo(f"{entry.name}: {entry.row_count} rows -> {config.duckdb_path}")


def _not_implemented(stage: str) -> None:
    typer.echo(f"nordic-risk {stage}: not implemented yet (Phase 0 is ingest-only)", err=True)
    raise typer.Exit(1)


@app.command()
def validate() -> None:
    """Pandera schema validation of every raw_* table (Phase 1)."""
    from nordic_power_risk.validate.run import validate_all

    config = get_config()
    try:
        results = validate_all(config)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    failed = False
    for result in results:
        if result.passed:
            typer.echo(f"{result.table}: PASS")
        else:
            failed = True
            typer.echo(f"{result.table}: FAIL", err=True)
            typer.echo(result.failure_cases, err=True)
    if failed:
        raise typer.Exit(1)


@app.command()
def features() -> None:
    """Day-ahead SE3 feature table: as_of()-gated price lags + calendar features (Phase 2)."""
    from nordic_power_risk.features.run import build_all_features

    config = get_config()
    result = build_all_features(config)
    typer.echo(f"{result.table}: {result.row_count} rows -> {config.duckdb_path}")


@app.command()
def models() -> None:
    """Naive/seasonal-naive/LEAR/DNN benchmark ladder over T08 rolling-origin folds (Phase 2)."""
    from nordic_power_risk.models.run import run_benchmark_ladder, select_best_rung

    config = get_config()
    results = run_benchmark_ladder(config)
    for result in results:
        dm = (
            f"DM vs naive: stat={result.dm_stat:.3f} p={result.dm_pvalue:.4f}"
            if result.dm_stat is not None and result.dm_pvalue is not None
            else "DM vs naive: n/a (reference rung)"
        )
        dm_sn = (
            f"DM vs seasonal_naive: stat={result.dm_stat_vs_seasonal_naive:.3f} "
            f"p={result.dm_pvalue_vs_seasonal_naive:.4f}"
            if result.dm_stat_vs_seasonal_naive is not None
            and result.dm_pvalue_vs_seasonal_naive is not None
            else "DM vs seasonal_naive: n/a"
        )
        typer.echo(
            f"{result.rung} (n={result.n_obs}): pinball={result.pinball_loss:.4f} "
            f"crps={result.crps:.4f} coverage_80={result.coverage_80:.3f} "
            f"winkler_80={result.winkler_80:.4f} pit_mean={result.pit_mean:.3f} {dm} {dm_sn}"
        )

    best = select_best_rung(results)
    typer.echo(f"promoted: {best.rung} (pinball={best.pinball_loss:.4f})")


@app.command()
def optimize() -> None:
    """Optimize fixed D-1 energy, FCR capacity, and T-60 imbalance recourse."""
    from nordic_power_risk.optimize.run import run_energy_dispatch

    config = get_config()
    try:
        result = run_energy_dispatch(config)
    except (RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"{result.table}: {result.row_count} rows, "
        f"objective={result.objective_eur:.2f} EUR; "
        f"{result.imbalance.table}: {result.imbalance.row_count} rows, "
        f"objective={result.imbalance.objective_eur:.2f} EUR; "
        f"{result.reserve.table}: {result.reserve.row_count} rows, "
        f"capacity-value={result.reserve.capacity_value_eur:.2f} EUR "
        f"-> {config.duckdb_path}"
    )


@app.command()
def risk() -> None:
    """CVaR/drawdown risk checks (Phase 3)."""
    _not_implemented("risk")


@app.command()
def settle() -> None:
    """Settlement and P&L attribution (Phase 4)."""
    _not_implemented("settle")


@app.command()
def monitor() -> None:
    """Drift and performance monitoring (Phase 5)."""
    _not_implemented("monitor")


@app.command()
def promote() -> None:
    """Model promotion gate (Phase 5)."""
    _not_implemented("promote")


if __name__ == "__main__":
    app()
