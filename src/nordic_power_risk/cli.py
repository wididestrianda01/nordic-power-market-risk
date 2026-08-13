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
    """Feature engineering (Phase 1/2)."""
    _not_implemented("features")


@app.command()
def models() -> None:
    """Forecast model training (Phase 2)."""
    _not_implemented("models")


@app.command()
def optimize() -> None:
    """MILP dispatch optimization (Phase 3)."""
    _not_implemented("optimize")


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
