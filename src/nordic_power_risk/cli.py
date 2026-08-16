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
def facts() -> None:
    """Build event_time/issue_time fact tables from raw_* (Phase 1)."""
    from nordic_power_risk.facts.run import build_all_facts

    config = get_config()
    results = build_all_facts(config)
    for result in results:
        typer.echo(f"{result.table}: {result.row_count} rows -> {config.duckdb_path}")


@app.command()
def features() -> None:
    """Build day-ahead + secondary (FCR/imbalance) feature tables (Phase 2)."""
    from nordic_power_risk.features.run import build_all_features, build_secondary_features

    config = get_config()
    result = build_all_features(config)
    typer.echo(f"{result.table}: {result.row_count} rows -> {config.duckdb_path}")
    for result in build_secondary_features(config):
        typer.echo(f"{result.table}: {result.row_count} rows -> {config.duckdb_path}")


@app.command()
def models() -> None:
    """Build primary, secondary, and tertiary market forecasts."""
    from nordic_power_risk.models.run import run_benchmark_ladder, select_best_rung
    from nordic_power_risk.models.secondary_run import (
        run_secondary_benchmark,
        run_tertiary_forecast,
    )

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

    secondary_results = run_secondary_benchmark(config)
    for secondary in secondary_results:
        typer.echo(
            f"{secondary.target}/{secondary.rung} (n={secondary.n_obs}): "
            f"pinball={secondary.pinball_loss:.4f} crps={secondary.crps:.4f} "
            f"coverage_80={secondary.coverage_80:.3f} "
            f"winkler_80={secondary.winkler_80:.4f} pit_mean={secondary.pit_mean:.3f}"
        )

    tertiary_results = run_tertiary_forecast(config)
    for tertiary in tertiary_results:
        typer.echo(
            f"{tertiary.target}/{tertiary.source} (n={tertiary.n_obs}): mae={tertiary.mae:.4f}"
        )


@app.command()
def optimize() -> None:
    """Run integrated risk-gated balancing, energy, FCR, and imbalance backtest."""
    from nordic_power_risk.risk.backtest import run_risk_backtest

    config = get_config()
    try:
        integrated = run_risk_backtest(config)
        result = integrated.dispatch
    except (RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"{result.table}: {result.row_count} rows, "
        f"objective={result.objective_eur:.2f} EUR; "
        f"{result.imbalance.table}: {result.imbalance.row_count} rows, "
        f"objective={result.imbalance.objective_eur:.2f} EUR; "
        f"{result.reserve.table}: {result.reserve.row_count} rows, "
        f"capacity-value={result.reserve.capacity_value_eur:.2f} EUR; "
        f"risk={integrated.gate_state}, decisions={integrated.decision_count}, "
        f"blocked={integrated.blocked_decisions} -> {config.duckdb_path}"
    )


@app.command()
def risk() -> None:
    """Report the latest append-only CVaR/drawdown gate state."""
    from nordic_power_risk.risk.run import read_risk_status

    status = read_risk_status(get_config())
    typer.echo(
        f"risk: gate={status.gate_state}, records={status.record_count}, "
        f"drawdown={status.drawdown_eur if status.drawdown_eur is not None else 'n/a'} EUR, "
        f"fallback={status.fallback_reason or 'none'}"
    )


@app.command()
def settle() -> None:
    """Settle, reconcile, compare policies, attribute, and stress (Phase 4)."""
    from nordic_power_risk.settle.attribution import attribute
    from nordic_power_risk.settle.compare import compare_policies
    from nordic_power_risk.settle.run import reconcile, run_settlement
    from nordic_power_risk.settle.stress import run_stresses

    config = get_config()
    result = run_settlement(config)
    reconciliation = reconcile(config)
    comparison = compare_policies(config)
    attribution = attribute(config)
    stress = run_stresses(config)

    typer.echo(
        f"{result.table}: {result.row_count} rows, "
        f"total-pnl={result.total_pnl_eur:.2f} EUR; "
        f"reconcile residual={reconciliation.residual_eur:.2f} EUR"
    )
    typer.echo(
        "policies: " + ", ".join(f"{n}={v:.2f}" for n, v in comparison.policies.items())
    )
    typer.echo(
        "attribution: "
        + ", ".join(f"{n}={v:.2f}" for n, v in attribution.components.items())
        + f" (gap={attribution.gap_eur:.2f} EUR)"
    )
    typer.echo(
        f"stress: baseline={stress.baseline_eur:.2f} EUR, "
        + ", ".join(f"{n}={v:.2f}" for n, v in stress.scenarios.items())
    )


@app.command()
def monitor() -> None:
    """Drift and performance monitoring report (Phase 5)."""
    from nordic_power_risk.monitor.run import run_monitoring

    result = run_monitoring(get_config())
    typer.echo(
        f"monitor: missingness={result.missingness}, "
        f"mae={result.forecast_mae}, coverage_80={result.interval_coverage_80}, "
        f"pnl={result.realized_pnl_eur}, max_drawdown={result.max_drawdown_eur}, "
        f"breaches={result.breach_count}, failures={result.optimizer_failures}, "
        f"latency_h={result.data_latency_hours}, drift_share={result.drift_share} "
        f"-> {result.summary_path}"
    )


@app.command()
def rollback() -> None:
    """Revert the champion alias to the previous registered version (Phase 5)."""
    from nordic_power_risk.models.registry import DAY_AHEAD_MODEL, rollback_champion

    config = get_config()
    version = rollback_champion(DAY_AHEAD_MODEL, config.mlflow_tracking_uri)
    typer.echo(f"rolled back: champion -> v{version}")


@app.command()
def promote() -> None:
    """Promote the newest challenger if it clears the DM-significance gate (Phase 5)."""
    from nordic_power_risk.models.promote import promote_champion
    from nordic_power_risk.models.registry import DAY_AHEAD_MODEL

    config = get_config()
    result = promote_champion(DAY_AHEAD_MODEL, config.mlflow_tracking_uri)
    if result.promoted:
        typer.echo(
            f"promoted: v{result.challenger_version} -> champion "
            f"(now v{result.champion_version}): {result.reason}"
        )
    else:
        typer.echo(
            f"retained: champion v{result.champion_version}; "
            f"challenger v{result.challenger_version} rejected: {result.reason}"
        )


@app.command()
def bridge() -> None:
    """Thesis-bridge analysis: MDE gate + pre/post imbalance-forecast difficulty (Phase 6)."""
    from nordic_power_risk.bridge.run import run_thesis_bridge

    config = get_config()
    try:
        result = run_thesis_bridge(config)
    except (RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"bridge: framing={result.framing}, "
        f"MDE={result.gate.mde:.4f} (sigma={result.gate.sigma:.4f}), "
        f"n_pre={result.gate.n_pre}, n_post={result.gate.n_post}"
    )
    typer.echo(
        f"difficulty: pre pinball={result.pre.pinball_loss_lgbm:.4f}, "
        f"post pinball={result.post.pinball_loss_lgbm:.4f}, "
        f"effect={result.effect_pinball:+.4f} "
        f"(detected={result.effect_detected})"
    )
    typer.echo(f"chapter: {result.chapter_path}")


if __name__ == "__main__":
    app()
