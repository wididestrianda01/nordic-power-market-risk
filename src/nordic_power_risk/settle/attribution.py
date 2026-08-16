"""P&L attribution (Phase 4, ticket 08).

Decomposes the gap between the realized paper policy and the perfect-foresight
upper bound into forecast error, constraint cost, and degradation. Components
are signed to sum to the gap (attribution identity).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from nordic_power_risk.config import PipelineConfig
from nordic_power_risk.ingest.duckdb_io import fetch_scalar, get_connection, write_table
from nordic_power_risk.settle.compare import compare_policies
from nordic_power_risk.settle.run import reconcile

ATTRIBUTION_COLUMNS = {"component": "VARCHAR", "value_eur": "DOUBLE"}


@dataclass(frozen=True)
class AttributionResult:
    table: str
    gap_eur: float
    components: dict[str, float]


def _read_sum(config: PipelineConfig, table: str, column: str) -> float:
    conn = get_connection(config.duckdb_path)
    try:
        return float(fetch_scalar(conn, f"SELECT COALESCE(SUM({column}), 0.0) FROM {table}"))
    finally:
        conn.close()


def _forecast_objective(config: PipelineConfig) -> float:
    """Forecast-based cash P&L the optimizer expected (terminal value excluded)."""
    energy = _read_sum(config, "dispatch_energy", "objective_eur") - _read_sum(
        config, "dispatch_energy", "terminal_value_eur"
    )
    imbalance = _read_sum(config, "dispatch_imbalance", "objective_eur") - _read_sum(
        config, "dispatch_imbalance", "terminal_value_eur"
    )
    reserve = _read_sum(config, "dispatch_reserve", "capacity_value_eur")
    return energy + imbalance + reserve


def attribute(config: PipelineConfig) -> AttributionResult:
    """Attribute the perfect-foresight gap to forecast error, constraint cost, and degradation."""
    comparison = compare_policies(config)
    recon = reconcile(config)
    optimized = comparison.policies["optimized"]
    perfect = comparison.policies["perfect_foresight"]
    gap = perfect - optimized

    forecast_error = _forecast_objective(config) - optimized
    degradation = -recon.components.get("degradation", 0.0)
    # Reserve capacity/activation are already inside the perfect-foresight bound
    # (compare.py), so they cancel out of the gap; no separate "unavailable
    # reserve" term remains once the benchmark is co-optimized. The residual —
    # physical constraint cost and unmodeled frictions — is constraint cost.
    constraint_cost = gap - forecast_error - degradation

    components = {
        "forecast_error": forecast_error,
        "constraint_cost": constraint_cost,
        "degradation": degradation,
    }
    assert isclose(sum(components.values()), gap, abs_tol=1e-6)

    rows = [{"component": name, "value_eur": value} for name, value in components.items()]
    conn = get_connection(config.duckdb_path)
    try:
        write_table(conn, "attribution", rows, columns=ATTRIBUTION_COLUMNS)
    finally:
        conn.close()
    return AttributionResult(table="attribution", gap_eur=gap, components=components)


__all__ = ["ATTRIBUTION_COLUMNS", "AttributionResult", "attribute"]
