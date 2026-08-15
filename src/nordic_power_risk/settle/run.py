"""Settlement: reconcile paper positions to observed prices (Phase 4).

Settlement consumes the persisted dispatch tables and joins each position against
the corresponding observed-price fact table, emitting signed P&L contributions keyed
by component. Every settlement is ex-post: only realized prices are used, never the
pre-gate forecasts that informed the decision. A position with no observed price is
left unsettled (fail closed) rather than settled against a fabricated value.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite, nan
from typing import Any

from nordic_power_risk.config import PipelineConfig
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.ingest.entsoe import ACTIVATION_PROCESS_TYPES

SETTLEMENT_COLUMNS = {
    "delivery_time": "TIMESTAMP",
    "component": "VARCHAR",
    "value_eur": "DOUBLE",
}


@dataclass(frozen=True)
class SettlementResult:
    table: str
    row_count: int
    total_pnl_eur: float


def _read_rows(config: PipelineConfig, table: str) -> list[dict[str, Any]]:
    conn = get_connection(config.duckdb_path)
    try:
        return conn.execute(f"SELECT * FROM {table}").fetchdf().to_dict("records")
    finally:
        conn.close()


def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _as_float(value: object) -> float:
    return nan if value is None else float(value)


def _energy_settlement(config: PipelineConfig) -> list[dict[str, Any]]:
    """Settle day-ahead energy against realized day-ahead prices.

    Degradation is emitted by `_imbalance_settlement` (the actual physical
    throughput subsumes the day-ahead commitment), never here.
    """
    energy_rows = _read_rows(config, "dispatch_energy")
    prices = {
        _as_datetime(row["event_time"]): _as_float(row["price_eur_mwh"])
        for row in _read_rows(config, "fact_day_ahead_price")
    }

    rows: list[dict[str, Any]] = []
    for interval in energy_rows:
        delivery = _as_datetime(interval["delivery_time"])
        price = prices.get(delivery)
        if price is None or not isfinite(price):
            # Fail closed: no observed price -> interval left unsettled.
            continue
        duration = _as_float(interval["duration_hours"])
        charge = _as_float(interval["charge_mw"])
        discharge = _as_float(interval["discharge_mw"])
        revenue = discharge * duration * price
        purchase = -charge * duration * price
        rows.append(
            {"delivery_time": delivery, "component": "day_ahead_revenue", "value_eur": revenue}
        )
        rows.append(
            {"delivery_time": delivery, "component": "day_ahead_purchase", "value_eur": purchase}
        )
    return rows


def _imbalance_settlement(config: PipelineConfig) -> list[dict[str, Any]]:
    """Settle imbalance positions at the final imbalance price (estimated as fallback)."""
    imbalance_rows = _read_rows(config, "dispatch_imbalance")
    final_prices: dict[datetime, float] = {}
    estimated_prices: dict[datetime, float] = {}
    for row in _read_rows(config, "fact_imbalance_price"):
        event_time = _as_datetime(row["event_time"])
        price = _as_float(row["imbalance_price_eur_mwh"])
        if row["price_type"] == "final":
            final_prices[event_time] = price
        else:
            estimated_prices[event_time] = price

    rows: list[dict[str, Any]] = []
    for interval in imbalance_rows:
        delivery = _as_datetime(interval["delivery_time"])
        price = final_prices.get(delivery, estimated_prices.get(delivery))
        if price is None or not isfinite(price):
            continue
        duration = _as_float(interval["duration_hours"])
        position = _as_float(interval["imbalance_position_mw"])
        rows.append(
            {
                "delivery_time": delivery,
                "component": "imbalance",
                "value_eur": position * duration * price,
            }
        )
        rows.append(
            {
                "delivery_time": delivery,
                "component": "degradation",
                "value_eur": -_as_float(interval["degradation_cost_eur"]),
            }
        )
    return rows


def _capacity_fact_table(product: str, direction: str) -> str | None:
    """Map a dispatched reserve product/direction to its observed-capacity fact table."""
    if product == "FCR_D":
        return f"fact_svk_fcr_d_{direction}"
    if product == "FCR_N":
        return "fact_svk_fcr_n"
    if product in {"AFRR", "MFRR"}:
        # The SvK aFRR/mFRR capacity resource is a single series; the repo's
        # data model does not split it by product or direction.
        return "fact_svk_afrr_mfrr_capacity"
    return None


def _reserve_capacity_settlement(config: PipelineConfig) -> list[dict[str, Any]]:
    """Settle conditionally-accepted reserve capacity at observed capacity prices."""
    reserve_rows = _read_rows(config, "dispatch_reserve")
    price_lookups = {
        table: {
            _as_datetime(row["event_time"]): _as_float(row["price"])
            for row in _read_rows(config, table)
        }
        for table in (
            "fact_svk_fcr_d_up",
            "fact_svk_fcr_d_down",
            "fact_svk_fcr_n",
            "fact_svk_afrr_mfrr_capacity",
        )
    }

    rows: list[dict[str, Any]] = []
    for interval in reserve_rows:
        if not interval.get("conditional_acceptance"):
            # Not accepted (or risk-blocked flat) -> no capacity revenue.
            continue
        table = _capacity_fact_table(str(interval["product"]), str(interval["direction"]))
        if table is None:
            continue
        delivery = _as_datetime(interval["delivery_time"])
        price = price_lookups[table].get(delivery)
        if price is None or not isfinite(price):
            continue
        capacity = _as_float(interval["capacity_mw"])
        duration = _as_float(interval["duration_hours"])
        rows.append(
            {
                "delivery_time": delivery,
                "component": "reserve_capacity",
                "value_eur": capacity * duration * price,
            }
        )
    return rows


def _reserve_activation_settlement(config: PipelineConfig) -> list[dict[str, Any]]:
    """Allocate observed aggregate activation pro rata to the asset's accepted share.

    The asset's accepted capacity is `dispatch_reserve.capacity_mw`; its share of the
    zone's aggregate activation is that capacity divided by the total procured volume
    for the same (product, direction) interval (fact_reserve_volume, ENTSO-E A75).
    Activated energy is settled at the observed activated-balancing energy price
    (fact_activation_price, ENTSO-E A84), falling back to the final imbalance price
    when A84 is unavailable. FCR-N symmetric activation nets to ~zero and is skipped;
    FCR-D and FFR activated energy are not published (T03) and settle no activation.
    """
    reserve_rows = _read_rows(config, "dispatch_reserve")
    activation: dict[tuple[datetime, str, str], float] = {}
    for row in _read_rows(config, "fact_activation"):
        key = (_as_datetime(row["event_time"]), str(row["product"]), str(row["direction"]))
        activation[key] = _as_float(row["activated_mw"])
    procured: dict[tuple[datetime, str, str], float] = {}
    for row in _read_rows(config, "fact_reserve_volume"):
        key = (_as_datetime(row["event_time"]), str(row["product"]), str(row["direction"]))
        procured[key] = _as_float(row["procured_mw"])
    activation_prices: dict[tuple[datetime, str, str], float] = {}
    for row in _read_rows(config, "fact_activation_price"):
        key = (_as_datetime(row["event_time"]), str(row["product"]), str(row["direction"]))
        activation_prices[key] = _as_float(row["activation_price_eur_mwh"])
    final_prices: dict[datetime, float] = {}
    for row in _read_rows(config, "fact_imbalance_price"):
        if row["price_type"] == "final":
            final_prices[_as_datetime(row["event_time"])] = _as_float(
                row["imbalance_price_eur_mwh"]
            )

    rows: list[dict[str, Any]] = []
    for interval in reserve_rows:
        if not interval.get("conditional_acceptance"):
            continue
        delivery = _as_datetime(interval["delivery_time"])
        product = str(interval["product"])
        direction = str(interval["direction"])
        if direction == "symmetric" or product not in ACTIVATION_PROCESS_TYPES:
            continue
        aggregate = activation.get((delivery, product, direction))
        if aggregate is None or not isfinite(aggregate) or aggregate <= 0:
            continue
        total_procured = procured.get((delivery, product, direction))
        if total_procured is None or not isfinite(total_procured) or total_procured <= 0:
            continue  # fail-closed: no denominator -> no pro-rata share
        capacity = _as_float(interval["capacity_mw"])
        duration = _as_float(interval["duration_hours"])
        share = min(capacity / total_procured, 1.0)
        energy = share * aggregate * duration
        price = activation_prices.get((delivery, product, direction))
        if price is None or not isfinite(price):
            price = final_prices.get(delivery)
        if price is None or not isfinite(price):
            continue
        sign = 1.0 if direction == "up" else -1.0
        rows.append(
            {
                "delivery_time": delivery,
                "component": "reserve_activation",
                "value_eur": sign * energy * price,
            }
        )
    return rows


def run_settlement(config: PipelineConfig) -> SettlementResult:
    """Settle every persisted position against observed prices and write `settlement`."""
    rows = (
        _energy_settlement(config)
        + _imbalance_settlement(config)
        + _reserve_capacity_settlement(config)
        + _reserve_activation_settlement(config)
    )
    conn = get_connection(config.duckdb_path)
    try:
        write_table(conn, "settlement", rows, columns=SETTLEMENT_COLUMNS)
    finally:
        conn.close()
    total = sum(row["value_eur"] for row in rows)
    return SettlementResult(table="settlement", row_count=len(rows), total_pnl_eur=total)


RECONCILIATION_COMPONENTS = (
    "day_ahead_revenue",
    "day_ahead_purchase",
    "imbalance",
    "reserve_capacity",
    "reserve_activation",
    "degradation",
    "imbalance_fee",
)


@dataclass(frozen=True)
class ReconciliationResult:
    table: str
    components: dict[str, float]
    total_pnl_eur: float
    residual_eur: float


def reconcile(config: PipelineConfig) -> ReconciliationResult:
    """Group settlement by component; report total and any unallocated residual."""
    components: dict[str, float] = {}
    for row in _read_rows(config, "settlement"):
        component = str(row["component"])
        value = _as_float(row["value_eur"])
        components[component] = components.get(component, 0.0) + value

    total = sum(components.values())
    residual = sum(
        value
        for component, value in components.items()
        if component not in RECONCILIATION_COMPONENTS
    )

    recon_rows = [{"component": c, "value_eur": components.get(c, 0.0)} for c in components]
    conn = get_connection(config.duckdb_path)
    try:
        write_table(
            conn,
            "reconciliation",
            recon_rows,
            columns={"component": "VARCHAR", "value_eur": "DOUBLE"},
        )
    finally:
        conn.close()
    return ReconciliationResult(
        table="reconciliation",
        components=components,
        total_pnl_eur=total,
        residual_eur=residual,
    )


__all__ = [
    "RECONCILIATION_COMPONENTS",
    "SETTLEMENT_COLUMNS",
    "ReconciliationResult",
    "SettlementResult",
    "reconcile",
    "run_settlement",
]
