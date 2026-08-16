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
from math import isfinite
from typing import Any

from nordic_power_risk.config import PipelineConfig
from nordic_power_risk.energy import energy_value
from nordic_power_risk.ingest.duckdb_io import (
    coerce_datetime,
    coerce_float,
    get_connection,
    read_table,
    write_table,
)
from nordic_power_risk.ingest.entsoe import ACTIVATION_PROCESS_TYPES
from nordic_power_risk.reserves import RESERVE_PRODUCTS, fact_table

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


def _energy_settlement(config: PipelineConfig) -> list[dict[str, Any]]:
    """Settle day-ahead energy against realized day-ahead prices.

    Degradation is emitted by `_imbalance_settlement` (the actual physical
    throughput subsumes the day-ahead commitment), never here.
    """
    energy_rows = read_table(config, "dispatch_energy")
    prices = {
        coerce_datetime(row["event_time"]): coerce_float(row["price_eur_mwh"])
        for row in read_table(config, "fact_day_ahead_price")
    }

    rows: list[dict[str, Any]] = []
    for interval in energy_rows:
        delivery = coerce_datetime(interval["delivery_time"])
        price = prices.get(delivery)
        if price is None or not isfinite(price):
            # Fail closed: no observed price -> interval left unsettled.
            continue
        duration = coerce_float(interval["duration_hours"])
        charge = coerce_float(interval["charge_mw"])
        discharge = coerce_float(interval["discharge_mw"])
        revenue = energy_value(price, discharge, duration)
        purchase = energy_value(price, -charge, duration)
        rows.append(
            {"delivery_time": delivery, "component": "day_ahead_revenue", "value_eur": revenue}
        )
        rows.append(
            {"delivery_time": delivery, "component": "day_ahead_purchase", "value_eur": purchase}
        )
    return rows


def _imbalance_settlement(config: PipelineConfig) -> list[dict[str, Any]]:
    """Settle imbalance positions at the final imbalance price (estimated as fallback)."""
    imbalance_rows = read_table(config, "dispatch_imbalance")
    final_prices: dict[datetime, float] = {}
    estimated_prices: dict[datetime, float] = {}
    for row in read_table(config, "fact_imbalance_price"):
        event_time = coerce_datetime(row["event_time"])
        price = coerce_float(row["imbalance_price_eur_mwh"])
        if row["price_type"] == "final":
            final_prices[event_time] = price
        else:
            estimated_prices[event_time] = price

    rows: list[dict[str, Any]] = []
    for interval in imbalance_rows:
        delivery = coerce_datetime(interval["delivery_time"])
        settled_price = final_prices.get(delivery, estimated_prices.get(delivery))
        if settled_price is None or not isfinite(settled_price):
            continue
        duration = coerce_float(interval["duration_hours"])
        position = coerce_float(interval["imbalance_position_mw"])
        rows.append(
            {
                "delivery_time": delivery,
                "component": "imbalance",
                "value_eur": energy_value(settled_price, position, duration),
            }
        )
        rows.append(
            {
                "delivery_time": delivery,
                "component": "degradation",
                "value_eur": -coerce_float(interval["degradation_cost_eur"]),
            }
        )
    return rows


def _reserve_capacity_settlement(config: PipelineConfig) -> list[dict[str, Any]]:
    """Settle conditionally-accepted reserve capacity at observed capacity prices."""
    reserve_rows = read_table(config, "dispatch_reserve")
    price_lookups = {
        table: {
            coerce_datetime(row["event_time"]): coerce_float(row["price"])
            for row in read_table(config, table)
        }
        for table in (fact_table(p, d) for p, d in RESERVE_PRODUCTS)
    }

    rows: list[dict[str, Any]] = []
    for interval in reserve_rows:
        if not interval.get("conditional_acceptance"):
            # Not accepted (or risk-blocked flat) -> no capacity revenue.
            continue
        table = fact_table(str(interval["product"]), str(interval["direction"]))
        if table not in price_lookups:
            continue
        delivery = coerce_datetime(interval["delivery_time"])
        price = price_lookups[table].get(delivery)
        if price is None or not isfinite(price):
            continue
        capacity = coerce_float(interval["capacity_mw"])
        duration = coerce_float(interval["duration_hours"])
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
    reserve_rows = read_table(config, "dispatch_reserve")
    activation: dict[tuple[datetime, str, str], float] = {}
    for row in read_table(config, "fact_activation"):
        key = (coerce_datetime(row["event_time"]), str(row["product"]), str(row["direction"]))
        activation[key] = coerce_float(row["activated_mw"])
    procured: dict[tuple[datetime, str, str], float] = {}
    for row in read_table(config, "fact_reserve_volume"):
        key = (coerce_datetime(row["event_time"]), str(row["product"]), str(row["direction"]))
        procured[key] = coerce_float(row["procured_mw"])
    activation_prices: dict[tuple[datetime, str, str], float] = {}
    for row in read_table(config, "fact_activation_price"):
        key = (coerce_datetime(row["event_time"]), str(row["product"]), str(row["direction"]))
        activation_prices[key] = coerce_float(row["activation_price_eur_mwh"])
    final_prices: dict[datetime, float] = {}
    for row in read_table(config, "fact_imbalance_price"):
        if row["price_type"] == "final":
            final_prices[coerce_datetime(row["event_time"])] = coerce_float(
                row["imbalance_price_eur_mwh"]
            )

    rows: list[dict[str, Any]] = []
    for interval in reserve_rows:
        if not interval.get("conditional_acceptance"):
            continue
        delivery = coerce_datetime(interval["delivery_time"])
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
        capacity = coerce_float(interval["capacity_mw"])
        duration = coerce_float(interval["duration_hours"])
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
    for row in read_table(config, "settlement"):
        component = str(row["component"])
        value = coerce_float(row["value_eur"])
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
