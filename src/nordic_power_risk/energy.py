"""Shared battery energy identities — single source of truth across optimizer, risk, and settlement.

Two rules are load-bearing everywhere: the state-of-charge evolution and the cash
value of a net-discharge position. Each was previously reimplemented inline in
three to five modules with the sign conventions hand-synced; a drift here is a
silent P&L or physics bug. They live here so the convention is defined once.

``soc_after`` is also used symbolically inside the MILP constraints, where the
``charge_mw``/``discharge_mw``/``soc_mwh`` arguments are Pyomo expressions rather
than floats; the arithmetic is deliberately plain Python so it works in both
numeric and symbolic contexts.
"""

from __future__ import annotations


def soc_after(
    soc_mwh: float,
    charge_mw: float,
    discharge_mw: float,
    duration_hours: float,
    one_way_efficiency: float,
) -> float:
    """State of charge after one interval.

    ``soc' = soc + efficiency * charge * duration - discharge * duration / efficiency``
    """
    return (
        soc_mwh
        + one_way_efficiency * charge_mw * duration_hours
        - discharge_mw * duration_hours / one_way_efficiency
    )


def soc_before(
    soc_mwh: float,
    charge_mw: float,
    discharge_mw: float,
    duration_hours: float,
    one_way_efficiency: float,
) -> float:
    """State of charge before an interval, recovered from the after-state."""
    return soc_after(soc_mwh, -charge_mw, -discharge_mw, duration_hours, one_way_efficiency)


def energy_value(price_eur_mwh: float, net_discharge_mw: float, duration_hours: float) -> float:
    """Cash value of a net-discharge position at a price (degradation excluded)."""
    return price_eur_mwh * net_discharge_mw * duration_hours


__all__ = ["energy_value", "soc_after", "soc_before"]
