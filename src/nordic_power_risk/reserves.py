"""Canonical reserve product/direction taxonomy and its table/forecaster mappings.

The seven reserve products — FCR_D/FCR_N/AFRR/MFRR across up/down/symmetric — flow
through every pipeline stage (facts, features, models, optimize, settle), and each
stage previously hand-synced its own product→table and product→forecaster mapping
with inconsistent spellings (FCRD vs FCR_D, etc.). This module is the single source
of truth for the enumeration and the fact/feature table names derived from it.

The raw source spellings (SvK's FCRD/FCRN, ENTSO-E's A51/A52/A47) are source
adapters and stay in ingest/facts; only the cross-layer canonical names live here.
"""

from __future__ import annotations

RESERVE_PRODUCTS: tuple[tuple[str, str], ...] = (
    ("FCR_D", "up"),
    ("FCR_D", "down"),
    ("FCR_N", "symmetric"),
    ("AFRR", "up"),
    ("AFRR", "down"),
    ("MFRR", "up"),
    ("MFRR", "down"),
)

# Products forecast by a tuned quantile LightGBM; the rest are seasonal-naive point models.
_LGBM_PRODUCTS = {"FCR_D", "FCR_N"}


def fact_table(product: str, direction: str) -> str:
    """Observed-capacity fact table for a reserve product/direction."""
    if product == "FCR_N":
        return "fact_svk_fcr_n"
    return f"fact_svk_{product.lower()}_{direction}"


def feature_table(product: str, direction: str) -> str:
    """Feature table for a reserve product/direction."""
    if product == "FCR_N":
        return "feature_fcr_n"
    return f"feature_{product.lower()}_{direction}"


def forecast_source(product: str) -> str:
    """Forecaster that produces a product's quantile forecast."""
    return "lgbm" if product in _LGBM_PRODUCTS else "seasonal_naive"


__all__ = ["RESERVE_PRODUCTS", "fact_table", "feature_table", "forecast_source"]
