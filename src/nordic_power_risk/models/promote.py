"""Promotion gate (Phase 5): auto-promote the newest challenger only when it beats
the champion on pinball loss with Diebold-Mariano significance vs seasonal-naive.

This is the predeclared single gate from T09/T11 — the metric the spec names, no
manual comparison step. The challenger is the newest registered version; the
champion is the version carrying the ``champion`` alias.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient

from nordic_power_risk.models.registry import (
    RegistryError,
    get_champion,
    latest_version,
    set_champion,
)

DM_SIGNIFICANCE_ALPHA = 0.05
PROMOTION_TAG = "promotion"
PROMOTION_REASON_TAG = "promotion_reason"


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    challenger_version: str
    champion_version: str  # the champion after this call
    reason: str


def _run_metrics(client: MlflowClient, run_id: str) -> dict[str, float]:
    return {k: float(v) for k, v in client.get_run(run_id).data.metrics.items()}


def _significant_vs_seasonal_naive(metrics: dict[str, float], alpha: float) -> bool:
    stat = metrics.get("dm_stat_vs_seasonal_naive")
    pvalue = metrics.get("dm_pvalue_vs_seasonal_naive")
    return stat is not None and pvalue is not None and stat < 0.0 and pvalue < alpha


def _tag_version(
    client: MlflowClient, model_name: str, version: str, outcome: str, reason: str
) -> None:
    client.set_model_version_tag(model_name, version, PROMOTION_TAG, outcome)
    client.set_model_version_tag(model_name, version, PROMOTION_REASON_TAG, reason)


def promote_champion(
    model_name: str, tracking_uri: str, *, alpha: float = DM_SIGNIFICANCE_ALPHA
) -> PromotionResult:
    """Promote the newest registered version if it clears the gate; else retain champion.

    Records the outcome as tags on the challenger version, so a failed challenger is
    never silently discarded.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    challenger = latest_version(model_name)
    if challenger.run_id is None:
        raise RegistryError(f"challenger version {challenger.version} has no run_id")
    challenger_metrics = _run_metrics(client, challenger.run_id)

    champion = get_champion(model_name)

    if champion is None:
        set_champion(model_name, challenger.version)
        _tag_version(client, model_name, challenger.version, "promoted", "first champion")
        return PromotionResult(True, challenger.version, challenger.version, "first champion")

    if challenger.run_id == champion.run_id:
        _tag_version(client, model_name, challenger.version, "noop", "challenger already champion")
        return PromotionResult(
            False, challenger.version, champion.version, "challenger already champion"
        )

    challenger_pinball = challenger_metrics.get("pinball_loss")
    if challenger_pinball is None:
        raise RegistryError(f"challenger version {challenger.version} lacks a pinball_loss metric")

    significant = _significant_vs_seasonal_naive(challenger_metrics, alpha)
    improved = challenger_pinball < champion.metrics.get("pinball_loss", float("inf"))

    if significant and improved:
        set_champion(model_name, challenger.version)
        _tag_version(
            client, model_name, challenger.version, "promoted", "significant improvement"
        )
        return PromotionResult(
            True, challenger.version, challenger.version, "significant improvement"
        )

    reason = (
        "no pinball improvement"
        if significant
        else "not significant vs seasonal-naive"
    )
    _tag_version(client, model_name, challenger.version, "rejected", reason)
    return PromotionResult(False, challenger.version, champion.version, reason)


__all__ = [
    "DM_SIGNIFICANCE_ALPHA",
    "PromotionResult",
    "promote_champion",
]
