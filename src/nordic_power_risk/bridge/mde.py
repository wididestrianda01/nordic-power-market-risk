"""Minimum-detectable-effect gate for the thesis-bridge (Phase 6 ticket 01).

Pure statistics, no DB/IO: given a per-observation loss standard deviation and
the two sample sizes, compute the smallest mean-loss shift a two-sample
comparison can detect at a fixed significance and power, then apply the
pre-declared relevance threshold to choose the chapter framing.

The framing is decided here, *before* the interrupted-time-series result is
seen, so it cannot be reverse-engineered from the p-value (T12 a-priori rule).
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats

# Pre-declared relevance threshold: a shift is policy-relevant if the MDE is at
# most this fraction of the pre-period baseline loss (i.e. the design can detect
# a 20% relative worsening of forecast difficulty).
MDE_RELEVANCE_FRACTION = 0.20

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80

FRAMING_INFERENTIAL = "inferential_exhibit"
FRAMING_METHODOLOGY = "methodology_demonstration"


def minimum_detectable_effect(
    sigma: float,
    n_pre: int,
    n_post: int,
    *,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> float:
    """Smallest detectable absolute mean-loss shift for a two-sample comparison.

    ``sigma`` is the pooled per-observation loss standard deviation; ``n_pre``
    and ``n_post`` are the pre/post observation counts.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if n_pre <= 0 or n_post <= 0:
        raise ValueError("sample sizes must be positive")
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    return float((z_alpha + z_power) * sigma * (1.0 / n_pre + 1.0 / n_post) ** 0.5)


def relative_mde(mde: float, baseline_loss: float) -> float:
    """MDE expressed as a fraction of the pre-period baseline pinball loss."""
    if baseline_loss <= 0:
        raise ValueError("baseline_loss must be positive")
    return mde / baseline_loss


def decide_framing(
    mde: float,
    baseline_loss: float,
    *,
    relevance_fraction: float = MDE_RELEVANCE_FRACTION,
) -> str:
    """Apply the pre-declared rule.

    Inferential exhibit iff the MDE is small enough (<= ``relevance_fraction`` of
    the baseline loss) to be policy-relevant; otherwise methodology demonstration.
    """
    if relative_mde(mde, baseline_loss) <= relevance_fraction:
        return FRAMING_INFERENTIAL
    return FRAMING_METHODOLOGY


@dataclass(frozen=True)
class MdeGate:
    sigma: float
    n_pre: int
    n_post: int
    alpha: float
    power: float
    mde: float
    baseline_loss: float
    relative_mde: float
    relevance_fraction: float
    framing: str


def compute_mde_gate(
    sigma: float,
    n_pre: int,
    n_post: int,
    baseline_loss: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    relevance_fraction: float = MDE_RELEVANCE_FRACTION,
) -> MdeGate:
    mde = minimum_detectable_effect(sigma, n_pre, n_post, alpha=alpha, power=power)
    return MdeGate(
        sigma=sigma,
        n_pre=n_pre,
        n_post=n_post,
        alpha=alpha,
        power=power,
        mde=mde,
        baseline_loss=baseline_loss,
        relative_mde=relative_mde(mde, baseline_loss),
        relevance_fraction=relevance_fraction,
        framing=decide_framing(mde, baseline_loss, relevance_fraction=relevance_fraction),
    )


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_POWER",
    "FRAMING_INFERENTIAL",
    "FRAMING_METHODOLOGY",
    "MDE_RELEVANCE_FRACTION",
    "MdeGate",
    "compute_mde_gate",
    "decide_framing",
    "minimum_detectable_effect",
    "relative_mde",
]
