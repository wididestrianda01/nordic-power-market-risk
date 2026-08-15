"""Pinball loss, CRPS/PIT/Winkler diagnostics, and the Diebold-Mariano significance test."""

from __future__ import annotations

import numpy as np
from scipy import stats

QUANTILE_GRID: tuple[float, ...] = (
    0.01,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def mean_pinball_over_grid(y_true: np.ndarray, quantile_preds: dict[float, np.ndarray]) -> float:
    losses = [pinball_loss(y_true, quantile_preds[q], q) for q in quantile_preds]
    return float(np.mean(losses))


def mean_pinball_per_row(y_true: np.ndarray, quantile_preds: dict[float, np.ndarray]) -> np.ndarray:
    """Per-observation mean pinball loss across the quantile grid (for DM test loss series)."""
    per_quantile = []
    for q, pred in quantile_preds.items():
        diff = y_true - pred
        per_quantile.append(np.maximum(q * diff, (q - 1) * diff))
    return np.mean(np.stack(per_quantile, axis=0), axis=0)


def crps_approx(y_true: np.ndarray, quantile_preds: dict[float, np.ndarray]) -> float:
    """CRPS ~= 2 * mean pinball loss over the quantile grid (exact as grid -> continuum)."""
    return 2.0 * mean_pinball_over_grid(y_true, quantile_preds)


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    width = upper - lower
    below = y_true < lower
    above = y_true > upper
    score = np.where(below, width + (2 / alpha) * (lower - y_true), width)
    score = np.where(above, width + (2 / alpha) * (y_true - upper), score)
    return float(np.mean(score))


def pit_values(y_true: np.ndarray, quantile_preds: dict[float, np.ndarray]) -> np.ndarray:
    """Per-observation PIT: interpolated quantile level at which y_true falls."""
    levels = np.array(sorted(quantile_preds))
    preds = np.stack([quantile_preds[q] for q in levels], axis=1)  # (n, n_quantiles)
    pit = np.empty(len(y_true))
    for i in range(len(y_true)):
        pit[i] = np.interp(y_true[i], preds[i], levels)
    return pit


def diebold_mariano_test(
    loss_challenger: np.ndarray, loss_baseline: np.ndarray
) -> tuple[float, float]:
    """One-step DM test (Diebold & Mariano 1995, HLN small-sample correction).

    H0: challenger and baseline have equal expected loss. Two-sided p-value.
    """
    d = np.asarray(loss_challenger) - np.asarray(loss_baseline)
    n = len(d)
    d_mean = d.mean()
    var_d = np.var(d, ddof=0) / n
    if var_d == 0:
        return (0.0, 1.0) if d_mean == 0 else (np.sign(d_mean) * np.inf, 0.0)
    dm_stat = d_mean / np.sqrt(var_d)
    correction = np.sqrt((n + 1) / n)
    dm_stat_corrected = dm_stat * correction
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat_corrected), df=n - 1))
    return float(dm_stat_corrected), float(p_value)


__all__ = [
    "QUANTILE_GRID",
    "crps_approx",
    "diebold_mariano_test",
    "interval_coverage",
    "mean_pinball_over_grid",
    "mean_pinball_per_row",
    "pinball_loss",
    "pit_values",
    "winkler_score",
]
