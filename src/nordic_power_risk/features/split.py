"""Rolling-origin split protocol (T08, reused verbatim): expanding window, monthly
re-fit, last-12-months rolling test folds, 1-day embargo, final 3-months untouched holdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: date
    train_end: date  # exclusive
    test_start: date
    test_end: date  # exclusive


def rolling_origin_folds(
    spine_start: date,
    spine_end: date,
    *,
    holdout_months: int = 3,
    test_months: int = 12,
    embargo_days: int = 1,
) -> list[Fold]:
    """Expanding-window folds over the 12 months before the holdout; holdout itself untouched."""
    holdout_start = (pd.Timestamp(spine_end) - pd.DateOffset(months=holdout_months)).date()
    first_test_start = (pd.Timestamp(holdout_start) - pd.DateOffset(months=test_months)).date()

    folds = []
    for fold_id, test_start in enumerate(
        pd.date_range(first_test_start, holdout_start, freq="MS", inclusive="left")
    ):
        test_start_date = test_start.date()
        test_end_date = (test_start + pd.DateOffset(months=1)).date()
        if test_end_date > holdout_start:
            break
        train_end_date = (pd.Timestamp(test_start_date) - pd.Timedelta(days=embargo_days)).date()
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=spine_start,
                train_end=train_end_date,
                test_start=test_start_date,
                test_end=test_end_date,
            )
        )
    return folds


def holdout_window(spine_end: date, *, holdout_months: int = 3) -> tuple[date, date]:
    holdout_start = (pd.Timestamp(spine_end) - pd.DateOffset(months=holdout_months)).date()
    return holdout_start, spine_end


__all__ = ["Fold", "holdout_window", "rolling_origin_folds"]
