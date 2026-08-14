from datetime import date

from nordic_power_risk.features.split import holdout_window, rolling_origin_folds


def test_folds_expanding_window_from_spine_start() -> None:
    folds = rolling_origin_folds(date(2019, 1, 1), date(2026, 6, 30))
    assert all(f.train_start == date(2019, 1, 1) for f in folds)


def test_folds_cover_roughly_twelve_months_before_holdout() -> None:
    folds = rolling_origin_folds(date(2019, 1, 1), date(2026, 6, 30))
    assert len(folds) == 11
    assert folds[0].test_start == date(2025, 4, 1)
    assert folds[-1].test_end == date(2026, 3, 1)


def test_folds_never_spill_into_holdout() -> None:
    folds = rolling_origin_folds(date(2019, 1, 1), date(2026, 6, 30))
    holdout_start, _ = holdout_window(date(2026, 6, 30))
    assert all(f.test_end <= holdout_start for f in folds)


def test_folds_respect_embargo_gap() -> None:
    folds = rolling_origin_folds(date(2019, 1, 1), date(2026, 6, 30), embargo_days=1)
    for fold in folds:
        assert fold.train_end == date.fromordinal(fold.test_start.toordinal() - 1)


def test_holdout_window_is_last_three_months() -> None:
    holdout_start, holdout_end = holdout_window(date(2026, 6, 30), holdout_months=3)
    assert holdout_start == date(2026, 3, 30)
    assert holdout_end == date(2026, 6, 30)
