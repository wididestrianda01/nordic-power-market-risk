"""Direct quantile regression (LightGBM) for secondary targets (Phase 2 ticket 04).

Unlike the day-ahead ladder's residual-quantile trick (a point forecast plus
in-sample residual quantiles), LightGBM's `objective="quantile"` fits one model
per quantile level directly -- a LightGBM API constraint, not a design choice.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from nordic_power_risk.models.design_matrix import build_design_matrix, missing_mask

SECONDARY_QUANTILE_GRID: tuple[float, ...] = (0.1, 0.5, 0.9)
MIN_TRAIN_ROWS = 10


def lgbm_quantile_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    value_column: str,
    *,
    quantile_grid: tuple[float, ...] = SECONDARY_QUANTILE_GRID,
) -> dict[float, pd.Series]:
    numeric_features = (f"{value_column}_lag_24h", f"{value_column}_lag_168h")
    fit_rows = train.dropna(subset=[*numeric_features, value_column])
    if len(fit_rows) < MIN_TRAIN_ROWS:
        return {q: pd.Series(float("nan"), index=test.index) for q in quantile_grid}

    x_train = build_design_matrix(fit_rows, numeric_features=numeric_features)
    y_train = fit_rows[value_column].to_numpy()
    x_test = build_design_matrix(
        test, columns=x_train.columns, numeric_features=numeric_features
    ).fillna(0.0)
    test_missing = missing_mask(test, numeric_features)

    sorted_grid = sorted(quantile_grid)
    raw_preds: list[np.ndarray] = []
    for q in sorted_grid:
        model = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=100, verbose=-1)
        model.fit(x_train, y_train)
        pred = np.asarray(model.predict(x_test)) if len(x_test) > 0 else np.empty(0)
        raw_preds.append(pred)

    # Independently fit per-quantile models can cross; sort per row to restore
    # monotonicity (standard rearrangement fix for quantile crossing).
    stacked = (
        np.sort(np.column_stack(raw_preds), axis=1)
        if len(x_test) > 0
        else np.empty((0, len(sorted_grid)))
    )

    forecasts: dict[float, pd.Series] = {}
    for i, q in enumerate(sorted_grid):
        preds = pd.Series(stacked[:, i], index=test.index)
        preds[test_missing] = float("nan")
        forecasts[q] = preds
    return forecasts


__all__ = ["SECONDARY_QUANTILE_GRID", "lgbm_quantile_forecast"]
