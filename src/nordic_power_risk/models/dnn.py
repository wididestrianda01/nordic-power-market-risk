"""DNN point forecaster (Phase 2 ticket 03), epftoolbox-style shallow feedforward net.

Trained with a pinball-loss objective at the median (q=0.5) rather than MSE, so the
"pinball-loss objective" from the ticket is literally the training loss — but the point
forecast still plugs into the ladder's existing residual-quantile machinery
(models.baselines) for the rest of the quantile grid, same as LEAR, so there's no
separate multi-quantile output head to maintain.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch import nn

from nordic_power_risk.models.design_matrix import (
    MIN_TRAIN_ROWS,
    NUMERIC_FEATURES,
    build_design_matrix,
    missing_mask,
)

HIDDEN_SIZES = (32, 16)
EPOCHS = 200
LEARNING_RATE = 0.01
MEDIAN_QUANTILE = 0.5


def _pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, quantile: float) -> torch.Tensor:
    diff = y_true - y_pred
    return torch.mean(torch.maximum(quantile * diff, (quantile - 1) * diff))


def _build_net(in_features: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_features
    for hidden in HIDDEN_SIZES:
        layers += [nn.Linear(prev, hidden), nn.ReLU()]
        prev = hidden
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def dnn_forecast(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Fit a small feedforward net on train, return point forecasts for train and test."""
    fit_rows = train.dropna(subset=[*NUMERIC_FEATURES, "price_eur_mwh"])
    if len(fit_rows) < MIN_TRAIN_ROWS:
        return (
            pd.Series(float("nan"), index=train.index),
            pd.Series(float("nan"), index=test.index),
        )

    x_train = build_design_matrix(fit_rows)
    x_mean = x_train.mean()
    x_std = x_train.std().replace(0.0, 1.0)
    x_train_scaled = (x_train - x_mean) / x_std

    x_tensor = torch.tensor(x_train_scaled.to_numpy(), dtype=torch.float32)
    y_tensor = torch.tensor(fit_rows["price_eur_mwh"].to_numpy(), dtype=torch.float32).unsqueeze(1)

    torch.manual_seed(0)
    net = _build_net(x_tensor.shape[1])
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    net.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = _pinball_loss(y_tensor, net(x_tensor), MEDIAN_QUANTILE)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    x_train_full = build_design_matrix(train, columns=x_train.columns).fillna(0.0)
    x_test_full = build_design_matrix(test, columns=x_train.columns).fillna(0.0)
    x_train_full_scaled = (x_train_full - x_mean) / x_std
    x_test_full_scaled = (x_test_full - x_mean) / x_std

    net.eval()
    with torch.no_grad():
        train_preds = net(torch.tensor(x_train_full_scaled.to_numpy(), dtype=torch.float32))
        train_point = pd.Series(train_preds.squeeze(1).numpy(), index=train.index)
        if len(x_test_full_scaled) > 0:
            test_preds = net(torch.tensor(x_test_full_scaled.to_numpy(), dtype=torch.float32))
            test_point = pd.Series(test_preds.squeeze(1).numpy(), index=test.index)
        else:
            test_point = pd.Series(dtype=float, index=test.index)

    train_missing = missing_mask(train, NUMERIC_FEATURES)
    test_missing = missing_mask(test, NUMERIC_FEATURES)
    train_point[train_missing] = float("nan")
    test_point[test_missing] = float("nan")

    return train_point, test_point


__all__ = ["dnn_forecast"]
