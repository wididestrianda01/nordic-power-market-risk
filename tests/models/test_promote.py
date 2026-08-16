"""Promotion gate tests: bootstrap, promote-on-significance, retain-on-failure/equal."""

import mlflow
import pytest

from nordic_power_risk.models.promote import promote_champion
from nordic_power_risk.models.registry import (
    RegistryError,
    get_champion,
    register_forecast,
    rollback_champion,
)

MODEL = "day-ahead-ladder"


def _uri(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def _register(tmp_path, rung: str, pinball: float, dm_stat: float, dm_pvalue: float) -> None:
    mlflow.set_tracking_uri(_uri(tmp_path))
    mlflow.set_experiment("test-ladder")
    with mlflow.start_run(run_name=rung) as run:
        mlflow.log_metrics(
            {
                "pinball_loss": pinball,
                "dm_stat_vs_seasonal_naive": dm_stat,
                "dm_pvalue_vs_seasonal_naive": dm_pvalue,
            }
        )
        run_id = run.info.run_id
    register_forecast(MODEL, run_id, rung, [{"event_time": "2020-01-01", "q0_5": 1.0}])


def test_get_champion_is_none_before_any_promotion(tmp_path) -> None:
    mlflow.set_tracking_uri(_uri(tmp_path))
    assert get_champion(MODEL) is None


def test_first_promotion_bootstraps(tmp_path) -> None:
    _register(tmp_path, "lear", 0.3, -3.0, 0.01)
    result = promote_champion(MODEL, _uri(tmp_path))
    assert result.promoted is True
    assert result.reason == "first champion"
    champion = get_champion(MODEL)
    assert champion is not None
    assert champion.metrics["pinball_loss"] == 0.3


def test_promote_on_significant_improvement(tmp_path) -> None:
    _register(tmp_path, "lear", 0.3, -3.0, 0.01)
    promote_champion(MODEL, _uri(tmp_path))
    _register(tmp_path, "dnn", 0.1, -4.0, 0.005)
    result = promote_champion(MODEL, _uri(tmp_path))
    assert result.promoted is True
    assert result.reason == "significant improvement"
    assert get_champion(MODEL).metrics["pinball_loss"] == 0.1


def test_retain_on_worse_pinball(tmp_path) -> None:
    _register(tmp_path, "lear", 0.3, -3.0, 0.01)
    promote_champion(MODEL, _uri(tmp_path))
    _register(tmp_path, "dnn", 0.5, -4.0, 0.005)  # significant but worse pinball
    result = promote_champion(MODEL, _uri(tmp_path))
    assert result.promoted is False
    assert result.reason == "no pinball improvement"
    assert get_champion(MODEL).metrics["pinball_loss"] == 0.3


def test_retain_on_not_significant(tmp_path) -> None:
    _register(tmp_path, "lear", 0.3, -3.0, 0.01)
    promote_champion(MODEL, _uri(tmp_path))
    _register(tmp_path, "dnn", 0.1, 1.0, 0.9)  # better pinball but DM not significant
    result = promote_champion(MODEL, _uri(tmp_path))
    assert result.promoted is False
    assert result.reason == "not significant vs seasonal-naive"
    assert get_champion(MODEL).metrics["pinball_loss"] == 0.3


def test_retain_on_equal_pinball(tmp_path) -> None:
    _register(tmp_path, "lear", 0.3, -3.0, 0.01)
    promote_champion(MODEL, _uri(tmp_path))
    _register(tmp_path, "dnn", 0.3, -4.0, 0.005)  # equal pinball, significant
    result = promote_champion(MODEL, _uri(tmp_path))
    assert result.promoted is False
    assert result.reason == "no pinball improvement"


def test_rollback_reverts_to_prior_champion(tmp_path) -> None:
    _register(tmp_path, "lear", 0.3, -3.0, 0.01)
    promote_champion(MODEL, _uri(tmp_path))
    first = get_champion(MODEL).version

    _register(tmp_path, "dnn", 0.1, -4.0, 0.005)
    promote_champion(MODEL, _uri(tmp_path))
    assert get_champion(MODEL).version != first

    rolled_back = rollback_champion(MODEL, _uri(tmp_path))
    assert rolled_back == first
    assert get_champion(MODEL).version == first


def test_rollback_without_prior_version_raises(tmp_path) -> None:
    _register(tmp_path, "lear", 0.3, -3.0, 0.01)
    promote_champion(MODEL, _uri(tmp_path))
    with pytest.raises(RegistryError):
        rollback_champion(MODEL, _uri(tmp_path))
