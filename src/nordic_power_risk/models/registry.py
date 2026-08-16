"""MLflow Model Registry helpers for the promotion gate (Phase 5).

The day-ahead ladder is the single gated forecast tier. Each ``models`` run
registers its selected best rung as a versioned model under ``DAY_AHEAD_MODEL``,
and the ``champion`` alias always points at the current best. Promotion owns the
alias; registration and alias management are the seam the gate and rollback both
consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.pyfunc import log_model
from mlflow.pyfunc.model import PythonModel
from mlflow.tracking import MlflowClient

CHAMPION_ALIAS = "champion"
DAY_AHEAD_MODEL = "day-ahead-ladder"


class RegistryError(RuntimeError):
    """The Model Registry is missing or inconsistent."""


@dataclass(frozen=True)
class Champion:
    version: str
    run_id: str
    metrics: dict[str, float]


class _ForecastModel(PythonModel):
    """Minimal pyfunc wrapper carrying the champion's forecast records.

    The registry tracks a real logged model so the ``champion`` alias has an
    artifact to point at; the forecast records are the model's output, while the
    gate reads the run's metrics rather than re-loading the artifact.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def predict(
        self, context: Any, model_input: Any, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self.records


def register_forecast(
    model_name: str, run_id: str, rung: str, records: list[dict[str, Any]]
) -> str:
    """Log the winning rung's forecast as a pyfunc model and register the run as a version.

    Returns the new version string. The ``champion`` alias is not set here — promotion
    owns the alias; this only appends a challenger candidate to the version lineage.
    """
    with mlflow.start_run(run_id=run_id):
        info = log_model(
            artifact_path="forecast_model",
            python_model=_ForecastModel(records),
            registered_model_name=model_name,
            await_registration_for=60,
        )
    version = info.registered_model_version
    if version is None:
        raise RegistryError(f"registration of '{model_name}' returned no version")
    return str(version)


def set_champion(model_name: str, version: str) -> None:
    MlflowClient().set_registered_model_alias(model_name, CHAMPION_ALIAS, version)


def get_champion(model_name: str) -> Champion | None:
    """Return the current champion (version + run_id + metrics), or ``None`` if unset."""
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except mlflow.MlflowException:
        return None
    if version.run_id is None:
        raise RegistryError(f"champion version {version.version} has no run_id")
    metrics = {k: float(v) for k, v in client.get_run(version.run_id).data.metrics.items()}
    return Champion(version=version.version, run_id=version.run_id, metrics=metrics)


def latest_version(model_name: str) -> ModelVersion:
    """Return the newest registered version, or raise if none exists."""
    versions = MlflowClient().search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RegistryError(f"model '{model_name}' has no registered versions")
    return max(versions, key=lambda v: int(v.version))


def rollback_champion(model_name: str, tracking_uri: str) -> str:
    """Re-alias the champion to the previous registered version.

    Returns the new champion version. Raises if there is no champion or no prior
    version to roll back to.
    """
    mlflow.set_tracking_uri(tracking_uri)
    champion = get_champion(model_name)
    if champion is None:
        raise RegistryError(f"model '{model_name}' has no champion to roll back")
    versions = MlflowClient().search_model_versions(f"name='{model_name}'")
    prior = max(
        (v for v in versions if int(v.version) < int(champion.version)),
        key=lambda v: int(v.version),
        default=None,
    )
    if prior is None:
        raise RegistryError(f"model '{model_name}' has no prior version to roll back to")
    set_champion(model_name, prior.version)
    return prior.version


__all__ = [
    "CHAMPION_ALIAS",
    "DAY_AHEAD_MODEL",
    "Champion",
    "RegistryError",
    "get_champion",
    "latest_version",
    "register_forecast",
    "rollback_champion",
    "set_champion",
]
