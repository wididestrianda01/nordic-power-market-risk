"""Secrets (.env) and pipeline parameters (config.yaml) as typed settings."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Secrets. Never put non-secret parameters here — those belong in config.yaml."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    entsoe_api_token: str | None = None


class Window(BaseModel):
    start: date
    end: date


class DispatchConfig(BaseModel):
    horizon_days: int = Field(default=1, ge=1, le=7)
    power_limit_mw: float = Field(default=1.0, gt=0)
    energy_capacity_mwh: float = Field(default=2.0, gt=0)
    one_way_efficiency: float = Field(default=0.9487, gt=0, le=1)
    initial_soc_mwh: float = Field(default=1.0, ge=0)
    terminal_value_eur_mwh: float = Field(default=0.0, ge=0)
    degradation_cost_eur_mwh: float = Field(default=15.0, ge=15.0, le=40.0)

    @model_validator(mode="after")
    def validate_initial_soc(self) -> DispatchConfig:
        if self.initial_soc_mwh > self.energy_capacity_mwh:
            raise ValueError("initial_soc_mwh cannot exceed energy_capacity_mwh")
        return self

class PipelineConfig(BaseModel):
    zone: str
    windows: dict[str, Window]
    duckdb_path: Path
    manifest_path: Path
    mlflow_tracking_uri: str = "sqlite:///data/mlflow.db"
    mlflow_experiment: str = "day-ahead-ladder"
    optimizer: DispatchConfig = Field(default_factory=DispatchConfig)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_config(path: Path = REPO_ROOT / "config.yaml") -> PipelineConfig:
    raw = yaml.safe_load(path.read_text())
    return PipelineConfig.model_validate(raw)
