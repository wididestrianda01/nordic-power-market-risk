"""Secrets (.env) and pipeline parameters (config.yaml) as typed settings."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Secrets. Never put non-secret parameters here — those belong in config.yaml."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    entsoe_api_token: str | None = None


class Window(BaseModel):
    start: date
    end: date


class PipelineConfig(BaseModel):
    zone: str
    windows: dict[str, Window]
    duckdb_path: Path
    manifest_path: Path
    mlflow_tracking_uri: str = "sqlite:///data/mlflow.db"
    mlflow_experiment: str = "day-ahead-ladder"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_config(path: Path = REPO_ROOT / "config.yaml") -> PipelineConfig:
    raw = yaml.safe_load(path.read_text())
    return PipelineConfig.model_validate(raw)
