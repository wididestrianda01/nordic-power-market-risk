from datetime import date
from pathlib import Path

from nordic_power_risk.config import PipelineConfig, Settings, get_config


def test_settings_reads_missing_token_as_none(monkeypatch):
    monkeypatch.delenv("ENTSOE_API_TOKEN", raising=False)
    settings = Settings(_env_file=None)
    assert settings.entsoe_api_token is None


def test_settings_reads_token_from_env(monkeypatch):
    monkeypatch.setenv("ENTSOE_API_TOKEN", "secret-token")
    settings = Settings(_env_file=None)
    assert settings.entsoe_api_token == "secret-token"


def test_get_config_parses_repo_config_yaml():
    config = get_config()
    assert config.zone == "SE3"
    assert config.windows["primary"].start == date(2019, 1, 1)
    assert config.windows["primary"].end == date(2026, 6, 30)
    assert config.windows["secondary"].start == date(2025, 10, 1)


def test_pipeline_config_validates_from_dict():
    config = PipelineConfig.model_validate(
        {
            "zone": "SE3",
            "windows": {"primary": {"start": "2020-01-01", "end": "2020-12-31"}},
            "duckdb_path": "data/nordic_power_risk.duckdb",
            "manifest_path": "data/manifest.json",
        }
    )
    assert config.duckdb_path == Path("data/nordic_power_risk.duckdb")
