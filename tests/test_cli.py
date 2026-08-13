from typer.testing import CliRunner

from nordic_power_risk.cli import app

runner = CliRunner()


def test_validate_stub_exits_nonzero():
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 1
    assert "not implemented" in result.output


def test_ingest_without_token_exits_nonzero(monkeypatch):
    monkeypatch.delenv("ENTSOE_API_TOKEN", raising=False)
    from nordic_power_risk import config as config_module

    config_module.get_settings.cache_clear()
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 1
    assert "ENTSOE_API_TOKEN" in result.output
