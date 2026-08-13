import json
from datetime import date
from pathlib import Path

import pytest
import responses

from p16.config import PipelineConfig, Settings, Window
from p16.ingest.entsoe import BASE_URL as ENTSOE_URL
from p16.ingest.esett import BASE_URL as ESETT_URL
from p16.ingest.run import ingest_all
from p16.ingest.smhi import BASE_URL as SMHI_URL
from p16.ingest.svk import BASE_URL as SVK_URL

ENTSOE_XML = b"""<?xml version="1.0"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
  <TimeSeries>
    <Period>
      <timeInterval><start>2020-01-01T00:00Z</start></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><price.amount>10.5</price.amount></Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>
"""
ESETT_JSON = json.dumps([{"timestamp": "2020-01-01T00:00:00Z", "value": 1.0}]).encode()
SVK_JSON = json.dumps({"result": {"records": [{"value": 1.0}]}}).encode()
SMHI_JSON = json.dumps({"value": [{"date": 1577836800000, "value": "1.0"}]}).encode()


@pytest.fixture
def config(tmp_path) -> PipelineConfig:
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2020, 1, 1), end=date(2020, 1, 2))},
        duckdb_path=Path(tmp_path / "p16.duckdb"),
        manifest_path=Path(tmp_path / "manifest.json"),
    )


def test_ingest_all_raises_without_entsoe_token(config):
    settings = Settings(_env_file=None)
    with pytest.raises(RuntimeError, match="ENTSOE_API_TOKEN"):
        ingest_all(config, settings)


@responses.activate
def test_ingest_all_writes_tables_and_manifest(config):
    responses.add(responses.GET, ENTSOE_URL, body=ENTSOE_XML, status=200)
    responses.add(responses.GET, ESETT_URL, body=ESETT_JSON, status=200)
    responses.add(responses.GET, SVK_URL, body=SVK_JSON, status=200)
    responses.add(
        responses.GET,
        __import__("re").compile(rf"{SMHI_URL}/.*"),
        body=SMHI_JSON,
        status=200,
    )
    settings = Settings(_env_file=None, entsoe_api_token="token")

    entries = ingest_all(config, settings)

    names = {e.name for e in entries}
    assert names == {
        "entsoe_day_ahead_price",
        "esett_imbalance_price",
        "svk_day_ahead_price",
        "svk_fcr_capacity",
        "svk_afrr_mfrr_capacity",
        "smhi_observations",
    }
    assert config.manifest_path.exists()
    assert config.duckdb_path.exists()
