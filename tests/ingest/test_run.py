import json
from datetime import date
from pathlib import Path

import pytest
import responses

from nordic_power_risk.config import PipelineConfig, Settings, Window
from nordic_power_risk.ingest.entsoe import ACTIVATION_PROCESS_TYPES
from nordic_power_risk.ingest.entsoe import BASE_URL as ENTSOE_URL
from nordic_power_risk.ingest.esett import BASE_URL as ESETT_URL
from nordic_power_risk.ingest.run import ingest_all
from nordic_power_risk.ingest.smhi import BASE_URL as SMHI_URL
from nordic_power_risk.ingest.svk import BASE_URL as SVK_URL

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
ACTIVATION_PRICE_XML = b"""<?xml version="1.0"?>
<Balancing_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:1">
  <TimeSeries>
    <businessType>A96</businessType>
    <flowDirection.direction>A01</flowDirection.direction>
    <Period>
      <timeInterval><start>2020-01-01T00:00Z</start></timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><activation_Price.amount>70.0</activation_Price.amount></Point>
    </Period>
  </TimeSeries>
</Balancing_MarketDocument>
"""
PROCURED_VOLUME_XML = b"""<?xml version="1.0"?>
<Balancing_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:4">
  <TimeSeries>
    <flowDirection.direction>A01</flowDirection.direction>
    <Period>
      <timeInterval><start>2020-01-01T00:00Z</start></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>2.0</quantity></Point>
    </Period>
  </TimeSeries>
</Balancing_MarketDocument>
"""
IMBALANCE_VOLUME_XML = b"""<?xml version="1.0"?>
<Balancing_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:balancingdocument:3:0">
  <TimeSeries>
    <businessType>A19</businessType>
    <flowDirection.direction>A01</flowDirection.direction>
    <Period>
      <timeInterval><start>2020-01-01T00:00Z</start></timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><quantity>5.0</quantity></Point>
    </Period>
  </TimeSeries>
</Balancing_MarketDocument>
"""



def _zip_xml(xml: bytes) -> bytes:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("file.xml", xml)
    return buffer.getvalue()


PROCURED_VOLUME_ZIP = _zip_xml(PROCURED_VOLUME_XML)
IMBALANCE_VOLUME_ZIP = _zip_xml(IMBALANCE_VOLUME_XML)
ESETT_JSON = json.dumps(
    [{"timestampUTC": "2020-01-01T00:00:00Z", "imblPurchasePrice": 1.0}]
).encode()
SVK_JSON = json.dumps(
    {"result": {"records": [{"start_time_utc": "2020-01-01T00:00:00", "value": 1.0}]}}
).encode()
SMHI_CSV = b"Datum;Tid (UTC);Lufttemperatur;Kvalitet\n2020-01-01;00:00:00;1.0;G\n"


@pytest.fixture
def config(tmp_path) -> PipelineConfig:
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2020, 1, 1), end=date(2020, 1, 2))},
        duckdb_path=Path(tmp_path / "nordic_power_risk.duckdb"),
        manifest_path=Path(tmp_path / "manifest.json"),
    )


def test_ingest_all_raises_without_entsoe_token(config):
    settings = Settings(_env_file=None)
    with pytest.raises(RuntimeError, match="ENTSOE_API_TOKEN"):
        ingest_all(config, settings)


@responses.activate
def test_ingest_all_writes_tables_and_manifest(config):
    responses.add(responses.GET, ENTSOE_URL, body=ENTSOE_XML, status=200)
    responses.add(responses.GET, ENTSOE_URL, body=ACTIVATION_PRICE_XML, status=200)
    for _ in ACTIVATION_PROCESS_TYPES:
        responses.add(responses.GET, ENTSOE_URL, body=PROCURED_VOLUME_ZIP, status=200)
    responses.add(responses.GET, ENTSOE_URL, body=IMBALANCE_VOLUME_ZIP, status=200)
    responses.add(responses.GET, ESETT_URL, body=ESETT_JSON, status=200)
    responses.add(responses.GET, SVK_URL, body=SVK_JSON, status=200)
    responses.add(
        responses.GET,
        __import__("re").compile(rf"{SMHI_URL}/.*"),
        body=SMHI_CSV,
        status=200,
    )
    settings = Settings(_env_file=None, entsoe_api_token="token")

    entries = ingest_all(config, settings)

    names = {e.name for e in entries}
    assert names == {
        "entsoe_day_ahead_price",
        "esett_imbalance_price",
        "svk_fcr_capacity",
        "svk_afrr_mfrr_capacity",
        "smhi_observations",
        "entsoe_activation_price",
        "entsoe_reserve_volume",
        "entsoe_imbalance_volume",
        "svk_activated_energy",
    }
    assert config.manifest_path.exists()
    assert config.duckdb_path.exists()
