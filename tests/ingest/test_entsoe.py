from datetime import date

import responses

from nordic_power_risk.ingest.entsoe import (
    BASE_URL,
    chunk_date_range,
    fetch_day_ahead_prices,
    parse_activated_energy_prices,
    parse_day_ahead_prices,
    parse_imbalance_volumes,
    parse_procured_volume,
)

SAMPLE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
  <TimeSeries>
    <Period>
      <timeInterval><start>2020-01-01T00:00Z</start></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><price.amount>10.5</price.amount></Point>
      <Point><position>25</position><price.amount>20.0</price.amount></Point>
    </Period>
  </TimeSeries>
  <TimeSeries>
    <Period>
      <timeInterval><start>2020-01-01T00:00Z</start></timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><price.amount>99.0</price.amount></Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>
"""


@responses.activate
def test_fetch_day_ahead_prices_calls_entsoe_api():
    responses.add(responses.GET, BASE_URL, body=SAMPLE_XML, status=200)
    raw_chunks = fetch_day_ahead_prices("token", "SE3", date(2020, 1, 1), date(2020, 1, 2))
    assert raw_chunks == [SAMPLE_XML]


@responses.activate
def test_fetch_day_ahead_prices_issues_one_request_per_chunk():
    responses.add(responses.GET, BASE_URL, body=SAMPLE_XML, status=200)
    raw_chunks = fetch_day_ahead_prices("token", "SE3", date(2019, 1, 1), date(2021, 6, 1))
    assert len(raw_chunks) == 3
    assert len(responses.calls) == 3


def test_chunk_date_range_splits_multi_year_span_into_year_chunks():
    chunks = chunk_date_range(date(2019, 1, 1), date(2021, 6, 1))
    assert chunks == [
        (date(2019, 1, 1), date(2020, 1, 1)),
        (date(2020, 1, 1), date(2020, 12, 31)),
        (date(2020, 12, 31), date(2021, 6, 1)),
    ]


def test_chunk_date_range_single_chunk_for_short_span():
    chunks = chunk_date_range(date(2020, 1, 1), date(2020, 1, 2))
    assert chunks == [(date(2020, 1, 1), date(2020, 1, 2))]


def test_parse_day_ahead_prices_advances_hour_and_day():
    rows = parse_day_ahead_prices(SAMPLE_XML)
    assert rows == [
        {"timestamp": "2020-01-01T00:00:00", "price_eur_mwh": 10.5},
        {"timestamp": "2020-01-02T00:00:00", "price_eur_mwh": 20.0},
    ]


def test_parse_day_ahead_prices_skips_non_hourly_resolution():
    rows = parse_day_ahead_prices(SAMPLE_XML)
    assert len(rows) == 2


ACTIVATION_PRICE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
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
  <TimeSeries>
    <businessType>A97</businessType>
    <flowDirection.direction>A02</flowDirection.direction>
    <Period>
      <timeInterval><start>2020-01-01T00:00Z</start></timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>2</position><activation_Price.amount>55.0</activation_Price.amount></Point>
    </Period>
  </TimeSeries>
</Balancing_MarketDocument>
"""

PROCURED_VOLUME_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<Balancing_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:4">
  <TimeSeries>
    <flowDirection.direction>A01</flowDirection.direction>
    <Period>
      <timeInterval><start>2020-01-01T00:00Z</start></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>2.0</quantity></Point>
      <Point><position>2</position><quantity>3.0</quantity></Point>
    </Period>
  </TimeSeries>
</Balancing_MarketDocument>
"""


def _zip_xml(xml: bytes, name: str = "file.xml") -> bytes:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, xml)
    return buffer.getvalue()


def test_parse_activated_energy_prices_maps_business_type_and_direction():
    rows = parse_activated_energy_prices(ACTIVATION_PRICE_XML)
    assert rows == [
        {
            "timestamp": "2020-01-01T00:00:00",
            "product": "AFRR",
            "direction": "up",
            "activation_price_eur_mwh": 70.0,
        },
        {
            "timestamp": "2020-01-01T00:15:00",
            "product": "MFRR",
            "direction": "down",
            "activation_price_eur_mwh": 55.0,
        },
    ]


def test_parse_procured_volume_reads_zip_and_quantity():
    rows = parse_procured_volume(_zip_xml(PROCURED_VOLUME_XML), "AFRR")
    assert rows == [
        {
            "timestamp": "2020-01-01T00:00:00",
            "product": "AFRR",
            "direction": "up",
            "procured_mw": 2.0,
        },
        {
            "timestamp": "2020-01-01T01:00:00",
            "product": "AFRR",
            "direction": "up",
            "procured_mw": 3.0,
        },
    ]

IMBALANCE_VOLUME_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<Balancing_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:balancingdocument:3:0">
  <TimeSeries>
    <businessType>A19</businessType>
    <flowDirection.direction>A01</flowDirection.direction>
    <Period>
      <timeInterval><start>2020-01-01T00:00Z</start></timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><quantity>1.0</quantity></Point>
      <Point><position>2</position><quantity>2.0</quantity></Point>
      <Point><position>3</position><quantity>3.0</quantity></Point>
      <Point><position>4</position><quantity>4.0</quantity></Point>
    </Period>
  </TimeSeries>
  <TimeSeries>
    <businessType>A19</businessType>
    <flowDirection.direction>A02</flowDirection.direction>
    <Period>
      <timeInterval><start>2020-01-01T01:00Z</start></timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><quantity>5.0</quantity></Point>
      <Point><position>2</position><quantity>6.0</quantity></Point>
    </Period>
  </TimeSeries>
</Balancing_MarketDocument>
"""


def test_parse_imbalance_volumes_aggregates_15m_to_hourly_mfrr():
    rows = parse_imbalance_volumes(_zip_xml(IMBALANCE_VOLUME_XML))
    assert rows == [
        {
            "timestamp": "2020-01-01T00:00:00",
            "product": "MFRR",
            "direction": "up",
            "activated_mw": 10.0,
        },
        {
            "timestamp": "2020-01-01T01:00:00",
            "product": "MFRR",
            "direction": "down",
            "activated_mw": 11.0,
        },
    ]
