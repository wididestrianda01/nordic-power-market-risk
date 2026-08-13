from datetime import date

import responses

from nordic_power_risk.ingest.entsoe import BASE_URL, fetch_day_ahead_prices, parse_day_ahead_prices

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
    raw = fetch_day_ahead_prices("token", "SE3", date(2020, 1, 1), date(2020, 1, 2))
    assert raw == SAMPLE_XML


def test_parse_day_ahead_prices_advances_hour_and_day():
    rows = parse_day_ahead_prices(SAMPLE_XML)
    assert rows == [
        {"timestamp": "2020-01-01T00:00:00", "price_eur_mwh": 10.5},
        {"timestamp": "2020-01-02T00:00:00", "price_eur_mwh": 20.0},
    ]


def test_parse_day_ahead_prices_skips_non_hourly_resolution():
    rows = parse_day_ahead_prices(SAMPLE_XML)
    assert len(rows) == 2
