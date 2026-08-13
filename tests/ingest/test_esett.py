import json
from datetime import date

import responses

from nordic_power_risk.ingest.esett import BASE_URL, fetch_imbalance_prices, parse_imbalance_prices

SAMPLE_JSON = json.dumps(
    [
        {"timestamp": "2020-01-01T00:00:00Z", "value": 5.5},
        {"timestamp": "2020-01-01T00:15:00Z", "value": -2.0},
    ]
).encode()


@responses.activate
def test_fetch_imbalance_prices_calls_esett_api():
    responses.add(responses.GET, BASE_URL, body=SAMPLE_JSON, status=200)
    raw = fetch_imbalance_prices("SE3", date(2020, 1, 1), date(2020, 1, 2))
    assert raw == SAMPLE_JSON


def test_parse_imbalance_prices_extracts_timestamp_and_value():
    rows = parse_imbalance_prices(SAMPLE_JSON)
    assert rows == [
        {"timestamp": "2020-01-01T00:00:00Z", "imbalance_price_eur_mwh": 5.5},
        {"timestamp": "2020-01-01T00:15:00Z", "imbalance_price_eur_mwh": -2.0},
    ]
