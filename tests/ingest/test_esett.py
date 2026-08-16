import json
from datetime import date

import responses

from nordic_power_risk.ingest.esett import BASE_URL, fetch_imbalance_prices, parse_imbalance_prices

# Real EXP14/Prices response shape: timestampUTC (naive-UTC after stripping "Z")
# and imblPurchasePrice (single-price regime: imblPurchasePrice == imblSalesPrice).
SAMPLE_JSON = json.dumps(
    [
        {"timestampUTC": "2026-01-01T00:00:00Z", "imblPurchasePrice": 5.5},
        {"timestampUTC": "2026-01-01T00:15:00Z", "imblPurchasePrice": -2.0},
        {"timestampUTC": "2026-01-01T00:30:00Z", "imblPurchasePrice": -1.0},
    ]
).encode()


@responses.activate
def test_fetch_imbalance_prices_uses_eic_and_utc_datetimes():
    responses.add(responses.GET, BASE_URL, body=SAMPLE_JSON, status=200)
    raw = fetch_imbalance_prices("SE3", date(2026, 1, 1), date(2026, 1, 2))
    assert raw == [SAMPLE_JSON]
    request = responses.calls[0].request
    assert request.params["mba"] == "10Y1001A1001A46L"
    assert request.params["start"] == "2026-01-01T00:00:00.000Z"
    assert request.params["end"] == "2026-01-02T00:00:00.000Z"


def test_parse_imbalance_prices_extracts_timestamp_and_price():
    rows = parse_imbalance_prices(SAMPLE_JSON)
    assert rows == [
        {"timestamp": "2026-01-01T00:00:00", "imbalance_price_eur_mwh": 5.5},
        {"timestamp": "2026-01-01T00:15:00", "imbalance_price_eur_mwh": -2.0},
    ]


def test_parse_imbalance_prices_drops_missing_price_sentinel():
    rows = parse_imbalance_prices(SAMPLE_JSON)
    assert all(r["imbalance_price_eur_mwh"] != -1.0 for r in rows)
    assert len(rows) == 2  # the -1.0 sentinel row is dropped
