import json

import responses

from nordic_power_risk.ingest.svk import BASE_URL, RESOURCE_IDS, fetch_resource, parse_resource

SAMPLE_JSON = json.dumps(
    {"result": {"records": [{"timestamp": "2020-01-01T00:00:00", "value": 12.3}]}}
).encode()


@responses.activate
def test_fetch_resource_calls_ckan_datastore_search():
    responses.add(responses.GET, BASE_URL, body=SAMPLE_JSON, status=200)
    raw = fetch_resource("day_ahead_price")
    assert raw == SAMPLE_JSON


def test_fetch_resource_unknown_series_raises_key_error():
    try:
        fetch_resource("not_a_series")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown series")


def test_parse_resource_extracts_records():
    rows = parse_resource(SAMPLE_JSON)
    assert rows == [{"timestamp": "2020-01-01T00:00:00", "value": 12.3}]


def test_resource_ids_cover_expected_series():
    assert set(RESOURCE_IDS) == {"day_ahead_price", "fcr_capacity", "afrr_mfrr_capacity"}
