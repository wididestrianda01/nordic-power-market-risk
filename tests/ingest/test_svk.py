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


def test_resource_ids_fcr_and_afrr_not_swapped():
    # fcr_capacity's CKAN resource serves FCRD/FCRN records; afrr_mfrr_capacity's
    # serves aFRRCapacityMarket records. Regression guard for a T03 mapping bug.
    assert RESOURCE_IDS["fcr_capacity"] == "72ef5ec0-d0d7-4d22-95e9-4f22b3048af4"
    assert RESOURCE_IDS["afrr_mfrr_capacity"] == "6351d2cc-1657-43eb-b112-b8408c700529"
