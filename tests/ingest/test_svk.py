import json

import responses

from nordic_power_risk.ingest.svk import (
    ACTIVATION_RESOURCE_IDS,
    BASE_URL,
    RESOURCE_IDS,
    fetch_activated_energy,
    fetch_resource,
    parse_activated_energy,
    parse_resource,
)

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


ACTIVATION_JSON = json.dumps(
    {
        "result": {
            "records": [
                {
                    "start_time_utc": "2024-01-01T00:00:00",
                    "bidding_zone": "SE3",
                    "reserve_product": "aFRRCapacityMarket",
                    "reserve_direction": "up",
                    "volume": 3.5,
                },
                {
                    "start_time_utc": "2024-01-01T00:00:00",
                    "bidding_zone": "SE3",
                    "reserve_product": "FCRN",
                    "reserve_direction": "down",
                    "volume": -2.0,
                },
                {
                    "start_time_utc": "2024-01-01T00:00:00",
                    "bidding_zone": "SE4",
                    "reserve_product": "aFRRCapacityMarket",
                    "reserve_direction": "up",
                    "volume": 9.9,
                },
                {
                    "start_time_utc": "2024-01-01T00:00:00",
                    "bidding_zone": "SE3",
                    "reserve_product": "FCRD",
                    "reserve_direction": "up",
                    "volume": 1.0,
                },
            ]
        }
    }
).encode()


@responses.activate
def test_fetch_activated_energy_calls_ckan():
    responses.add(responses.GET, BASE_URL, body=ACTIVATION_JSON, status=200)
    assert fetch_activated_energy("afrr") == ACTIVATION_JSON


def test_parse_activated_energy_filters_zone_and_maps_products():
    rows = parse_activated_energy(ACTIVATION_JSON, "SE3")
    assert rows == [
        {
            "timestamp": "2024-01-01T00:00:00",
            "product": "AFRR",
            "direction": "up",
            "activated_mw": 3.5,
        },
        {
            "timestamp": "2024-01-01T00:00:00",
            "product": "FCR_N",
            "direction": "down",
            "activated_mw": 2.0,
        },
        {
            "timestamp": "2024-01-01T00:00:00",
            "product": "FCR_D",
            "direction": "up",
            "activated_mw": 1.0,
        },
    ]


def test_activation_resource_ids_cover_expected_series():
    assert set(ACTIVATION_RESOURCE_IDS) == {"afrr", "fcr"}


def test_resource_ids_fcr_and_afrr_not_swapped():
    # fcr_capacity's CKAN resource serves FCRD/FCRN records; afrr_mfrr_capacity's
    # serves aFRRCapacityMarket records. Regression guard for a T03 mapping bug.
    assert RESOURCE_IDS["fcr_capacity"] == "72ef5ec0-d0d7-4d22-95e9-4f22b3048af4"
    assert RESOURCE_IDS["afrr_mfrr_capacity"] == "6351d2cc-1657-43eb-b112-b8408c700529"
