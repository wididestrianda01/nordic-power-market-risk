import json
import re

import responses

from nordic_power_risk.ingest.smhi import BASE_URL, fetch_observations, parse_observations

SAMPLE_JSON = json.dumps({"value": [{"date": 1577836800000, "value": "-3.2"}]}).encode()


@responses.activate
def test_fetch_observations_calls_smhi_api():
    responses.add(
        responses.GET, re.compile(rf"{re.escape(BASE_URL)}/.*"), body=SAMPLE_JSON, status=200
    )
    raw = fetch_observations(1, 97270)
    assert raw == SAMPLE_JSON


def test_parse_observations_extracts_timestamp_and_float_value():
    rows = parse_observations(SAMPLE_JSON)
    assert rows == [{"timestamp": 1577836800000, "value": -3.2}]


def test_parse_observations_handles_missing_value_key():
    rows = parse_observations(b"{}")
    assert rows == []
