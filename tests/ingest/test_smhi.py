import re

import responses

from nordic_power_risk.ingest.smhi import BASE_URL, fetch_observations, parse_observations

# Corrected-archive CSV: semicolon-delimited, header lines before the data.
SAMPLE_CSV = (
    b"Stationsnamn;Stationsnummer\n"
    b"Stockholm-Arlanda Flygplats;97400\n"
    b"\n"
    b"Datum;Tid (UTC);Lufttemperatur;Kvalitet\n"
    b"2008-02-01;00:00:00;1.4;G\n"
    b"2008-02-01;01:00:00;0.8;Y\n"
)


@responses.activate
def test_fetch_observations_calls_corrected_archive_csv():
    responses.add(
        responses.GET, re.compile(rf"{re.escape(BASE_URL)}/.*"), body=SAMPLE_CSV, status=200
    )
    raw = fetch_observations(1, 97400)
    assert raw == SAMPLE_CSV
    assert "/period/corrected-archive/data.csv" in responses.calls[0].request.url


def test_parse_observations_extracts_epoch_ms_and_float_value():
    rows = parse_observations(SAMPLE_CSV)
    assert rows == [
        {"timestamp": 1201824000000, "value": 1.4},
        {"timestamp": 1201827600000, "value": 0.8},
    ]


def test_parse_observations_skips_header_and_empty_input():
    assert parse_observations(b"") == []
