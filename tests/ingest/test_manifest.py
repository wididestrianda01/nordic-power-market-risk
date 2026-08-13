import json
from datetime import date

from nordic_power_risk.ingest.manifest import checksum_of, make_entry, write_manifest


def test_checksum_of_is_stable_sha256():
    raw = b"hello"
    assert checksum_of(raw) == checksum_of(b"hello")
    assert len(checksum_of(raw)) == 64


def test_make_entry_computes_checksum_and_row_count():
    entry = make_entry(
        name="test_source",
        licence="CC BY 4.0",
        coverage_start=date(2020, 1, 1),
        coverage_end=date(2020, 1, 2),
        endpoint="https://example.test",
        raw=b"raw-bytes",
        row_count=42,
    )
    assert entry.name == "test_source"
    assert entry.row_count == 42
    assert entry.checksum == checksum_of(b"raw-bytes")


def test_write_manifest_writes_json_list(tmp_path):
    entry = make_entry(
        name="test_source",
        licence="CC BY 4.0",
        coverage_start=date(2020, 1, 1),
        coverage_end=date(2020, 1, 2),
        endpoint="https://example.test",
        raw=b"raw-bytes",
        row_count=1,
    )
    manifest_path = tmp_path / "nested" / "manifest.json"
    write_manifest([entry], manifest_path)

    payload = json.loads(manifest_path.read_text())
    assert len(payload) == 1
    assert payload[0]["name"] == "test_source"
