"""Orchestrate all four source pulls into DuckDB + write the source manifest."""

from __future__ import annotations

from nordic_power_risk.config import PipelineConfig, Settings
from nordic_power_risk.ingest import entsoe, esett, smhi, svk
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.ingest.manifest import ManifestEntry, make_entry, write_manifest

# SMHI station representative of SE3 (Stockholm-Arlanda), air temperature (parameter 1).
SMHI_PARAMETER = 1
SMHI_STATION = 97270


def ingest_all(config: PipelineConfig, settings: Settings) -> list[ManifestEntry]:
    if not settings.entsoe_api_token:
        raise RuntimeError(
            "ENTSOE_API_TOKEN is not set. Copy .env.example to .env and add your token."
        )

    start = config.windows["primary"].start
    end = config.windows["primary"].end
    conn = get_connection(config.duckdb_path)
    entries: list[ManifestEntry] = []

    try:
        raw = entsoe.fetch_day_ahead_prices(settings.entsoe_api_token, config.zone, start, end)
        rows = entsoe.parse_day_ahead_prices(raw)
        row_count = write_table(conn, "raw_entsoe_day_ahead_price", rows)
        entries.append(
            make_entry(
                name="entsoe_day_ahead_price",
                licence="ENTSO-E Transparency Platform terms (no bulk redistribution)",
                coverage_start=start,
                coverage_end=end,
                endpoint=entsoe.BASE_URL,
                raw=raw,
                row_count=row_count,
            )
        )

        raw = esett.fetch_imbalance_prices(config.zone, start, end)
        rows = esett.parse_imbalance_prices(raw)
        row_count = write_table(conn, "raw_esett_imbalance_price", rows)
        entries.append(
            make_entry(
                name="esett_imbalance_price",
                licence="eSett Open Data terms (public, no formal open licence)",
                coverage_start=start,
                coverage_end=end,
                endpoint=esett.BASE_URL,
                raw=raw,
                row_count=row_count,
            )
        )

        for series in svk.RESOURCE_IDS:
            raw = svk.fetch_resource(series)
            rows = svk.parse_resource(raw)
            row_count = write_table(conn, f"raw_svk_{series}", rows)
            entries.append(
                make_entry(
                    name=f"svk_{series}",
                    licence="CC BY 4.0",
                    coverage_start=start,
                    coverage_end=end,
                    endpoint=svk.BASE_URL,
                    raw=raw,
                    row_count=row_count,
                )
            )

        raw = smhi.fetch_observations(SMHI_PARAMETER, SMHI_STATION)
        rows = smhi.parse_observations(raw)
        row_count = write_table(conn, "raw_smhi_observations", rows)
        entries.append(
            make_entry(
                name="smhi_observations",
                licence="SMHI Open Data (CC BY 4.0)",
                coverage_start=start,
                coverage_end=end,
                endpoint=smhi.BASE_URL,
                raw=raw,
                row_count=row_count,
            )
        )
    finally:
        conn.close()

    write_manifest(entries, config.manifest_path)
    return entries
