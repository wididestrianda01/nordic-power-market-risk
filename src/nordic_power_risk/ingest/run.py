"""Orchestrate all four source pulls into DuckDB + write the source manifest."""

from __future__ import annotations

from datetime import UTC, date, datetime, time

from nordic_power_risk.config import PipelineConfig, Settings
from nordic_power_risk.ingest import entsoe, esett, smhi, svk
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.ingest.manifest import ManifestEntry, make_entry, write_manifest

# SMHI station representative of SE3: Stockholm-Arlanda Flygplats (97400),
# air temperature (parameter 1). 97270 is decommissioned (Strängnäs, 1980-1990).
SMHI_PARAMETER = 1
SMHI_STATION = 97400


def _epoch_ms(day: date) -> int:
    """Naive-UTC midnight for `day` as epoch milliseconds."""
    return int(datetime.combine(day, time.min, tzinfo=UTC).timestamp() * 1000)


def _in_window_date(timestamp: str, start: date, end: date) -> bool:
    """True if an ISO timestamp's calendar date falls in [start, end] (inclusive)."""
    return start.isoformat() <= timestamp[:10] <= end.isoformat()


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
        raw_chunks = entsoe.fetch_day_ahead_prices(
            settings.entsoe_api_token, config.zone, start, end
        )
        rows = [row for raw in raw_chunks for row in entsoe.parse_day_ahead_prices(raw)]
        rows = [r for r in rows if _in_window_date(r["timestamp"], start, end)]
        row_count = write_table(conn, "raw_entsoe_day_ahead_price", rows)
        entries.append(
            make_entry(
                name="entsoe_day_ahead_price",
                licence="ENTSO-E Transparency Platform terms (no bulk redistribution)",
                coverage_start=start,
                coverage_end=end,
                endpoint=entsoe.BASE_URL,
                raw=b"".join(raw_chunks),
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
            rows = [r for r in rows if _in_window_date(r["start_time_utc"], start, end)]
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
        lo_ms = _epoch_ms(start)
        hi_ms = _epoch_ms(end) + 86_400_000  # end date inclusive
        rows = [r for r in rows if lo_ms <= r["timestamp"] < hi_ms]
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

        # Activated balancing energy prices (A84): fetched in month chunks; the
        # product is read back from each TimeSeries' businessType.
        raw_chunks = entsoe.fetch_activated_energy_prices(
            settings.entsoe_api_token, config.zone, start, end
        )
        activation_price_rows = [
            row for raw in raw_chunks for row in entsoe.parse_activated_energy_prices(raw)
        ]
        row_count = write_table(conn, "raw_activation_price", activation_price_rows)
        entries.append(
            make_entry(
                name="entsoe_activation_price",
                licence="ENTSO-E Transparency Platform terms (no bulk redistribution)",
                coverage_start=start,
                coverage_end=end,
                endpoint=entsoe.BASE_URL,
                raw=b"".join(raw_chunks),
                row_count=row_count,
            )
        )

        procured_rows = []
        procured_raw = b""
        for product, process_type in entsoe.ACTIVATION_PROCESS_TYPES.items():
            raw_chunks = entsoe.fetch_procured_volume(
                settings.entsoe_api_token, config.zone, process_type, start, end
            )
            procured_raw += b"".join(raw_chunks)
            for raw in raw_chunks:
                procured_rows.extend(entsoe.parse_procured_volume(raw, product))
        row_count = write_table(conn, "raw_reserve_volume", procured_rows)
        entries.append(
            make_entry(
                name="entsoe_reserve_volume",
                licence="ENTSO-E Transparency Platform terms (no bulk redistribution)",
                coverage_start=start,
                coverage_end=end,
                endpoint=entsoe.BASE_URL,
                raw=procured_raw,
                row_count=row_count,
            )
        )
        # Activated balancing energy VOLUMES: SvK 60-min aFRR/FCR-N/FCR-D
        # (discontinued 2025-03-04) plus ENTSO-E A86 imbalance volume, which IS
        # the mFRR activation volume and covers SE3 continuously (15-min -> hourly).
        activation_rows = []
        activation_raw = b""
        for series in svk.ACTIVATION_RESOURCE_IDS:
            raw = svk.fetch_activated_energy(series)
            activation_raw += raw
            activation_rows.extend(svk.parse_activated_energy(raw, config.zone))
        imbalance_chunks = entsoe.fetch_imbalance_volumes(
            settings.entsoe_api_token, config.zone, start, end
        )
        for raw in imbalance_chunks:
            activation_rows.extend(entsoe.parse_imbalance_volumes(raw))
        activation_columns = {
            "timestamp": "VARCHAR",
            "product": "VARCHAR",
            "direction": "VARCHAR",
            "activated_mw": "DOUBLE",
        }
        row_count = write_table(conn, "raw_activation", activation_rows, columns=activation_columns)
        entries.append(
            make_entry(
                name="svk_activated_energy",
                licence="CC BY 4.0",
                coverage_start=start,
                coverage_end=end,
                endpoint=svk.BASE_URL,
                raw=activation_raw,
                row_count=row_count,
            )
        )
        entries.append(
            make_entry(
                name="entsoe_imbalance_volume",
                licence="ENTSO-E Transparency Platform terms (no bulk redistribution)",
                coverage_start=start,
                coverage_end=end,
                endpoint=entsoe.BASE_URL,
                raw=b"".join(imbalance_chunks),
                row_count=row_count,
            )
        )
    finally:
        conn.close()

    write_manifest(entries, config.manifest_path)
    return entries
