# nordic-power-market-risk

A decision and risk system for the Nordic day-ahead, imbalance, and reserve
power markets (SE3 bidding zone). Pulls public market and weather data,
lands it in DuckDB with a full source manifest, and builds toward forecasting,
MILP dispatch optimization, risk-gated position sizing, and settlement.

## Status

Phase 0 (repo scaffold and ingestion) is complete: `p16 ingest` pulls
ENTSO-E day-ahead prices, eSett imbalance prices, SvK capacity/price series,
and SMHI weather observations into DuckDB, with a licence/checksum/coverage
manifest per source. Forecasting, optimization, risk, and settlement stages
are scaffolded as CLI subcommands and implemented in later phases.

## Stack

Python 3.13, [uv](https://docs.astral.sh/uv/) for dependency management,
[typer](https://typer.tiangolo.com/) for the CLI, [DuckDB](https://duckdb.org/)
for storage, [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
for config, [pandera](https://pandera.readthedocs.io/) for schema validation.

## Setup

```bash
uv sync
cp .env.example .env   # add your ENTSO-E API token
uv run p16 --help
uv run p16 ingest
```

## Development

```bash
uv run ruff check .    # lint
uv run mypy             # strict type checking
uv run pytest           # tests, coverage gated at 80%
docker build -t p16 .   # multi-stage build, slim runtime image
```

CI (GitHub Actions) runs lint, typecheck, test, and a Docker build on every
push and pull request.

## Architecture

- **CLI** (`src/p16/cli.py`) — one typer command per pipeline stage
  (`ingest`, `validate`, `features`, `models`, `optimize`, `risk`, `settle`,
  `monitor`, `promote`). Each command delegates to its own package; no
  business logic lives in the CLI layer.
- **Config** (`src/p16/config.py`) — secrets (API tokens, via `.env`) and
  frozen domain parameters (bidding zone, date windows, storage paths, via
  `config.yaml`) are kept as two separate, cached accessors.
- **Ingest** (`src/p16/ingest/`) — one client module per data source
  (ENTSO-E, eSett, SvK, SMHI), each exposing a `fetch_*` (HTTP) and `parse_*`
  (pure) function. An orchestrator pulls all four sources into DuckDB and
  writes a manifest documenting licence, coverage window, endpoint, and
  checksum for every table written.

## Data sources

| Source | Series | Licence |
|---|---|---|
| ENTSO-E Transparency Platform | Day-ahead prices | Platform terms (no bulk redistribution) |
| eSett Open Data | Imbalance prices | Public, no formal open licence |
| Svenska kraftnät (SvK) | Day-ahead price, FCR/aFRR/mFRR capacity | CC BY 4.0 |
| SMHI Open Data | Weather observations | CC BY 4.0 |
