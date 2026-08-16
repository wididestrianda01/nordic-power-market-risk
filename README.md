# nordic-power-market-risk

A decision and risk system for a small battery in the Swedish SE3 bidding zone. It
allocates energy and reserve capacity across day-ahead, imbalance, and reserve markets
(FCR/aFRR/mFRR), then settles every paper position against observed prices.

This is a historical decision demonstrator. It places no live orders.

## Headline result

On the 11-month walk-forward window (Apr 2025 to Feb 2026), the optimized policy earns
EUR 483,956 net of declared costs, against EUR 0 for no-trade and EUR 86,516 for the
heuristic benchmark. The reserve-capacity allocation (FCR/aFRR/mFRR) supplies the bulk of
the margin; the perfect-foresight bound of EUR 635,813 is a ceiling, not a target.

![Cumulative paper P&L net of costs](docs/figures/hero_cumulative_pnl.png)

The figure renders from the walk-forward outputs through `nordic-risk figures`.

## What it does

The pipeline runs nine stages, each a separate command, each landing its output in DuckDB:

```
ingest → validate → features → models → optimize → risk → settle → monitor → promote
```

- **ingest** pulls public prices, reserves, and weather, and records a manifest per source.
- **validate** applies Pandera schemas so a malformed row fails loudly, never silently.
- **features** builds lagged and calendar features under an issue-time cutoff that forbids
  look-ahead.
- **models** fits a probabilistic forecast ladder (naive, seasonal-naive, LEAR, DNN) and
  promotes the best on a frozen holdout.
- **optimize** solves a rolling mixed-integer linear program (MILP) that dispatches energy
  and reserve capacity.
- **risk** gates each decision on CVaR and drawdown tail-risk limits.
- **settle** reconciles every paper position against observed prices.
- **monitor** reports drift and missingness from the live-feature path.
- **promote** promotes the champion model only when it clears the promotion test.

## How a decision is made

One day ahead, the loop runs four steps:

1. **Forecast.** LEAR (a sparse linear autoregressive model) regresses the day-ahead price
   on lags and calendar features at a grid of quantiles (0.1 through 0.9 plus the tails),
   producing a predictive distribution rather than a point estimate.
2. **Dispatch.** A MILP allocates the 1 MW / 2 MWh battery across energy and reserve
   capacity. A binary variable enforces that the battery charges or discharges, never both
   at once. Reserve capacity must fit inside the state-of-charge headroom, so the energy
   dispatch and the reserve offer are co-optimized, not stacked.
3. **Gate.** A CVaR and drawdown gate blocks any decision that breaches tail-risk limits; a
   breach returns a flat decision and a three-day cooldown.
4. **Settle.** Every position settles at the observed price: day-ahead energy at the
   day-ahead price, imbalance at the single imbalance price, and conditionally accepted
   reserve capacity at the observed reserve price.

## Results, beyond the headline

The gap between the optimized policy and perfect foresight is EUR 151,857. Attribution
splits it into named causes: forecast error EUR 62,590, constraint cost EUR 88,775, and
degradation EUR 492. Constraint cost is the dominant drag, because reserve capacity ties up
power that could otherwise arbitrage energy.

Reserve capacity is the margin driver. Settled at observed MFRR and AFRR prices, reserve
capacity earns EUR 484,735: MFRR up EUR 235,675, MFRR down EUR 133,602, AFRR down
EUR 111,161, and AFRR up EUR 4,280. FCR contributes near zero, so the battery's value here
is in restoration reserves (aFRR and mFRR), not containment. Reserve activation costs
EUR 1,076, and energy is near-neutral: day-ahead revenue EUR 928 against purchase EUR 309,
degradation EUR 492, and imbalance EUR 171.

## Evaluation

The forecast ladder promoted LEAR over naive on the frozen holdout: average pinball loss
6.66 against 7.67 (Diebold-Mariano p < 0.0001, n = 7,985). The test suite holds 92 percent
line coverage, and the settlement reconciles paper P&L to observed prices with a residual
of zero.

## The thesis bridge

The single foregrounded analysis measures whether the 15-minute transition raised the
difficulty of forecasting the imbalance price. The design is an interrupted time series on
SE3's own series, before and after 1 October 2025. A minimum-detectable-effect calculation
runs first; the effect was detected. Imbalance-forecast pinball rose from 15.36 in the
pre-period to 19.99 in the post-period, an increase of 4.63 against a minimum detectable
effect of 0.28.

## Real data, real provenance

The data are public and real. Day-ahead prices come from ENTSO-E Transparency Platform and
Svenska kraftnät. Imbalance prices come from eSett. Weather comes from SMHI. Every source
carries a manifest with its licence, coverage window, endpoint, pull timestamp, and
checksum.

## Failure modes

The default on error is flat, never a guess. An infeasible or missing-input solve, a
tail-risk breach, or a drawdown breach returns a no-trade decision and, on a breach, a
three-day cooldown. No stale forecast is ever substituted.

## What this is and is not

This is a paper-trading backtest on observed public data. It places no live orders. It
simulates the control obligations of revised REMIT (EU 2024/1106) Article 5a rather than
discharging them. The report states this plainly.

## What I'd change

Add intraday continuous trading, source a rejected-bid feed so reserve value is not
conditional on acceptance, and extend the post-transition window as more 15-minute history
accumulates.

## Read more

- [Technical report](docs/report.md): the full methodology, theory, regulation, and
  decision log.
- [Results notebook](notebooks/results.ipynb): reproduces the headline figures.

## Stack

Python 3.13, uv, typer, DuckDB, Pyomo + HiGHS, LightGBM, scikit-learn, MLflow, Pandera,
Evidently, matplotlib. The code demonstrates forecasting, MILP dispatch, tail-risk gating,
and settlement.

## Setup

```bash
uv sync
cp .env.example .env   # add your ENTSO-E API token
uv run nordic-risk --help
uv run nordic-risk ingest
```

## Development

```bash
uv run ruff check .    # lint
uv run mypy            # strict type checking
uv run pytest          # tests, coverage gated at 80%
docker build -t nordic-risk .   # multi-stage build, slim runtime image
```

CI (GitHub Actions) runs lint, typecheck, test, and a Docker build on every push and pull
request.
