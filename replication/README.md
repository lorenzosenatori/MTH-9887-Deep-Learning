# Replication: Attention Factors for Statistical Arbitrage

A re-implementation of Epstein, Wang, Choi and Pelger (2025), "Attention Factors for Statistical
Arbitrage" (ICAIF '25, arXiv:2510.11616), on CRSP and Compustat data from WRDS, with the paper's
sample (daily returns 1990 to 2021, out of sample 1998 to 2021), its universe size (500 stocks),
its 39 characteristics, its model equations, its objective, its two benchmarks and its
rolling-window protocol.

The paper publishes no code. Everything here is written from the paper's text; where the text
leaves something open, the choice made is listed under [Choices the paper leaves open](#choices-the-paper-leaves-open).

**Status: see [DIFFERENCES.md](DIFFERENCES.md) and [README_ISSUES.md](README_ISSUES.md).** The
first holds the latest round (a universe rebuilt to match the firms that the paper names) beside
the paper, and everything that still differs. The second lists the differences from the paper
that were found, how each was handled, and the results of rounds 1 to 3. This file
and [RESULTS.md](RESULTS.md) describe round 1, the first implementation, whose numbers are on the
slides. Two later rounds changed the training loss, three inputs and the construction of the PCA
benchmark, and a fourth the universe; they are run by `scripts/replicate_daily_sharpe.py`,
`scripts/replicate_adjusted_loss.py`, `scripts/replicate_as_cited.py` and
`scripts/replicate_company_universe.py`. Their commands are at the end of `README_ISSUES.md` and
of `DIFFERENCES.md`. The raw-data archive below holds the files of rounds 1 to 3 only.

## What the code covers and what was run

| Paper | Code | Run for `RESULTS.md` |
|---|---|---|
| Table 2: the Attention Factor model | `scripts/run.py attention --k K` | 8, 30 and 100 factors (two seeds for 30, one for 100) |
| Table 2: PCA with the long convolution model | `scripts/run.py pca --k K` | 8 and 30 factors (two seeds for 30) |
| Table 2: PCA with an OU threshold rule | `scripts/run.py ou --k K` | 1, 3, 5, 8, 10, 15, 30 and 100 factors |
| Table 2: the market row | `scripts/build.py check` | yes |
| Table 3: one group of characteristics removed | `scripts/run.py attention --k 30 --drop GROUP` | past returns only (one seed) |
| Figure 2: cumulative returns | `scripts/report.py` | `results/cumulative_returns.png`, with curves after costs |

The Attention Factor model with 100 factors (the last row of that block of Table 2, with the
paper's highest Sharpe ratio before costs, 4.52) and the row of Table 3 without the past returns
were run later, with one seed each, by splitting the test years over several processes
(`--first-year`, `--last-year`); `RESULTS.md`, "Open choices tested", has their results. With
100 factors the Sharpe ratio is 3.04 before costs and 0.33 after, so the statement "weak factors
are important" does not reproduce here.

Not run, for compute (the six first trained runs took 16 to 82 minutes each on the laptop CPU
used here; see `RESULTS.md`, "What did not work"):

- **PCA with the long convolution model with 100 factors.** Of the two benchmarks, only PCA with
  an OU threshold rule is run with 100 factors.
- **The two trained models with any other number of factors** (1, 3, 5 and 15 for both, and 10
  for the benchmark), and **the other groups of Table 3**. The code has the options (`--k`,
  `--drop`).

Also not reproduced:

- Figure 3 (t-SNE of the loadings) and Figure 4 (largest factor weights) by name;
  `diagnostics.json` of each run holds the share of a factor's weight in its ten largest names.

## How to run

```bash
cd replication
conda create -p .venv python=3.12 pip
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

export WRDS_API_TOKEN=...            # or a line WRDS_API_TOKEN=... in ~/.env; never commit it
.venv/bin/python scripts/download.py   # about 20 minutes, 0.7 GB under data/raw (resumable)
.venv/bin/python scripts/build.py      # about 4 minutes: panel, characteristics, inputs, PCA, checks
.venv/bin/python -m pytest tests -q    # 34 tests, a few seconds

.venv/bin/python scripts/run.py attention --k 30     # 24 yearly retrainings; resumable per year
.venv/bin/python scripts/run.py pca --k 30
.venv/bin/python scripts/run.py ou --k 30
.venv/bin/python scripts/report.py     # tables beside the paper's numbers, figure
```

The raw data are CRSP and Compustat data under a WRDS license, so the repository neither holds
nor links them: `scripts/download.py` fetches them with your own WRDS access.

**Without any data.** The tables and the figure can be rebuilt from the repository alone: the
two small result files of every run (`metrics.json`, `daily.csv`: performance numbers and the
daily return of the strategy, no security-level data) are committed under `results/runs/`.

```bash
.venv/bin/python scripts/report.py --from-saved
```

The notebook [notebooks/results.ipynb](notebooks/results.ipynb) walks through the main results
from the same files, with no data and no training; it is saved with its outputs, so it also
reads on GitHub without running.

Seeds are fixed (`--seed`, default 0); a run is deterministic for a fixed seed and thread count.
No GPU is needed: the model has about 4,600 parameters at 30 factors. On a laptop CPU (2 threads)
one yearly retraining takes under a minute for the PCA benchmark and one and a half to three
minutes for the Attention Factor model with 8 to 30 factors, more when several runs share the
CPU.

## Layout

```
requirements.txt        Python packages, pinned to the versions the runs used
notebooks/results.ipynb the main results from the committed result files (no data needed)
scripts/download.py     WRDS data API -> data/raw/*.parquet (resumable)
scripts/build.py        raw files -> data/panel/*.npy, with data checks
scripts/run.py          one configuration over the rolling windows -> runs/<name>/
scripts/report.py       runs -> results/ (tables beside the paper's numbers, figure)
scripts/replicate_*.py  the later rounds of runs (README_ISSUES.md); *_comparison.py write their tables
scripts/check_issues.py checks of the differences from the paper (README_ISSUES.md)
src/attnfactors/
  wrds_api.py           client of the WRDS data API (token from the environment)
  panel.py              dense arrays [day, security]; timing conventions
  chars.py              the 39 characteristics of Table 1, raw
  features.py           rank normalization, the 40 "level" inputs, the rolling PCA
  model.py              attention factor layer, loadings, residuals, LongConv policy, costs, objective
  train.py              rolling estimation, the two benchmarks, evaluation
  pca_regression.py     PCA residuals with loadings from a regression on 60 days (round 3)
tests/                  properties of the model and the data code (no data needed)
results/                tables and figures: aggregate performance only
results/runs/<name>/    metrics.json and daily.csv of each run of round 1 (committed)
results/daily_sharpe/, results/adjusted_loss/, results/as_cited/, results/company/   the same for the later rounds
data/, runs/            licensed data and full run output (git-ignored, local only)
```

## Data handling

- **Source.** WRDS data API: `crsp.msf`, `crsp.dsf`, `crsp.msenames`, `crsp.dsedelist`,
  `crsp.ccmxpf_lnkhist`, `comp.funda`, `ff.factors_daily`, `crsp.dsi`.
- **Licensed data stay out of the repository.** `data/` and `runs/` are git-ignored. Only
  aggregate performance numbers and the daily returns of the strategies are written to
  `results/`. Never commit or publish rows of CRSP or Compustat data. `runs/` holds the full
  output of `scripts/run.py` (portfolio weights per security, models, logs); all of it is
  regenerated by the commands at the end of `RESULTS.md`.
- **Universe.** Each month the 500 largest stocks by the CRSP market capitalization of the
  security at the end of the previous month (2,096 securities over 1988 to 2021).
- **Returns.** CRSP daily returns with delisting returns; a missing return of a member counts as
  zero (0.025 percent of the out-of-sample cells).
- **Timing.** A weight dated t is chosen at the close of day t and earns the return of day t+1.
  A monthly characteristic uses market data to the end of the previous month; accounting data
  follow the Fama-French timing (at least six months old). `tests/test_model.py` checks that a
  weight does not change when later data change; an independent audit repeated this on the
  real panel for the characteristics, the PCA and both model classes.

## Choices the paper leaves open

The paper specifies the equations and Table 4. The following had to be chosen. Each is a place
where this replication may differ from the authors' implementation.

These are the choices of round 1. A later check showed that two of them depart from the formula
that the paper prints (the scale of the Sharpe ratio and the bound on the explained-variance
term), and that for three others a source that the paper cites gives a value: the batch (125
days), the definition of Beta and the construction of the PCA residuals. Round 3 uses the
printed formula and those values (`README_ISSUES.md`). What the authors themselves used for the
batch and for the PCA residuals is still not known.

**Selected on validation data (1996 to 1997, never the 1998 to 2021 test years)**

| Choice | Value | Why it is open |
|---|---|---|
| Ridge penalty of the loadings | 1e-5 | The paper does not report it (Table 4 omits it). |
| Scale of the Sharpe ratio in the objective | annualized (times the square root of 252) | The paper gives the weight 100 on explained variance, not the unit of its Sharpe ratio. Per day, the trading term hardly reaches the factors. |
| Batch | 64 consecutive decision dates per gradient step | Not stated. |

**Data**

| Choice | Value |
|---|---|
| Which securities are "stocks" | Share codes 10, 11, 12, 18, 48, 30, 31 on NYSE, AMEX, Nasdaq: common shares, firms incorporated abroad, real-estate trusts, depositary receipts. No exchange-traded or closed-end funds. The paper names no filter; its Figure 3 shows trusts and foreign firms. |
| Market capitalization for the ranking | Of the security in CRSP, at the end of the previous month. |
| Characteristic definitions | Reconstructed from the standard definitions of the literature the paper cites; `chars.py` marks each. Examples: Variance and Resid_Var on 60 days; Beta as an OLS slope on 252 days; SUV on daily turnover; Vol as dollar volume of five days; r2_1 equal to ST_Rev. |
| Accounting timing | Fama-French: fiscal year ending in calendar year y-1 from July of year y; firm market equity of December y-1. |
| Missing values | Last value (12 months, 5 days), else the cross-sectional median, as the paper says. A characteristic that fewer than half of the universe has is set to the median for all (the bid-ask spread before 1993). |
| Ties in the rank normalization | Average rank. |
| The 40 "level" inputs | Medians of the raw characteristics and the risk-free rate, standardized over the training window. Under the paper's equation they cannot change the factor weights (see below). |
| Delisting returns | Booked on the trading day after the last trade. |

**Model and training**

| Choice | Value |
|---|---|
| How the residual series enters the 32 channels and leaves as one number | Affine lift per channel; GELU; linear read-out. |
| Causality and read-out position of the convolution | Causal; read at the last position (the decision date). |
| Scale of the input | Residuals in percent. |
| How the gross exposure of one is enforced (the constraint is the paper's) | The asset weights of each date, after the allocation is mapped through the residual portfolios, are divided by the sum of their absolute values (`asset_weights` in `model.py`; `PcaData.weights` in `train.py` for the two PCA benchmarks). A date without any allocation keeps weights of zero. |
| Where dropout acts | On the 32 channels after the GELU. |
| Optimizer | AdamW (decoupled weight decay, the reading of the paper's citation), weight decay on the sequence model only. |
| Initialization of W and Q | Gaussian. |
| Explained variance per batch | Securities valid on at least half of the days; one security's term bounded below at -1. |
| Who gets an allocation | A security in the universe on the decision date and the 30 dates before, with a return on each day of the lookback. |
| How the residual history of the signal is built | Each day of the lookback has the residual of its own date's factor model: the residual of day d uses the factor weights and loadings chosen at the close of day d-1. The alternative applies the decision date's residual portfolio to the returns of the last 30 days, so that the signal is the recent history of the portfolio that is traded. The paper's notation fits both. Option `--lookback current` (see [Variants of open choices](#variants-of-open-choices)). |
| Timing of the cost | The return a weight earns is charged with the trade that established it. |
| Sharpe ratio of the tables | Mean over volatility of the daily returns, annualized, no risk-free rate subtracted (as the paper's tables appear to do). |

**Benchmarks**

| Choice | Value |
|---|---|
| PCA | Covariance matrix of the last 252 daily returns; eigenvectors as weights and loadings. The alternative of Avellaneda and Lee, correlation matrix with eigenportfolios and regression loadings, is option `--pca corr`. |
| Securities of the PCA benchmarks | Those in the universe with a return on each of the last 252 days, so a recent listing is not traded by the benchmarks for a year; the Attention Factor model trades it after 30 days. A missing return of a member counts as zero in the Attention Factor model and takes the security out of the PCA. The two trained models therefore trade slightly different sets of securities. |
| OU threshold rule | AR(1) fit to the cumulated residuals of the lookback; allocation -1 above 1.25 standard deviations, +1 below -1.25, 0 otherwise or when the fit has an R-squared below 0.25. The paper says "as in" Guijarro-Ordonez, Pelger and Zanotti; their Appendix B gives the same two constants, 1.25 and 0.25. |

## Variants of open choices

The headline difference from the paper is the ranking of the Attention Factor model and the
PCA benchmark, and it rests on the benchmark's volatility (3.21 percent here, 5.42 in the
paper). Two options rebuild the benchmark the other plausible ways. Their defaults are the
specification of every reported run; a variant run gets its own folder and name
(`pca_K30_corr_seed0`, `pca_K30_look-current_seed0`), stays out of Table 2, and is listed in
`results/variants.md` beside the default run and the paper's row.

| Option | Default | Variant |
|---|---|---|
| `--pca` (benchmarks only) | `cov`: covariance PCA, eigenvectors as weights and loadings | `corr`: correlation PCA of Avellaneda and Lee (2010), factor weights V / sigma, OLS loadings over the same 252 days. On that window the OLS loadings are exactly sigma V (`tests/test_variants.py` checks this), so the residual is R - (sigma V)(V / sigma)^T R. Needs `scripts/build.py pca_corr` once (about 1.7 GB, a few minutes). |
| `--lookback` (all models) | `own`: each day's residual from that day's factor model | `current`: the decision date's residual portfolio applied to the last 30 days of returns. The securities that get an allocation are the same as in the default, so only the signal changes. |

The experiments for the three open questions that a run can answer:

```bash
# 1. Does the benchmark's construction explain the ranking? (about 20 to 35 minutes each)
.venv/bin/python scripts/build.py pca_corr
.venv/bin/python scripts/run.py pca --k 30 --lookback current
.venv/bin/python scripts/run.py pca --k 30 --pca corr
.venv/bin/python scripts/run.py pca --k 30 --pca corr --lookback current
.venv/bin/python scripts/run.py attention --k 30 --lookback current   # the same convention for both models (about an hour)
# 2. Are weak factors important? The last row of the Attention Factor block of Table 2 (several hours)
.venv/bin/python scripts/run.py attention --k 100
# 3. Do past returns drive the result? The first row of Table 3 (about an hour)
.venv/bin/python scripts/run.py attention --k 30 --drop past_returns
.venv/bin/python scripts/report.py
```

## A property of the paper's equation

`tests/test_model.py::test_inputs_equal_for_all_securities_do_not_change_the_factor_weights`
shows that the 40 inputs that are equal for all securities (the cross-sectional medians and the
risk-free rate) cannot change the factor weights of equation (1) of the paper, and get a
gradient that is zero up to rounding: a term that is equal for all securities cancels in a
softmax over the securities. The paper includes them "to keep the level information". Either the authors'
implementation differs from the equation, or that information is not used.

## Tests

`python -m pytest tests -q` (34 tests): the residual projection (factor portfolios of the residuals are
zero without a ridge penalty, not zero with one), the identity between the residual portfolio
and the asset weights, no look-ahead, the cost of a worked example, the Squash operator and the
last-position convolution, the bounded explained-variance term, average ranks for ties, batches
that cover every date, the PCA mask, and the local check of API filters. `tests/test_variants.py`
covers the variant options: the correlation PCA's loadings equal the OLS regression loadings, the
traded benchmark portfolio has no factor exposure in both PCA modes, the "current" lookback
equals a hand computation and agrees with the default when the factor model does not change,
and it has no look-ahead. For the later rounds:
the whole-window loss and its gradient (round 2), the explained-variance term without its bound,
the PCA residuals with regression loadings, and the cited definition of Beta (round 3); the
history rule of the universe and the rule for days without a trade (round 4).

## AI usage

See [AI_USAGE.md](AI_USAGE.md).
