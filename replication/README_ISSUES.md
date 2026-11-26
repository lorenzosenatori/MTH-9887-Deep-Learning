# Replication: the issues found, how each was handled, and the current results

Status of 23 November 2026. This file is the map of the replication after its rounds of runs.
`README.md` says how to run the code, `RESULTS.md` holds the write-up of round 1 (the numbers on
the slides), and `results/as_cited/comparison.md` holds every table of rounds 1 to 3.

**A fourth round followed this file.** It rebuilds the universe (issue 5) and reruns a subset of
the paper's Table 2 on it. Its results and the list of what still differs from the paper are in
[DIFFERENCES.md](DIFFERENCES.md). The sections below describe rounds 1 to 3; where round 4
changes a statement, the text says so.

The paper is Epstein, Wang, Choi and Pelger (2025), "Attention Factors for Statistical
Arbitrage". It publishes no code. "After costs" always means after the paper's cost of 5 basis
points per unit traded and 1 basis point per unit held short. All Sharpe ratios in the tables
are annualized and out of sample, January 1998 to December 2021, unless a table says
"validation" (1996 to 1997).

## In short

- **No round reproduces the level of the paper** (Sharpe ratios of 3.97 before costs and 2.28
  after, with 30 factors). Round 1, with an annualized objective, reaches 3.00 and 0.99.
  `RESULTS.md` and the slides call this "the level reproduces"; that holds at most before costs.
  With the Sharpe ratio per day, as the paper prints it (rounds 2a and 3), the model reaches
  about 2.1 before costs and 0.1 after.
- **The numbers on the slides (3.00 and 0.99) rest on a departure from the printed objective.**
  Round 1 annualized the Sharpe ratio inside the objective. That makes the trading term 16 times
  stronger against the explained-variance term. It was disclosed as an open choice; it is a
  departure (issue 1).
- **The ranking against the PCA benchmark depends on how the benchmark is built.** Built as in
  round 1, the benchmark has the higher Sharpe ratio. Built as the predecessor's public code
  builds it, the benchmark with 30 factors falls to 1.73 before costs and -1.26 after, and the
  model is above it after costs. With 8 factors the two are level. Before costs the paper's
  benchmark (2.79) lies between our two constructions (3.47 and 1.73); after costs (1.57) it is
  above both (1.26 and -1.26). Neither can be shown to be the paper's (issue 6).
- **The other differences on the list are real and small.** The bound on the explained-variance
  term, the length of a batch and three inputs were all changed in round 3. It lands where round
  2a lands (issues 2 to 4).
- **One run of round 2 cannot answer its question.** The run with the whole training window as
  one batch takes 30 optimizer steps a year instead of about 950. Its result measures
  undertraining (issue 7).
- **The universe was the largest difference still open after round 3. Round 4 moved it toward
  the paper's, and the results did not follow.** The paper's figures name 108 firms among its
  500 on the first trading day of 2021. In that month 40 of them were not in the universe of
  rounds 1 to 3, and 29 of those were in it in no month of 2021. Round 4 ranks a firm by the
  value of the whole company, admits partnership units and asks for 120 months of returns. Then
  91 of the 108 are members, and with the same settings and the same seed the Sharpe ratios of
  the trained models move by at most 0.15. The paper's own universe is still not known, so this
  does not rule the universe out (issue 5, `DIFFERENCES.md`).
- **In every round most of the profit after costs is earned before 2003.** With the printed
  objective all of it is (rounds 2a, 3 and 4: the returns after costs from 2003 to 2021 sum to
  a loss). With the annualized objective it is 85 percent with 8 factors and all of it with 30
  in round 1, and 61 and 93 percent in round 4.

## The three rounds of runs

| Round | When | What it is | Script | Results |
|---|---|---|---|---|
| 1 | 21 to 22 Nov | The first implementation. Open choices were set on the validation years. One of them: the Sharpe ratio inside the training objective is annualized. These are the numbers on the slides. | `scripts/run.py` | `results/runs/`, `RESULTS.md` |
| 2a | 23 Nov | The Attention Factor model trained again with the Sharpe ratio per day, as the paper prints its objective. Nothing else changed. | `scripts/replicate_daily_sharpe.py` | `results/daily_sharpe/` |
| 2b | 23 Nov | As 2a, with the objective computed over the whole training window in one batch and without the lower bound on the explained-variance term. **Not a valid test**, see issue 7. | `scripts/replicate_adjusted_loss.py` | `results/adjusted_loss/` |
| 3 | 23 Nov | "As cited": where a source that the paper cites settles a choice, the choice is taken from that source. It changes the loss, three inputs and the construction of the PCA benchmark together (issues 1 to 4 and 6). | `scripts/replicate_as_cited.py` | `results/as_cited/` |
| 4 | 23 Nov | Round 3 on a universe rebuilt to hold the firms that the paper's figures name (issue 5): ranked by the value of the whole company, with partnership units, and with a history of 120 months. 8 and 30 factors, both readings of the objective and of the PCA residuals. | `scripts/replicate_company_universe.py` | `results/company/`, `DIFFERENCES.md` |

No round overwrites the files of an earlier one. Rounds 1 and 3 were written and run with Claude
Code, round 2 with OpenAI Codex (see `AI_USAGE.md`).

## Current results

Averages over the seeds of a model. Turnover is the sum of the absolute changes of the weights
per day, at a gross exposure of one.

<!-- results:start -->
**30 factors** (two seeds for the trained models)

| | Sharpe ratio | after costs | Mean % | after costs | Volatility % | Turnover a day |
|---|---|---|---|---|---|---|
| *Attention Factor model* | | | | | | |
| **Paper, Table 2** | **3.97** | **2.28** | **16.66** | **9.52** | **4.20** | |
| Round 1: annualized Sharpe (slides) | 3.00 | 0.99 | 12.81 | 4.20 | 4.26 | 0.59 |
| Round 2a: Sharpe per day | 2.10 | 0.09 | 10.54 | 0.42 | 5.02 | 0.70 |
| Round 2b: whole window, 30 steps (not a valid test) | 1.47 | -1.73 | 6.54 | -7.65 | 4.43 | 1.03 |
| Round 3: as cited | 2.05 | 0.09 | 10.12 | 0.43 | 4.95 | 0.67 |
| *PCA with the long convolution model* | | | | | | |
| **Paper, Table 2** | **2.79** | **1.57** | **15.15** | **8.47** | **5.42** | |
| Round 1: projection on the eigenvectors | 3.47 | 1.26 | 11.12 | 4.02 | 3.21 | 0.47 |
| Round 3: loadings from a regression on 60 days | 1.73 | -1.26 | 4.41 | -3.20 | 2.55 | 0.50 |
| *PCA with an OU threshold rule* | | | | | | |
| **Paper, Table 2** | **0.18** | **-6.45** | **0.40** | **-14.74** | **2.24** | |
| Round 1: projection on the eigenvectors | 1.31 | -3.95 | 3.33 | -10.01 | 2.53 | 0.96 |
| Round 3: loadings from a regression on 60 days | 0.90 | -5.61 | 1.80 | -11.18 | 1.99 | 0.93 |

**8 factors** (one seed for the trained models)

| | Sharpe ratio | after costs | Mean % | after costs | Volatility % | Turnover a day |
|---|---|---|---|---|---|---|
| *Attention Factor model* | | | | | | |
| **Paper, Table 2** | **3.35** | **1.94** | **15.70** | **9.05** | **4.68** | |
| Round 1: annualized Sharpe (slides) | 2.66 | 1.15 | 14.38 | 6.21 | 5.41 | 0.56 |
| Round 2a: Sharpe per day | 2.10 | 0.42 | 12.13 | 2.45 | 5.78 | 0.67 |
| Round 2b: whole window, 30 steps (not a valid test) | 1.47 | -0.90 | 8.18 | -5.02 | 5.55 | 0.95 |
| Round 3: as cited | 2.06 | 0.53 | 11.65 | 3.01 | 5.67 | 0.59 |
| *PCA with the long convolution model* | | | | | | |
| **Paper, Table 2** | **2.64** | **1.50** | **14.88** | **8.42** | **5.63** | |
| Round 1: projection on the eigenvectors | 3.07 | 1.35 | 13.87 | 6.10 | 4.52 | 0.53 |
| Round 3: loadings from a regression on 60 days | 2.29 | 0.47 | 9.45 | 1.93 | 4.12 | 0.50 |
| *PCA with an OU threshold rule* | | | | | | |
| **Paper, Table 2** | **0.78** | **-4.15** | **2.04** | **-10.83** | **2.61** | |
| Round 1: projection on the eigenvectors | 1.16 | -3.16 | 3.61 | -9.83 | 3.11 | 0.97 |
| Round 3: loadings from a regression on 60 days | 1.38 | -3.29 | 3.92 | -9.32 | 2.83 | 0.95 |
<!-- results:end -->

How to read the table:

- **Rounds 2a and 3 agree for the Attention Factor model**: 2.10 and 2.05 before costs, 0.09 and
  0.09 after, with 30 factors. The changes that round 3 adds to round 2a (batches of 125 days,
  no bound, three inputs as cited) do not move the result out of sample.
- **The gap to the paper** with the printed objective: about half of its Sharpe ratio before
  costs (2.05 of 3.97) and almost none of it after costs (0.09 of 2.28). The mean return is 10.1
  percent a year against 16.7, and the costs take 9.7 points of it.
- **The benchmark built the predecessor's way gets worse with more factors** (2.29 with 8, 1.73
  with 30). The paper's benchmark does not (2.64 and 2.79). Its volatility (5.42 percent with 30
  factors) is reproduced by neither construction (3.21 and 2.55).
- **The OU rule with 30 factors moves toward the paper's row** in round 3 (0.90 and -5.61; paper
  0.18 and -6.45).

The model against the benchmark, as a difference of Sharpe ratios (model minus benchmark). It is
computed on the daily returns averaged over the seeds. With 30 factors the two Sharpe ratios
behind a cell are therefore a little larger in size than those of the table (2.13 for 2.05, and
-1.34 for -1.26 for the round-3 benchmark after costs), so a cell is not the difference of two
table entries; with 8 factors there is one seed and they are the same. In parentheses: the
difference in units of its standard error, from a block bootstrap with blocks of 1, 21 and 63
days (`results/as_cited/comparison.md`).

| | Before costs | After costs |
|---|---|---|
| **30 factors.** Paper, Table 2 | +1.18 | +0.71 |
| Round 1 model, benchmark by projection | -0.38 (1.2 to 1.8) | -0.25 (1.0 to 1.3) |
| Round 3 model, benchmark by projection | -1.36 (4.5 to 6.1) | -1.18 (4.8 to 5.7) |
| Round 3 model, benchmark with loadings from 60 days | +0.29 (1.0 to 1.3) | +1.43 (5.5 to 6.5) |
| Round 1 model, benchmark with loadings from 60 days | +1.27 (4.2 to 5.3) | +2.36 (8.6 to 10.3) |
| **8 factors.** Paper, Table 2 | +0.71 | +0.44 |
| Round 1 model, benchmark by projection | -0.41 (1.8 to 2.1) | -0.20 (1.0 to 1.1) |
| Round 3 model, benchmark by projection | -1.01 (4.2 to 5.0) | -0.82 (3.9 to 4.4) |
| Round 3 model, benchmark with loadings from 60 days | -0.24 (1.0 to 1.2) | +0.06 (0.3) |
| Round 1 model, benchmark with loadings from 60 days | +0.37 (1.5 to 1.8) | +0.68 (3.2 to 3.6) |

So with the Sharpe ratio per day (round 3) the sign of the paper's comparison appears in one
cell beyond doubt: 30 factors, after costs, against the benchmark built the predecessor's way.
It appears there because that benchmark loses 3.2 percent a year after costs, not because the
model earns much (0.4 percent a year). The round-1 model against the same benchmark has the
paper's sign in all four cells, three of them beyond three standard errors. That pairing
combines the annualized objective (issue 1) with a benchmark that cannot be shown to be the
paper's (issue 6).

Round 3 by period, Sharpe ratio after costs: the Attention Factor model with 30 factors has 2.34
and 2.54 (two seeds) in 1998 to 2002, and -0.93 and -0.84 in 2003 to 2021. With 8 factors: 2.50
and -0.25.

## The issues, one by one

Issues 1 to 6 are differences between round 1 and the paper or the sources it cites. Each was
checked against the code. Where a cited source applies, the statement about it rests on
quotations gathered by AI research agents; nobody in the group has compared them with the
originals yet (`AI_USAGE.md`). Issue 7 is a mistake made while resolving them. The checks are in `scripts/check_issues.py`; their output is under
`results/as_cited/checks/`.

| # | Issue | Real? | Addressed by | Effect on validation (Sharpe ratio after costs) | Status |
|---|---|---|---|---|---|
| 1 | The Sharpe ratio in the objective was annualized | Yes | Round 2a | 7.50 annualized, 5.17 per day | Measured. Which scale the authors used is not known. |
| 2 | One security's explained-variance term was bounded below at -1 | Yes | Round 3 (no bound) | 5.17 to 5.39 at batches of 64 days | Measured, small |
| 3 | The objective is computed on batches of 64 days | Yes, as a choice | Round 3 (batches of 125 days, the predecessor's) | 5.17 to 5.56 | Measured, small. The whole window as one batch was not tested validly. |
| 4 | Three inputs were not built as the cited source defines them | Yes for Beta; small and ambiguous for two | Round 3 | 5.17 to 5.02 | Measured, small. One further input stays open. |
| 5 | The universe is not the paper's | Yes, and larger than recorded | Round 4, with an inferred rule | Not run on validation. Out of sample the trained models move by at most 0.15 | 91 of the 108 named firms are members after round 4 (68 before), and the results move little. The paper's rule is still not known. See `DIFFERENCES.md`. |
| 6 | The PCA benchmark is not built as the predecessor builds it | Yes | Round 3 | Benchmark: 6.00 to 1.27 | Measured, decisive for the ranking. The paper's own construction stays unknown. |
| 7 | Round 2b took 30 optimizer steps a year | Yes, our mistake | Shown by a check; superseded by round 3 | -0.98 after 30 steps, with and without the bound (the two runs are identical) | Its result must not be read as a result about the loss |

### 1. The Sharpe ratio in the objective was annualized

**The paper.** The objective is the Sharpe ratio after costs plus 100 times the explained
variance (Section 3.3, Table 4). The formula carries no annualization factor.

**Round 1.** The Sharpe ratio was multiplied by the square root of 252. This was a deliberate
choice on validation data (7.50 against 5.17 after costs) and was listed in the README as a
choice the paper leaves open. It was not an open choice: it departs from the printed formula,
and the headline numbers of the slides rest on it.

**Why it matters.** The trading policy does not see the explained-variance term, and the
optimizer (Adam) is insensitive to the scale of a gradient, so the scale changes one thing: how
the factor weights trade off the two terms. Per day, a Sharpe ratio of at most about 0.3 stands
against 100 times an explained variance of 0.3 to 0.45. The factors are then trained almost only to
explain variance. Annualized, the trading term is 16 times stronger. Annualizing is the same as
a weight of 6.3 instead of 100 on the explained variance.

**What a cited source does.** The public code of the predecessor paper (Guijarro-Ordonez, Pelger
and Zanotti, `dlsa-public`, `train_test.py`) computes `loss = -mean_ret/std` on the returns of
one batch: per day, not annualized.

**Resolution.** Round 2a refitted the three runs with the Sharpe ratio per day. The script and
the saved settings were checked: only that setting differs from round 1. The Sharpe ratio after
costs falls from 0.99 to 0.09 with 30 factors and from 1.15 to 0.42 with 8.

**What stays open.** The paper says the weight on the explained variance is "selected optimally
on the validation data" and that the term "empirically improves the performance". In this
implementation the weight of 100 with the Sharpe ratio per day does worse on validation (5.17)
than no term at all (6.88) and than the annualized objective (7.50), which by the argument above
is the same as a weight of 6.3 per day. The 6.88 comes from a run with the annualized scale and
a weight of zero, where the scale does not matter; no run with the Sharpe ratio per day and a
weight of 6.3 was made. One seed, two years. So either the authors' code scales the two terms
differently from the printed formula, or their validation behaves differently from ours. This
is an inference; it cannot be settled without their code.

### 2. One security's explained-variance term was bounded below at -1

**The paper.** The term is the average over the securities of 1 - Var(residual) / Var(return).
No bound.

**Round 1.** The term of one security is bounded below at -1, and a security counts only if it
has a return on half of the days of the batch. The bound was added because a stock whose price
hardly moves in a batch (a takeover target) has a term far below -1 (as low as -163 in the check
below) and took over the gradient. Below the bound a security contributes no gradient.

**Is it a problem?** Yes, the bound binds often. Replaying the saved round-1 model for a test
year over its own training window (`check_issues.py floor`):

| Model for the test year | Batches in which the bound binds | Terms below -1 | Mean term, bounded | As printed | Lowest term |
|---|---|---|---|---|---|
| 1998 | 10 of 32 | 12 of 15,961 | 0.2734 | 0.2705 | -13 |
| 2010 | 30 of 32 | 54 of 15,976 | 0.4395 | 0.4138 | -79 |
| 2021 | 25 of 32 | 43 of 16,008 | 0.4556 | 0.4070 | -163 |

Few terms are affected (at most about 3 in 1,000), but they move the mean by up to 4.9 points. The
review that raised this issue reported 22 of 32 batches and 6.3 points for the 2021 model; the
count depends on where the batches are cut.

**Resolution.** Round 3 trains without the bound. On validation the effect is small and has no
clear sign: 5.17 to 5.39 with batches of 64 days, 5.56 to 5.45 with batches of 125 days.

### 3. The objective is computed on batches of 64 days

**The paper.** It gives eight-year training windows and 30 "passes over the data" (Section 4.2
and Table 4). It does not give a batch.

**Round 1.** Each optimizer step uses 64 consecutive days: one Sharpe ratio and one variance per
security on those days. An eight-year window gives about 32 steps per pass, 950 per year.

**What a cited source does.** The predecessor's code and paper use batches of 125 consecutive
days and one Sharpe ratio per batch ("consecutive and non-overlapping" subwindows of 125 days,
their Appendix B.D). So a Sharpe ratio per batch is the convention of the prior work, not a
shortcut of this replication. The length differs: 64 against 125. That code also takes its
Sharpe ratio per day (issue 1).

**Resolution.** Round 3 uses batches of 125 days (about 16 steps per pass). Validation: 5.17 to
5.56 after costs. The reading "one batch is the whole eight-year window" was tried in round 2b
and is not answered by it (issue 7). With 30 passes it would mean 30 optimizer steps, which
cannot train this model; "passes over the data" also suggests several steps per pass.

### 4. Three inputs were not built as the cited source defines them

**The paper.** The 39 characteristics follow the Internet Appendix of Chen, Pelger and Zhu
(Deep Learning in Asset Pricing). The definitions were read in the appendix of that paper's
arXiv versions 1 to 3; the final Internet Appendix sits behind a paywall and was not read.

| Input | Cited definition | Round 1 | Round 3 |
|---|---|---|---|
| Beta | Correlation with the market from overlapping three-day log returns over five years (at least 750), times the ratio of volatilities from daily log returns over one year (at least 120), after Frazzini and Pedersen | Slope of a regression on 252 daily returns | As cited. The daily data start in 1988, so until the end of 1992 the correlation window is shorter than five years and 250 observations are required. |
| Variance | "Variance of daily returns in the past two months". Its own source, the Kenneth French library, says "60 days (minimum 20) of lagged returns". | 60 trading days, at least 20 | The two calendar months before the decision month, at least 20 days |
| Resid_Var | Variance of the residuals of a regression "in the past two months" on the three Fama-French factors | Same regression, 60 trading days | Same regression, two calendar months |
| r2_1 and ST_Rev | Both are the return of the previous month in the cited text. In the authors' own tables the two give different results, so they are not the same series. | Identical | Identical (open) |

**Is it a problem?** For Beta, yes: the two versions rank the stocks differently (correlation of
the two rank inputs across the universe 0.85 on average, 0.62 at the lowest). For Variance and
Resid_Var the difference is 60 against about 42 trading days and the cited text is ambiguous;
the rank correlation is 0.95 and 0.94. The accounting definitions of round 1 were compared with
the cited table and agree, apart from the choices that `README.md` lists. One side effect of the
cited Beta: it needs 750 observations, so a stock has none for three years after its listing
and gets the median instead. In 2000 that is one member in six (83.5 percent have a value,
against 96.7 percent in round 1).

**Resolution.** Round 3 uses a second set of inputs (`data/panel_as_cited`, built by
`chars.build("cited")`). The default still reproduces the inputs of round 1 bit for bit.
Validation: 5.17 to 5.02 after costs.

### 5. The universe is not the paper's

**The paper.** "The 500 largest stocks based on the previous month market capitalization", from
the securities on CRSP.

**Round 1.** The 500 largest securities by CRSP capitalization (price times shares outstanding
of the security), share codes 10, 11, 12, 18, 48, 30 and 31.

**Is it a problem?** Yes, and it is larger than `RESULTS.md` recorded (three firms). Figures 3
and 4 of the paper name 108 firms among "the 500 equities with the largest market cap on the
first trading day of 2021". In our universe (`check_issues.py universe`) 68 of them are members
in January 2021 and 40 are not. Of those 40, 11 are members in some later month of 2021 (firms
close to our cut, for example Imperial Oil and Celanese) and 29 in no month. The 29 are of
three kinds:

- **Depositary receipts of large foreign firms** (15): for example Toyota, Unilever, Sanofi,
  TotalEnergies, BHP, Equinor, Orange, NatWest, Mizuho, Ecopetrol. CRSP counts the receipts
  outstanding, not the company. Ecopetrol's receipts rank about 7,200 by CRSP capitalization;
  the company is worth 27 billion USD in Compustat. For the 13 such firms that were checked, the
  company value in Compustat is 27 to 218 billion USD (the table per firm is not published here:
  it holds CRSP and Compustat data under license);
  the 500th security of our universe is worth about 16 billion.
- **Partnership units** (2: Energy Transfer, Cheniere Energy Partners), which our share-code
  filter excludes.
- **Firms with ordinary shares that rank about 600 to 1,000** (12: for example First Solar,
  Molson Coors, Leidos, UDR). A strict top 500 by capitalization does not hold them in 2021.

So the paper's 500 count a foreign firm with its whole value, include more than common shares,
and reach further down among ordinary shares than a ranking of all CRSP securities does (which
would fit a ranking inside a smaller set of securities, for example those with complete
characteristics). This is an inference from its figures; the rule itself is not known. It would
also fit the market row of Table 2: our equally weighted universe earns 11.76 percent a year,
the paper's 8.61. That foreign large firms earned less than U.S. large firms in this period is
a conjecture, not measured here.

**Why it matters.** The universe sets every cross-sectional rank and every softmax of the model.
Factor 5 of the paper's Figure 4 shows ten names; seven are foreign receipts, and six of those
are never in our universe in 2021.

**Resolution.** Round 4 (`DIFFERENCES.md`). It ranks by company value from the Compustat monthly
security file, admits partnership units and asks for 120 months of returns, which the ages of
the named firms support: none is younger than 10.4 years. 91 of the 108 named firms are then
members in January 2021. One of the other 17 has no company value in Compustat; what keeps the
other 16 out is not known.

The results move little. With the same settings and the same seed, the Sharpe ratios of the
Attention Factor model and of the benchmark with loadings from 60 days change by at most 0.15
between the two universes, before and after costs. Two seeds of one run differ by 0.19 before
costs and 0.11 after. After costs every result is a little lower on the new universe (by 0.05
to 0.23 for the trained models, by up to 0.45 for the OU rule). So a universe that holds 23
more of the named firms does not bring the results closer to the paper. The paper's own
universe is still not known, so this does not rule the universe out.

The market row does not confirm the new universe. The note of the paper's Table 2 calls the row
"the equally weighted market portfolio". An equally weighted portfolio of the new universe earns
11.6 percent a year with a volatility of 19.6; that of the old universe earns 11.8 with 20.6;
the paper reports 8.61 and 20.37. So the mean is no closer and the volatility is further away.
Two series that are not equally weighted are close to the paper's mean: a value-weighted
portfolio of the new universe in excess of the risk-free rate (8.5) and the S&P 500 without
dividends (8.6). Which stocks the paper's row holds is not stated
(`results/company/checks/market_row.md`).

### 6. The PCA benchmark is not built as the predecessor builds it

**The paper.** "The latent factors are estimated with PCA using the past 252 trading days",
"essentially a benchmark in the spirit of" the predecessor paper; the OU benchmark uses "the
same implementation" as that paper.

**Round 1.** PCA of the covariance matrix of 252 days; the eigenvectors are factor weights and
loadings, so the residual is an orthogonal projection of the returns.

**What the cited source does** (`dlsa-public`, `factor_models/pca.py`, and Section III.B of that
paper). It standardizes the returns (a PCA of the correlation matrix), estimates the loadings of
each stock by a regression without intercept on the last 60 days, and takes the residual of the
next day as the return minus loadings times factor returns.

**Is it a problem?** Yes. It is the largest effect found. With loadings from 60 days the
residuals are noisier: out of sample they keep 66 percent of the variance of the returns with 30
factors, against 44 percent for the projection (58 against 52 percent with 8 factors). On
validation, with 30 factors:

| PCA with the long convolution model | Batch | Sharpe ratio | after costs |
|---|---|---|---|
| Round 1: covariance matrix, projection | 64 days | 9.11 | 6.00 |
| The same | 125 days | 9.02 | 5.90 |
| Correlation matrix, loadings from all 252 days | 125 days | 9.26 | 6.03 |
| Correlation matrix, loadings from 60 days (the predecessor's) | 125 days | 5.59 | 1.27 |

Standardizing changes nothing; the 60-day regression does. With 8 factors the drop is smaller
(6.04 to 4.65).

**A limit of this reading.** A regression on 60 days cannot estimate loadings on 100 factors, and
the paper reports a PCA row with 100 factors. So the authors cannot have used the predecessor's
60-day loadings unchanged for every row. How they built the residuals is not known.

**Resolution.** Round 3 builds the benchmark the predecessor's way
(`src/attnfactors/pca_regression.py`) and trains the same long convolution policy on it.
Out of sample with 30 factors its Sharpe ratio falls from 3.47 to 1.73 before costs and from
1.26 to -1.26 after; with 8 factors from 3.07 to 2.29 and from 1.35 to 0.47. The OU threshold
rule uses the same residuals. Its two constants (1.25 and 0.25) are those of the predecessor
paper's Appendix B; the rule itself is not in the public code.

### 7. Round 2b took 30 optimizer steps a year (our mistake)

Round 2b computes the objective once per pass over the whole training window. With 30 passes
that is 30 optimizer steps a year, where rounds 1 and 2a take about 950. Its report says so, but
its headline (-1.73 after costs with 30 factors) reads as a result about the loss.

It is a result about the number of steps. The ordinary training of round 2a, stopped early
(`check_issues.py steps`, validation years, 30 factors):

| Passes over the data | Optimizer steps a year | Sharpe ratio | after costs | Turnover a day |
|---|---|---|---|---|
| 0 | 0 | -1.42 | -9.01 | 1.54 |
| 1 | 24 | 3.87 | -2.79 | 1.33 |
| 2 | 48 | 5.64 | -0.15 | 1.19 |
| 3 | 72 | 6.59 | 1.49 | 1.09 |
| 5 | 120 | 7.37 | 3.27 | 0.91 |
| 10 | 239 | 7.93 | 4.64 | 0.72 |
| 30 | 714 | 8.24 | 5.17 | 0.68 |
| Round 2b: whole window as one batch | 30 | 4.89 | -0.98 | 1.18 |

After 30 steps round 2b stands where the ordinary training stands after 24 to 48 steps, in its
Sharpe ratios and in its turnover (1.18, against 1.19 after 48 steps, 0.68 at the end of the
ordinary training and 1.54 for the untrained policy).

Two more points. Over a whole window no term falls below -1 for the securities that pass the
eligibility rule, so the bound is irrelevant there. And that rule (a return on half of the days)
keeps only about 470 of 900 securities over eight years, because many are in the universe for
less than four years; the explained-variance term of round 2b ignores the other half.

**Resolution.** Round 2b stays in the repository, marked as not valid for its question. The two
things it set out to test are in round 3 in a form that trains: no bound, and the batch of the
prior work.

### Issues found earlier (round 1)

For completeness. All were fixed before the runs of round 1 or corrected in its documents; see
`RESULTS.md`, "What did not work", and `AI_USAGE.md`.

- **Found by the audit of the code before the runs:** the WRDS API ignored filters without an
  error (wrong Compustat records for about 300 firms); a masking bug kept one security out of
  the PCA benchmarks; the firm's market equity, the timing of delisting returns, ties in the
  ranks and batches that skipped dates. No use of future information was found.
- **Found while training:** the explained-variance term blew up for stocks whose price hardly
  moves (hence the bound of issue 2); too many jobs on four cores made every run look several
  times slower than it is, and the scope was cut on that impression.
- **Found by checks of the write-up:** sentences that claimed more than the tables carry; a
  README that listed runs which were never made; "the benchmark is ahead" where only its Sharpe
  ratio is higher, within about two standard errors.

## Validation, 1996 to 1997

Six training years before each of the two years. One setting changes from row to row where the
label starts with dots. "Bound": one security's explained-variance term is bounded below at -1
(issue 2).

<!-- validation:start -->
### Attention Factor model, K = 30, seed 0

| Configuration | Sharpe ratio | after costs | Mean after costs % | Volatility % | Turnover a day |
|---|---|---|---|---|---|
| Round 1 specification: annualized Sharpe, batch 64, bound | 10.91 | 7.50 | 20.90 | 2.79 | 0.66 |
| Sharpe per day (the printed loss), batch 64, bound | 8.24 | 5.17 | 16.49 | 3.19 | 0.68 |
| ... and no bound (run of round 2b) | 8.58 | 5.39 | 16.62 | 3.09 | 0.68 |
| ... and batch 125, bound | 8.71 | 5.56 | 16.67 | 3.00 | 0.65 |
| ... and batch 125, no bound | 8.60 | 5.45 | 16.60 | 3.05 | 0.66 |
| ... and cited inputs (batch 64, bound) | 8.09 | 5.02 | 15.91 | 3.17 | 0.67 |
| Round 3 specification: per day, batch 125, no bound, cited inputs | 8.50 | 5.36 | 16.32 | 3.05 | 0.66 |
| Round 3 specification with the annualized Sharpe | 10.26 | 7.13 | 21.09 | 2.96 | 0.64 |
| Whole six-year window as one batch, 30 optimizer steps a year (round 2b) | 4.89 | -0.98 | -2.70 | 2.75 | 1.18 |

### Benchmarks, K = 30 unless noted

| Configuration | Sharpe ratio | after costs | Mean after costs % | Volatility % | Turnover a day |
|---|---|---|---|---|---|
| PCA with the long convolution model: projection, batch 64 (round 1) | 9.11 | 6.00 | 17.00 | 2.84 | 0.61 |
| ... projection, batch 125 | 9.02 | 5.90 | 16.48 | 2.81 | 0.60 |
| ... correlation matrix, loadings from all 252 days, batch 125 | 9.26 | 6.03 | 16.63 | 2.77 | 0.61 |
| ... correlation matrix, loadings from 60 days, batch 64 | 5.64 | 1.29 | 2.81 | 2.18 | 0.65 |
| ... correlation matrix, loadings from 60 days, batch 125 (round 3) | 5.59 | 1.27 | 2.74 | 2.17 | 0.65 |
| PCA with an OU threshold rule: projection (K = 8, round 1) | 3.92 | -0.49 | -1.52 | 3.13 | 0.99 |
| ... loadings from 60 days (K = 8) | 3.56 | -1.54 | -4.09 | 2.66 | 0.98 |
| ... loadings from 60 days (K = 30) | 2.62 | -4.34 | -8.20 | 1.89 | 0.94 |

### K = 8

| Configuration | Sharpe ratio | after costs | Mean after costs % | Volatility % | Turnover a day |
|---|---|---|---|---|---|
| Attention Factor model, round 1 specification | 8.98 | 6.53 | 23.83 | 3.65 | 0.62 |
| Attention Factor model, round 3 specification | 7.75 | 5.17 | 18.59 | 3.60 | 0.64 |
| PCA with the long convolution model, projection (round 1) | 8.70 | 6.04 | 21.56 | 3.58 | 0.66 |
| PCA with the long convolution model, loadings from 60 days (round 3) | 7.53 | 4.65 | 14.56 | 3.14 | 0.62 |
<!-- validation:end -->

Read with care: two years, one seed, and round 1 chose its settings on these years.

## What is still open

1. **The universe (issue 5).** Round 4 rebuilt it with an inferred rule and holds 91 of the 108
   named firms; the results moved little. Still open: what keeps 16 of the other 17 out (the
   paper's 500 reach down to firms of about 10 billion USD and must leave out about 100 that
   the rule ranks above them), which stocks the paper's market row holds, and the firms that
   never report quarterly shares to Compustat, which still rank by their receipts.
2. **The scale of the two terms of the objective (issue 1).** Not knowable without the authors'
   code. Round 3 uses the printed formula. On validation the annualized variant of round 3
   scores 7.13 after costs against 5.36. Round 4 ran both out of sample on its universe: the
   annualized variant reaches 3.01 before costs and 0.96 after with 30 factors (2.58 and 1.14
   with 8), the printed formula 2.03 and -0.03 (1.91 and 0.42).
3. **The whole window as one batch (issue 3)** with enough optimizer steps. One step costs as
   much as one pass of the ordinary training, so 950 steps would cost about 30 runs.
4. **r2_1 and ST_Rev (issue 4)** are one series here and two in the authors' data.
5. **How the paper builds its PCA residuals (issue 6).** The predecessor's 60-day loadings
   cannot be its recipe for 100 factors.
6. **Unchanged limits of scope:** 100 factors and the row of Table 3 without the past returns
   were run in round 1 only, with one seed each (`RESULTS.md`, "Open choices tested"); no
   other row of Table 3; one or two seeds.

## What this changes on the slides

The slides and the two chapters of Part 3 on the site show round 1 only. They were not changed
with this file; what the talk says is for the group to decide. These statements are out of date:

| On the slides now | What the three rounds say |
|---|---|
| "Departure from the printed loss: its Sharpe ratio is annualized", one bullet | The headline rests on this departure. With the printed loss the 30-factor model has 2.1 before costs and 0.1 after. |
| "Reproduced in part: Sharpe ratio 3.00 of 3.97 before costs" | 3.00 with the annualized objective, 2.05 to 2.10 with the printed one |
| "Not reproduced: the lead over the baseline (net: 0.99 against 1.26)" | True against a baseline built by projection. Against the predecessor's construction the model is above the baseline after costs with 30 factors (0.09 against -1.26) and level with it with 8. |
| "Volatility: baseline 3.21 percent, model 4.26; paper's baseline 5.42" | The predecessor's construction gives 2.55. Neither reproduces 5.42. |
| "Validation only: net Sharpe 7.50; as printed 5.17; baseline 6.00" | Still true. The baseline built the predecessor's way has 1.27. |
| "Checks: 14 tests" | 34 tests |
| "Soft spot: almost no profit after costs since 2003" | Holds in every round: 2003 to 2021, the 30-factor model of round 3 has -0.93 and -0.84. |
| The universe differs by three named firms (chapter text) | 40 of 108 named firms are not in our universe on the date of the paper's figures |
| Headings "Before costs the level holds, not the ranking" and "The level reproduces; the ranking does not" | The level (3.00 of 3.97) holds only with the annualized objective; with the Sharpe ratio per day it is 2.05 to 2.10. The ranking depends on how the benchmark is built. |
| "Replication, 30 factors, 2003 to 2021: net Sharpe ratio -0.16 to 0.04" | Round 1, model and benchmark. Round 3: -0.93 and -0.84 for the model, -2.48 and -2.31 for the benchmark built the predecessor's way. |
| "Net Sharpe gap: -0.25; standard error: 0.20 to 0.25" | Round 1 against the benchmark by projection. Round 3 against the predecessor's construction: +1.43, standard error 0.22 to 0.26. |

## Reproduce

From `replication/`, with the data in place (`README.md`, "How to run"):

```bash
# round 1 (the slides): see README.md and RESULTS.md
.venv/bin/python scripts/replicate_daily_sharpe.py          # round 2a, about 2 hours
.venv/bin/python scripts/replicate_adjusted_loss.py         # round 2b, about 1.5 hours (not a valid test, issue 7)
.venv/bin/python scripts/replicate_as_cited.py build        # round 3: second inputs and PCA residuals, 3 minutes
.venv/bin/python scripts/replicate_as_cited.py validation   # 13 runs on 1996 to 1997, about 15 minutes
.venv/bin/python scripts/replicate_as_cited.py main         # 8 runs on 1998 to 2021, about 2 hours
.venv/bin/python scripts/check_issues.py                    # the checks of issues 2, 5 and 7
.venv/bin/python scripts/replicate_company_universe.py main  # round 4, after its stages download, universe, data, build
.venv/bin/python -m pytest tests -q                         # 34 tests
```

Without data, from the committed result files:

```bash
.venv/bin/python scripts/replicate_as_cited.py report       # results/as_cited/comparison.md and the tables of this file
```

Files of round 3:

```
scripts/replicate_as_cited.py       the runs of round 3 and its validation runs
scripts/as_cited_comparison.py      tables of all three rounds from the saved files
scripts/check_issues.py             the checks cited in this file
src/attnfactors/pca_regression.py   PCA residuals with loadings from a regression (issue 6)
src/attnfactors/chars.py            build("cited"): Beta, Variance, Resid_Var as cited (issue 4)
results/as_cited/<run>/             metrics.json and daily.csv of each run (committed)
results/as_cited/comparison.md      every table
results/as_cited/checks/            output of the checks
data/panel_as_cited/, runs/as_cited/  inputs and full run output (not committed)
```

Changes to code that round 1 also uses are optional settings with the old value as default: the
panel folder (`ATTNFACTORS_PANEL`), the bound of the explained-variance term (`ev_floor`), and
the definitions of three characteristics. Round 1 and round 2 reproduce as before.

## How this file and round 3 were checked

Two AI reviewers worked independently of the session that wrote the code and this file
(`AI_USAGE.md`). No human has repeated their checks.

- **Numbers.** One recomputed every number of this file from the saved daily returns with its
  own code, including the bootstrap standard errors. All numbers that come from a run were
  right. It found three wrong sentences and several overstated ones in the text; they are
  corrected.
- **Code.** The other looked for use of future information and misaligned dates in the new code.
  It found none. On a synthetic panel, changing the data after a day changes no residual, input
  or weight dated before it. On the real panel it recomputed the cited inputs for 720 cells
  and the stored PCA residuals for 45 days independently, and it reproduced the saved weights
  of round 3 from the saved models, bit for bit, with the settings that the run names claim.
- **What it flagged.** Safeguards, not results: a run folder could have been continued with
  another set of PCA residuals without an error, and the new residual files were not checked
  against the panel they were built from. Both are fixed. Two source files were edited 34
  seconds after four runs had started (the edit added the variant with 252-day loadings); the
  saved outputs are reproduced by the code as it is now.
- **Not covered by unit tests:** the dating inside `pca_regression.build` and inside
  `chars.build`. The reviewer's perturbation experiments cover them; they are not in `tests/`.

These checks are of rounds 1 to 3 and of this file. Round 4 has its own review, described in
`AI_USAGE.md`.
