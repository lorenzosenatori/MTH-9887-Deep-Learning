# Round 4: the closest reproduction, and what still differs from the paper

Status of 23 November 2026. This file covers a subset of the paper's Table 2: 8 and 30
factors, for the two-step benchmarks (PCA with an OU threshold rule, PCA with the long
convolution model) and the one-step Attention Factor model. It says how close this round comes
and lists every known difference that is left. `README_ISSUES.md` has the history of rounds 1
to 3; `results/company/comparison.md` has every table of this round.

The paper is Epstein, Wang, Choi and Pelger (2025), "Attention Factors for Statistical
Arbitrage". All numbers are out of sample, January 1998 to December 2021, and annualized.
"After costs" is after the paper's cost of 5 basis points per unit traded and 1 basis point per
unit held short.

## In short

- **What was run.** Thirteen runs on a rebuilt universe: the Attention Factor model, PCA with
  the long convolution model and PCA with an OU threshold rule, each with 8 and 30 factors. Two
  choices that the paper leaves open are run both ways: the scale of the Sharpe ratio inside the
  objective, and how the PCA residuals of the benchmarks are built.
- **Moving to a universe that holds more of the paper's firms does not move the results.** 91
  of the 108 firms that the paper names are members on the date of its figures (68 before). With
  the same settings and the same seed, the Sharpe ratio of a trained model moves by at most 0.15
  between the two universes, before and after costs. Two seeds of one run differ by 0.19 before
  costs and 0.11 after.
- **The paper's own universe is still not known, so this does not rule the universe out.** 17
  named firms are missing. The paper's equally weighted market portfolio earns 8.61 percent a
  year; ours earns 11.6 on this universe and 11.8 on the old one.
- **The level of the paper is not reached in any reading.** With the objective as the paper
  prints it, the Attention Factor model has a Sharpe ratio of 1.91 before costs and 0.42 after
  with 8 factors (paper: 3.35 and 1.94), and 2.03 and -0.03 with 30 factors (paper: 3.97 and
  2.28). With an annualized Sharpe ratio in the objective it has 2.58 and 1.14, and 3.01 and
  0.96. That is 77 and 76 percent of the paper's Sharpe ratio before costs, and 59 and 42 percent
  after.
- **For the Attention Factor model, the scale of the Sharpe ratio in the objective decides the
  level; nothing else that was changed does.** With 30 factors and the same seed, the
  annualized objective gave 2.97 and 0.94 in round 1 and gives 3.01 and 0.96 here. The printed
  objective gave 1.94 and 0.02 in round 3 and gives 2.03 and -0.03 here. Round 1 and this round
  differ in the universe, three inputs, the length of a batch and the bound on the
  explained-variance term. Round 3 and this round differ in the universe and in what came with
  it (the market value inside the inputs, the rule for days without a trade). One seed stands
  behind each number, and the changes were made together, so effects that offset each other are
  not excluded. (`README_ISSUES.md` quotes means of two seeds for the earlier rounds: 3.00 and
  0.99, 2.05 and 0.09.)
- **The benchmarks.** For PCA with the long convolution model the paper's Sharpe ratios before
  costs (2.64 and 2.79) lie between our two constructions (2.33 and 2.89 with 8 factors, 1.76
  and 3.64 with 30). After costs the paper (1.50 and 1.57) is above both (0.32 and 1.13; -1.38
  and 1.19). The OU threshold rule loses heavily after costs, as in the paper (-3.24 to -5.80;
  paper: -4.15 and -6.45). Before costs it does better here (1.07 to 1.66; paper: 0.78 and 0.18).
- **The paper's ranking appears against one construction of the benchmark only.** In the paper
  the Attention Factor model is ahead of PCA with the long convolution model by 0.71 and 1.18
  before costs (8 and 30 factors) and by 0.44 and 0.71 after. Against a benchmark with loadings
  from 60 days our model is ahead after costs in all four comparisons (by 0.10 to 2.34) and
  before costs in three of four. Against a benchmark built by projection it is behind in seven
  of eight comparisons (by 0.23 to 1.61) and level in the eighth. Nine of the sixteen
  differences are more than two standard errors from zero: four for the model, all against the
  loadings from 60 days, and five against it, all against the projection.
- **After costs most of the profit is earned before 2003.** With the printed objective all of
  it is: the returns after costs from 2003 to 2021 sum to a loss. With the annualized objective
  93 percent is earned before 2003 with 30 factors and 61 percent with 8. From 2003 to 2021 the
  Sharpe ratio of the Attention Factor model after costs is between -0.87 and 0.62.
- **What is left.** The table in "What still differs in the setup" lists every known
  difference. Those that could move the results and cannot be settled from the paper or its
  sources: the scale of the Sharpe ratio in the objective, the construction of the PCA
  residuals, the rule behind 16 of the 17 named firms that are still missing, the stocks behind
  the paper's market row, the length of a batch, the ridge penalty, and the number of seeds.

## What this round changes: the universe

Rounds 1 to 3 traded the 500 largest securities by CRSP capitalization. The paper's Figures 3
and 4 name 108 firms among "the 500 equities with the largest market cap on the first trading
day of 2021", and only 68 of them were in that universe. This round changes the rule in three
ways that the 108 names support:

1. **A firm counts with the value of the whole company.** The value is the price times the
   company's shares in the Compustat monthly security file, or the sum of the firm's securities
   in CRSP where that is larger or Compustat has no value. A depositary receipt of a foreign
   firm then ranks with the firm (Toyota: 216 billion USD) and not with the receipts outstanding
   (7 billion). Compustat has the company's shares only in the months that end a fiscal
   quarter. They are carried forward for up to a year, adjusted for splits, as the median of
   the last three quarters. A foreign firm that never reports quarterly shares to Compustat
   (TotalEnergies is one) has no company value and still ranks by its receipts. The firm's
   value stands on its largest security only. A second share class ranks with its own CRSP
   value: Alphabet's two classes are both members, as in the paper's figure; a class that
   hardly trades is not.
2. **Partnership units are admitted** (share codes 71 and 72), beside ordinary shares, trusts
   and depositary receipts. The paper names two partnerships.
3. **A firm needs 120 months of returns.** No firm that the paper names is younger than 10.4
   years at the end of 2020, although more than a quarter of the equally large firms are. The
   chance of that without a rule is below one in a billion. The paper does not state such a
   rule; 120 months is the round number that fits (8 years of training and 2 more).

<!-- universe_rules:start -->
Figures 3 and 4 of the paper name 108 firms among its 500 on the first trading day of 2021. A rule is
scored by how many of them it puts among its 500 for January 2021, and by the mean and the volatility of its
equally weighted universe over 1998 to 2021, computed here from monthly returns (the paper's market row,
from daily returns: 8.61 and 20.37 percent a year).

| Rule | Named firms among the 500 | Value of the 500th, billion USD | Depositary receipts | Partnership units | Mean % | Volatility % | Securities ever in the universe |
|---|---|---|---|---|---|---|---|
| Rounds 1 to 3: CRSP value of the security, no units, no history | 68 of 108 | 15.6 | 20 | 0 | 10.36 | 16.52 | 2,093 |
| Company value | 70 of 108 | 19.2 | 86 | 0 | 9.99 | 16.61 | 2,028 |
| Company value, units admitted | 69 of 108 | 19.3 | 86 | 2 | 9.98 | 16.61 | 2,039 |
| Company value, units, history of 96 months | 88 of 108 | 15.0 | 78 | 5 | 10.46 | 15.74 | 1,800 |
| This round: company value, units, history of 120 months | 91 of 108 | 14.2 | 77 | 5 | 10.32 | 15.64 | 1,768 |
| Company value, units, history of 144 months | 89 of 108 | 13.1 | 74 | 5 | 10.38 | 15.52 | 1,735 |

Named firms that this round's universe still does not hold in January 2021 (17): Alliant Energy, Camden Property Trust, Entegris, Equity Lifestyle Properties, Evergy, First Solar, H World, Huntington Bancshares, I P G Photonics, Imperial Oil, Liberty Global, Medical Properties Trust, Molson Coors, On Semiconductor, Totalenergies, U D R Inc, United Airlines.

Overlap with the universe of rounds 1 to 3: 80 percent of the member-months (1988-01 to 2021-12); in January 2021, 389 of 500 securities are in both.
<!-- universe_rules:end -->

The rule is an inference from 108 names on one date. The value alone does not help: it brings
the foreign firms in and pushes as many others out. The history rule does most of the work.
Of the 17 named firms that are still missing, one is TotalEnergies. What separates the other 16
from the 500 is not known: all of them are eligible and worth 10 to 14 billion USD, just below
this round's cut of 14.2 billion. So the paper's 500 reach down to about 10 billion and leave out
about 100 firms that this rule ranks above that, for a reason that the names do not reveal.

**One rule on the data is new as well.** A day on which a security did not trade (a volume of
zero) has no return in this round, and neither has the day after it. That is 0.46 percent of the
member-days, mostly of receipts and mostly before 1998. The universe now holds receipts with a
small float, and their quotes on days without a trade are sometimes far from any price: one
receipt shows +329 percent and -79 percent on two such days. A security without a return cannot
be traded by the models for the following 30 days, and it leaves the PCA of the benchmarks for a
year. The rule keeps about 15 of the 500 members out of trading on a day in 1990 to 1997, 3 in
1998 to 2009 and none after.

The same company value replaces the CRSP capitalization inside the inputs (the size
characteristic and the ratios to market equity). Everything else is round 3: the inputs as the
paper's source defines them, batches of 125 days, no bound on the explained-variance term.

**Two things the paper leaves open are run both ways:**

- the Sharpe ratio inside the objective: per day, as the formula is printed, and annualized
  (`README_ISSUES.md`, issue 1);
- the PCA residuals of the benchmarks: loadings from a regression on 60 days, as the
  predecessor's public code builds them, and an orthogonal projection on the eigenvectors
  (`README_ISSUES.md`, issue 6).

One seed for every trained model, and a second seed for one of them. The paper reports the
mean over several seeds.

## Results beside the paper

Sharpe ratio before costs / after costs.

<!-- summary:start -->
| | 8 factors: paper | this round | 30 factors: paper | this round |
|---|---|---|---|---|
| Attention Factor model, objective as printed | 3.35 / 1.94 | 1.91 / 0.42 | 3.97 / 2.28 | 2.03 / -0.03 |
| Attention Factor model, annualized Sharpe ratio in the objective | 3.35 / 1.94 | 2.58 / 1.14 | 3.97 / 2.28 | 3.01 / 0.96 |
| PCA with the long convolution model, loadings from 60 days | 2.64 / 1.50 | 2.33 / 0.32 | 2.79 / 1.57 | 1.76 / -1.38 |
| PCA with the long convolution model, projection | 2.64 / 1.50 | 2.89 / 1.13 | 2.79 / 1.57 | 3.64 / 1.19 |
| PCA with an OU threshold rule, loadings from 60 days | 0.78 / -4.15 | 1.30 / -3.74 | 0.18 / -6.45 | 1.07 / -5.80 |
| PCA with an OU threshold rule, projection | 0.78 / -4.15 | 1.30 / -3.24 | 0.18 / -6.45 | 1.66 / -4.10 |

Second seed of the Attention Factor model with 30 factors, objective as printed: 2.21 / 0.08.
<!-- summary:end -->

**Attention Factor model.** Both readings of the objective end below the paper, and the
annualized one ends closer: 2.58 and 3.01 before costs against 3.35 and 3.97. The second seed
with 30 factors gives 2.21 before costs and 0.08 after, against 2.03 and -0.03 for the first.
The paper reports a standard deviation of 0.13 across its seeds, so one seed is not what
separates 2.1 from 4.0. Averaging does not close the gap either: the mean of the two seeds'
daily returns has a Sharpe ratio of 2.22 before costs and 0.02 after, and the two series have a
correlation of 0.82.

**PCA with the long convolution model.** With 30 factors, how the residuals are built matters
more than anything else in this table: the projection gives 3.64 before costs and the loadings
from 60 days give 1.76. The paper's 2.79 is between them. With 8 factors the two constructions
are 0.56 apart, less than the two readings of the model's objective (0.68). After costs neither
construction reaches the paper's 1.50 and 1.57.

**PCA with an OU threshold rule.** The rule has no trained part. It reproduces the paper's
finding that the rule loses its return several times over in costs. With loadings from 60 days
it is within 0.7 of the paper's Sharpe ratio after costs (-3.74 against -4.15, -5.80 against
-6.45).

### Where the gap sits: mean, volatility, cost

<!-- decomposition:start -->
Each cell: the paper / this round. Cost: the mean minus the mean after costs.

| | Mean, % a year | Volatility, % a year | Cost, % a year |
|---|---|---|---|
| Attention Factor model, objective as printed, K = 8 | 15.70 / 10.74 | 4.68 / 5.62 | 6.65 / 8.35 |
| Attention Factor model, annualized Sharpe ratio in the objective, K = 8 | 15.70 / 13.86 | 4.68 / 5.36 | 6.65 / 7.74 |
| PCA with the long convolution model, loadings from 60 days, K = 8 | 14.88 / 8.78 | 5.63 / 3.77 | 6.46 / 7.58 |
| PCA with the long convolution model, projection, K = 8 | 14.88 / 12.60 | 5.63 / 4.35 | 6.46 / 7.71 |
| PCA with an OU threshold rule, loadings from 60 days, K = 8 | 2.04 / 3.42 | 2.61 / 2.62 | 12.87 / 13.22 |
| PCA with an OU threshold rule, projection, K = 8 | 2.04 / 3.84 | 2.61 / 2.95 | 12.87 / 13.42 |
| Attention Factor model, objective as printed, K = 30 | 16.66 / 9.10 | 4.20 / 4.49 | 7.14 / 9.25 |
| Attention Factor model, annualized Sharpe ratio in the objective, K = 30 | 16.66 / 12.20 | 4.20 / 4.05 | 7.14 / 8.33 |
| PCA with the long convolution model, loadings from 60 days, K = 30 | 15.15 / 4.12 | 5.42 / 2.34 | 6.68 / 7.34 |
| PCA with the long convolution model, projection, K = 30 | 15.15 / 11.03 | 5.42 / 3.03 | 6.68 / 7.44 |
| PCA with an OU threshold rule, loadings from 60 days, K = 30 | 0.40 / 2.02 | 2.24 / 1.89 | 15.14 / 12.97 |
| PCA with an OU threshold rule, projection, K = 30 | 0.40 / 3.84 | 2.24 / 2.31 | 15.14 / 13.31 |
<!-- decomposition:end -->

With 30 factors the Attention Factor model falls short in the mean alone. It earns 9.1 and
12.2 percent a year before costs where the paper reports 16.7, at a volatility within 7 percent
of the paper's. With 8 factors both differ: the mean is 10.7 and 13.9 against 15.7, and the
volatility is 15 to 20 percent above the paper's (5.36 and 5.62 against 4.68). At the paper's
volatility the Sharpe ratios with 8 factors would be 0.38 higher. That is half of the gap for
the annualized objective and a quarter of it for the printed one. The model also pays more in
costs: 7.7 to 9.2 points a year against 6.7 and 7.1.

The paper's PCA benchmark takes more risk than ours for the same gross exposure. Its
volatility is 5.4 to 5.6 percent where ours is 2.3 to 4.4, its mean is about 15 percent where
ours is 4.1 to 12.6, and its market beta is 0.09 where ours is 0.01 to 0.06. Neither of our two
constructions gives such a portfolio. This suggests that the paper builds or scales the
benchmark in a third way that we have not identified.

The OU rule agrees with the paper in volatility (1.9 to 3.0 against 2.2 and 2.6). In cost it
agrees with 8 factors (13.2 and 13.4 points against 12.9) and pays less than the paper's with
30 (13.0 and 13.3 against 15.1).

### What the new universe changes

Sharpe ratio before costs / after costs.

<!-- universe_effect:start -->
| | 8 factors: universe of rounds 1 to 3 | this round | 30 factors: universe of rounds 1 to 3 | this round |
|---|---|---|---|---|
| Attention Factor model, objective as printed | 2.06 / 0.53 | 1.91 / 0.42 | 1.94 / 0.02 | 2.03 / -0.03 |
| PCA with the long convolution model, loadings from 60 days | 2.29 / 0.47 | 2.33 / 0.32 | 1.64 / -1.33 | 1.76 / -1.38 |
| PCA with the long convolution model, projection | 3.07 / 1.35 | 2.89 / 1.13 | 3.47 / 1.26 | 3.64 / 1.19 |
| PCA with an OU threshold rule, loadings from 60 days | 1.38 / -3.29 | 1.30 / -3.74 | 0.90 / -5.61 | 1.07 / -5.80 |
| PCA with an OU threshold rule, projection | 1.16 / -3.16 | 1.30 / -3.24 | 1.31 / -3.95 | 1.66 / -4.10 |

The runs on the universe of rounds 1 to 3 are those of round 3 (`results/as_cited/`), with the same settings and the
same seed. For the two rows with a projection they are the runs of round 1 (`results/runs/`), where the residuals
are built the same way. The trained one of the two used batches of 64 days and an annualized Sharpe ratio in its
objective; that objective has one term, so the scale only rescales it.
<!-- universe_effect:end -->

The trained models move by at most 0.15 in the four pairs with the same settings (the Attention
Factor model and the benchmark with loadings from 60 days), and by at most 0.23 in the pairs
with a projection. Before costs the changes go both ways (-0.17 to +0.16 for the trained
models). After costs every result is a little lower on the new universe, by 0.05 to 0.23 for
the trained models. The OU rule after costs moves by up to 0.45, at a level of -3 to -6.

The universe was the largest difference that was still open after round 3. Bringing 23 more of
the named firms in does not bring the results closer to the paper. This tests one other
universe, built by an inferred rule, which shares 80 percent of its member-months with the old
one. It does not test the paper's universe, which is not known.

### The model against the benchmark

Sharpe ratio of the Attention Factor model minus that of PCA with the long convolution model,
on the same days, with the standard error of the difference from a block bootstrap.

<!-- gaps:start -->
| | | Sharpe ratios | Difference | Paper | Bootstrap s.e., blocks of 1 / 21 / 63 days |
|---|---|---|---|---|---|
| K = 30: model as printed, benchmark loadings from 60 days | before costs | 2.03 and 1.76 | +0.27 | +1.18 | 0.24 / 0.28 / 0.28 |
| K = 30: model as printed, benchmark loadings from 60 days | after costs | -0.03 and -1.38 | +1.35 | +0.71 | 0.23 / 0.25 / 0.25 |
| K = 30: model as printed, benchmark projection | before costs | 2.03 and 3.64 | -1.61 | +1.18 | 0.22 / 0.31 / 0.32 |
| K = 30: model as printed, benchmark projection | after costs | -0.03 and 1.19 | -1.22 | +0.71 | 0.20 / 0.24 / 0.26 |
| K = 30: model annualized, benchmark loadings from 60 days | before costs | 3.01 and 1.76 | +1.25 | +1.18 | 0.23 / 0.24 / 0.24 |
| K = 30: model annualized, benchmark loadings from 60 days | after costs | 0.96 and -1.38 | +2.34 | +0.71 | 0.23 / 0.23 / 0.22 |
| K = 30: model annualized, benchmark projection | before costs | 3.01 and 3.64 | -0.62 | +1.18 | 0.20 / 0.26 / 0.28 |
| K = 30: model annualized, benchmark projection | after costs | 0.96 and 1.19 | -0.23 | +0.71 | 0.19 / 0.22 / 0.23 |
| K = 8: model as printed, benchmark loadings from 60 days | before costs | 1.91 and 2.33 | -0.42 | +0.71 | 0.21 / 0.23 / 0.23 |
| K = 8: model as printed, benchmark loadings from 60 days | after costs | 0.42 and 0.32 | +0.10 | +0.44 | 0.19 / 0.20 / 0.19 |
| K = 8: model as printed, benchmark projection | before costs | 1.91 and 2.89 | -0.98 | +0.71 | 0.20 / 0.21 / 0.22 |
| K = 8: model as printed, benchmark projection | after costs | 0.42 and 1.13 | -0.70 | +0.44 | 0.18 / 0.18 / 0.19 |
| K = 8: model annualized, benchmark loadings from 60 days | before costs | 2.58 and 2.33 | +0.25 | +0.71 | 0.19 / 0.23 / 0.23 |
| K = 8: model annualized, benchmark loadings from 60 days | after costs | 1.14 and 0.32 | +0.82 | +0.44 | 0.18 / 0.19 / 0.19 |
| K = 8: model annualized, benchmark projection | before costs | 2.58 and 2.89 | -0.31 | +0.71 | 0.19 / 0.22 / 0.22 |
| K = 8: model annualized, benchmark projection | after costs | 1.14 and 1.13 | +0.01 | +0.44 | 0.17 / 0.17 / 0.18 |
<!-- gaps:end -->

A standard error is 0.17 to 0.32. Nine of the sixteen differences are more than two standard
errors from zero. Four of them favour the model, all against the loadings from 60 days (+0.82
to +2.34). Five go against it, all against the projection (-0.62 to -1.61). The other seven
(-0.42 to +0.27) are not distinguishable from zero. The paper's lead before costs with 30
factors (+1.18) is
matched by one pairing: the annualized objective against the benchmark with loadings from 60
days (+1.25). In that pairing both Sharpe ratios are far below the paper's (3.01 and 1.76
against 3.97 and 2.79).

### By period

<!-- periods:start -->
Sharpe ratio before costs / after costs / mean after costs in % a year.

| Run | 1998-2002 | 2003-2009 | 2010-2021 | 2003-2021 |
|---|---|---|---|---|
| Attention Factor model, objective as printed, K = 30 | 3.47 / 1.95 / 12.1 | 1.89 / -0.39 / -1.7 | 1.24 / -1.22 / -4.3 | 1.50 / -0.87 / -3.4 |
| Attention Factor model, objective as printed, K = 8 | 2.78 / 1.61 / 12.6 | 1.71 / -0.07 / -0.4 | 1.55 / -0.06 / -0.3 | 1.61 / -0.06 / -0.3 |
| Attention Factor model, annualized Sharpe ratio in the objective, K = 30 | 4.98 / 3.25 / 17.3 | 2.87 / 0.79 / 3.4 | 1.97 / -0.45 / -1.4 | 2.34 / 0.10 / 0.3 |
| Attention Factor model, annualized Sharpe ratio in the objective, K = 8 | 3.72 / 2.55 / 18.0 | 2.51 / 0.80 / 4.1 | 2.02 / 0.51 / 2.4 | 2.21 / 0.62 / 3.0 |
| PCA with the long convolution model, loadings from 60 days, K = 30 | 3.79 / 1.09 / 3.5 | 1.40 / -2.08 / -4.8 | 0.67 / -2.83 / -5.1 | 0.98 / -2.49 / -5.0 |
| PCA with the long convolution model, loadings from 60 days, K = 8 | 4.79 / 2.93 / 14.7 | 1.62 / -0.68 / -2.6 | 1.33 / -0.73 / -2.1 | 1.44 / -0.70 / -2.3 |
| PCA with the long convolution model, projection, K = 30 | 5.93 / 3.78 / 15.7 | 3.75 / 0.70 / 2.0 | 2.25 / -0.22 / -0.5 | 2.85 / 0.16 / 0.4 |
| PCA with the long convolution model, projection, K = 8 | 4.95 / 3.43 / 20.3 | 2.73 / 0.51 / 2.1 | 1.80 / 0.03 / 0.1 | 2.17 / 0.23 / 0.9 |
<!-- periods:end -->

Every run in the table has its highest Sharpe ratio in the period 1998 to 2002. After costs,
from 2003 to 2021, the Attention
Factor model has a Sharpe ratio of -0.87 to 0.62 and the benchmark of -2.49 to 0.23. From 2010
to 2021 only the model with 8 factors and the annualized objective keeps a positive mean after
costs (2.4 percent a year), apart from 0.1 percent for the projection benchmark with 8 factors.

### The forced delistings of January 2021

<!-- delisting_2021:start -->
Returns booked on 11 January 2021 for the receipts of China Unicom (-96 percent), China Telecom (-94 percent), China Mobile (-30 percent).
"Without": the result of that day without the three returns. Sharpe ratios before / after costs.

| Run | Weight on the three, % of the portfolio | Their part of the day's return, points | 1998 to 2021 | without | 2021 alone | without |
|---|---|---|---|---|---|---|
| Attention Factor model, objective as printed, K = 8 | -2.74 | +2.25 | 1.91 / 0.42 | 1.90 / 0.41 | 1.95 / 0.80 | 1.72 / 0.45 |
| Attention Factor model, annualized Sharpe ratio in the objective, K = 8 | -1.72 | +1.49 | 2.58 / 1.14 | 2.58 / 1.13 | 1.89 / 0.79 | 1.70 / 0.54 |
| PCA with the long convolution model, loadings from 60 days, K = 8 | -0.59 | +0.74 | 2.33 / 0.32 | 2.33 / 0.31 | 0.23 / -1.23 | 0.03 / -1.49 |
| PCA with the long convolution model, projection, K = 8 | -0.59 | +0.77 | 2.89 / 1.13 | 2.89 / 1.12 | 0.39 / -0.63 | 0.26 / -0.78 |
| PCA with an OU threshold rule, loadings from 60 days, K = 8 | +0.38 | +0.06 | 1.30 / -3.74 | 1.30 / -3.74 | -0.90 / -6.77 | -0.92 / -6.79 |
| PCA with an OU threshold rule, projection, K = 8 | +0.70 | -0.69 | 1.30 / -3.24 | 1.31 / -3.24 | -0.93 / -5.40 | -0.73 / -5.42 |
| Attention Factor model, objective as printed, K = 30 | -1.53 | +1.32 | 2.03 / -0.03 | 2.02 / -0.05 | 0.69 / -0.87 | 0.44 / -1.28 |
| Attention Factor model, annualized Sharpe ratio in the objective, K = 30 | +0.04 | +0.17 | 3.01 / 0.96 | 3.01 / 0.96 | 1.02 / -0.47 | 0.99 / -0.51 |
| PCA with the long convolution model, loadings from 60 days, K = 30 | +0.65 | -0.47 | 1.76 / -1.38 | 1.77 / -1.37 | -0.57 / -3.73 | -0.32 / -3.62 |
| PCA with the long convolution model, projection, K = 30 | +1.00 | -0.56 | 3.64 / 1.19 | 3.64 / 1.20 | -0.37 / -2.03 | -0.18 / -1.83 |
| PCA with an OU threshold rule, loadings from 60 days, K = 30 | +0.11 | -0.21 | 1.07 / -5.80 | 1.07 / -5.80 | -0.53 / -8.96 | -0.39 / -8.98 |
| PCA with an OU threshold rule, projection, K = 30 | +0.93 | -0.44 | 1.66 / -4.10 | 1.67 / -4.09 | -1.81 / -8.25 | -1.65 / -8.30 |
| Attention Factor model, objective as printed, K = 30, second seed | -1.83 | +1.54 | 2.21 / 0.08 | 2.21 / 0.07 | 1.61 / 0.09 | 1.42 / -0.25 |
<!-- delisting_2021:end -->

Three receipts of Chinese firms stopped trading in New York after 8 January 2021 because of a
U.S. executive order. CRSP reports delisting returns of -96 and -94 percent for two of them. It
reports none for the third, which gets the -30 percent that this code books for a forced
delisting without a reported return. The firms' shares kept trading in Hong Kong, so these are
not losses that a holder had to realize. The three receipts are members of this round's
universe only; the universe of rounds 1 to 3 did not hold them in 2021.

All three returns are kept: two as CRSP reports them, the third as this code fills a missing
one. The Attention Factor model with the printed
objective was short the three receipts and gains 1.3 to 2.3 points on that day. Over the 24
years this moves no Sharpe ratio by more than 0.02. In the year 2021 alone it moves the Sharpe
ratio after costs by up to 0.41.

### Every number of the subset

<!-- results:start -->
**8 factors**

| | Paper | This round | This round minus paper | Same specification, universe of rounds 1 to 3 |
|---|---|---|---|---|
| *Attention Factor model, objective as printed* | | | | |
| Sharpe ratio | 3.35 | 1.91 | -1.44 | 2.06 |
| Sharpe ratio after costs | 1.94 | 0.42 | -1.52 | 0.53 |
| Mean, % a year | 15.70 | 10.74 | -4.96 | 11.65 |
| Mean after costs, % a year | 9.05 | 2.39 | -6.66 | 3.01 |
| Volatility, % a year | 4.68 | 5.62 | +0.94 | 5.67 |
| Market beta | 0.07 | 0.07 | +0.00 | 0.07 |
| Turnover a day | | 0.57 | | 0.59 |
| *Attention Factor model, annualized Sharpe ratio in the objective* | | | | |
| Sharpe ratio | 3.35 | 2.58 | -0.77 |  |
| Sharpe ratio after costs | 1.94 | 1.14 | -0.80 |  |
| Mean, % a year | 15.70 | 13.86 | -1.84 |  |
| Mean after costs, % a year | 9.05 | 6.12 | -2.93 |  |
| Volatility, % a year | 4.68 | 5.36 | +0.68 |  |
| Market beta | 0.07 | 0.10 | +0.03 |  |
| Turnover a day | | 0.52 | |  |
| *PCA with the long convolution model, loadings from 60 days* | | | | |
| Sharpe ratio | 2.64 | 2.33 | -0.31 | 2.29 |
| Sharpe ratio after costs | 1.50 | 0.32 | -1.18 | 0.47 |
| Mean, % a year | 14.88 | 8.78 | -6.10 | 9.45 |
| Mean after costs, % a year | 8.42 | 1.21 | -7.21 | 1.93 |
| Volatility, % a year | 5.63 | 3.77 | -1.86 | 4.12 |
| Market beta | 0.09 | 0.04 | -0.05 | 0.03 |
| Turnover a day | | 0.50 | | 0.50 |
| *PCA with the long convolution model, projection* | | | | |
| Sharpe ratio | 2.64 | 2.89 | +0.25 | 3.07 |
| Sharpe ratio after costs | 1.50 | 1.13 | -0.37 | 1.35 |
| Mean, % a year | 14.88 | 12.60 | -2.28 | 13.87 |
| Mean after costs, % a year | 8.42 | 4.89 | -3.53 | 6.10 |
| Volatility, % a year | 5.63 | 4.35 | -1.28 | 4.52 |
| Market beta | 0.09 | 0.06 | -0.03 | 0.06 |
| Turnover a day | | 0.52 | | 0.53 |
| *PCA with an OU threshold rule, loadings from 60 days* | | | | |
| Sharpe ratio | 0.78 | 1.30 | +0.52 | 1.38 |
| Sharpe ratio after costs | -4.15 | -3.74 | +0.41 | -3.29 |
| Mean, % a year | 2.04 | 3.42 | +1.38 | 3.92 |
| Mean after costs, % a year | -10.83 | -9.80 | +1.03 | -9.32 |
| Volatility, % a year | 2.61 | 2.62 | +0.01 | 2.83 |
| Market beta | 0.00 | 0.01 | +0.01 | 0.01 |
| Turnover a day | | 0.95 | | 0.95 |
| *PCA with an OU threshold rule, projection* | | | | |
| Sharpe ratio | 0.78 | 1.30 | +0.52 | 1.16 |
| Sharpe ratio after costs | -4.15 | -3.24 | +0.91 | -3.16 |
| Mean, % a year | 2.04 | 3.84 | +1.80 | 3.61 |
| Mean after costs, % a year | -10.83 | -9.58 | +1.25 | -9.83 |
| Volatility, % a year | 2.61 | 2.95 | +0.34 | 3.11 |
| Market beta | 0.00 | 0.01 | +0.01 | 0.01 |
| Turnover a day | | 0.97 | | 0.97 |

**30 factors**

| | Paper | This round | This round minus paper | Same specification, universe of rounds 1 to 3 |
|---|---|---|---|---|
| *Attention Factor model, objective as printed* | | | | |
| Sharpe ratio | 3.97 | 2.03 | -1.94 | 1.94 |
| Sharpe ratio after costs | 2.28 | -0.03 | -2.31 | 0.02 |
| Mean, % a year | 16.66 | 9.10 | -7.56 | 9.79 |
| Mean after costs, % a year | 9.52 | -0.15 | -9.67 | 0.10 |
| Volatility, % a year | 4.20 | 4.49 | +0.29 | 5.05 |
| Market beta | 0.05 | 0.04 | -0.01 | 0.03 |
| Turnover a day | | 0.63 | | 0.67 |
| *Attention Factor model, annualized Sharpe ratio in the objective* | | | | |
| Sharpe ratio | 3.97 | 3.01 | -0.96 |  |
| Sharpe ratio after costs | 2.28 | 0.96 | -1.32 |  |
| Mean, % a year | 16.66 | 12.20 | -4.46 |  |
| Mean after costs, % a year | 9.52 | 3.87 | -5.65 |  |
| Volatility, % a year | 4.20 | 4.05 | -0.15 |  |
| Market beta | 0.05 | 0.05 | -0.00 |  |
| Turnover a day | | 0.56 | |  |
| *PCA with the long convolution model, loadings from 60 days* | | | | |
| Sharpe ratio | 2.79 | 1.76 | -1.03 | 1.64 |
| Sharpe ratio after costs | 1.57 | -1.38 | -2.95 | -1.33 |
| Mean, % a year | 15.15 | 4.12 | -11.03 | 4.22 |
| Mean after costs, % a year | 8.47 | -3.22 | -11.69 | -3.40 |
| Volatility, % a year | 5.42 | 2.34 | -3.08 | 2.57 |
| Market beta | 0.09 | 0.01 | -0.08 | 0.01 |
| Turnover a day | | 0.48 | | 0.50 |
| *PCA with the long convolution model, projection* | | | | |
| Sharpe ratio | 2.79 | 3.64 | +0.85 | 3.47 |
| Sharpe ratio after costs | 1.57 | 1.19 | -0.38 | 1.26 |
| Mean, % a year | 15.15 | 11.03 | -4.12 | 11.15 |
| Mean after costs, % a year | 8.47 | 3.59 | -4.88 | 4.02 |
| Volatility, % a year | 5.42 | 3.03 | -2.39 | 3.21 |
| Market beta | 0.09 | 0.03 | -0.06 | 0.04 |
| Turnover a day | | 0.50 | | 0.48 |
| *PCA with an OU threshold rule, loadings from 60 days* | | | | |
| Sharpe ratio | 0.18 | 1.07 | +0.89 | 0.90 |
| Sharpe ratio after costs | -6.45 | -5.80 | +0.65 | -5.61 |
| Mean, % a year | 0.40 | 2.02 | +1.62 | 1.80 |
| Mean after costs, % a year | -14.74 | -10.96 | +3.78 | -11.18 |
| Volatility, % a year | 2.24 | 1.89 | -0.35 | 1.99 |
| Market beta | -0.00 | 0.00 | +0.00 | 0.00 |
| Turnover a day | | 0.93 | | 0.93 |
| *PCA with an OU threshold rule, projection* | | | | |
| Sharpe ratio | 0.18 | 1.66 | +1.48 | 1.31 |
| Sharpe ratio after costs | -6.45 | -4.10 | +2.35 | -3.95 |
| Mean, % a year | 0.40 | 3.84 | +3.44 | 3.33 |
| Mean after costs, % a year | -14.74 | -9.47 | +5.27 | -10.01 |
| Volatility, % a year | 2.24 | 2.31 | +0.07 | 2.53 |
| Market beta | -0.00 | 0.00 | +0.00 | 0.01 |
| Turnover a day | | 0.96 | | 0.96 |

The runs on the universe of rounds 1 to 3 are those of round 3 (`results/as_cited/`), with the same settings and the
same seed. For the two rows with a projection they are the runs of round 1 (`results/runs/`), where the residuals
are built the same way. The trained one of the two used batches of 64 days and an annualized Sharpe ratio in its
objective; that objective has one term, so the scale only rescales it.
<!-- results:end -->

## What still differs in the setup

"Same": taken from the paper as printed. "Cited": taken from a source that the paper cites.
"Inferred": reconstructed from indirect evidence. "Ours": a detail the paper does not give,
chosen here. "Unknown": the paper does not say, and more than one reading is run or possible.

| | The paper | This round | Status |
|---|---|---|---|
| Sample and protocol | Daily returns 1990 to 2021; rolling window of 8 years, retrained every year; out of sample 1998 to 2021 | The same | Same |
| Universe | "The 500 largest stocks based on the previous month market capitalization" | The 500 largest by company value among firms with 120 months of returns; ordinary shares, trusts, depositary receipts and partnership units on NYSE, AMEX and Nasdaq | Inferred. 91 of the 108 firms the paper names are members in January 2021; the rule behind 16 of the other 17 is unknown |
| Market value inside the inputs | Not stated | The company value that ranks the universe | Inferred |
| The 35 monthly characteristics | "Internet Appendix of" Chen, Pelger and Zhu | Its definitions as read in the arXiv versions 1 to 3; the final appendix was not read | Cited, with gaps: r2_1 and ST_Rev are one series here and two in the authors' data; Beta has less than five years of data until 1992; the window of SUV and the formula of Spread are not given; Canadian firms report in CAD |
| The 4 daily characteristics | "The previous day and week return and volatility", weekly trading volume | Return of the day, return and standard deviation of five days, dollar volume of five days | Ours |
| Normalization and missing values | Rank quantiles; last value, else the cross-sectional median | Ranks in (-0.5, 0.5), average rank for ties; last value for 12 months or 5 days, else the median | Same in substance |
| The 40 inputs equal for all stocks | Medians and the risk-free rate, "to keep the level information" | Included. They cancel in the softmax of the paper's equation and change nothing | Same equation; its stated purpose cannot be met |
| Factor layer, loadings, residuals | Equations of Section 3 | As printed | Same |
| Ridge penalty of the loadings | Not reported | 1e-5 (chosen on validation in round 1) | Unknown |
| Long convolution model | One layer, hidden dimension 32, lookback 30, geometric-decay initialization, squash 0.001, dropout 0.1 | The same values. How the residuals enter the 32 channels and leave as one number is ours (affine lift, GELU, linear read-out at the last position) | Same values; details ours |
| Objective | Sharpe ratio after costs plus 100 times the explained variance; no scale printed | Run per day (as printed) and annualized | Unknown which the authors used |
| Explained-variance term | Average over the stocks of 1 - Var(residual) / Var(return) | The same, over the stocks with a return on half of the days of a batch | Same; eligibility ours |
| Batch | Not stated; 30 "passes over the data" | 125 consecutive days, one Sharpe ratio per batch, as in the predecessor's code | Cited source; the paper's own value is unknown |
| Costs | 5 basis points on turnover, 1 on short positions | The same. The return of a weight is charged with the trade that established it (the printed formula pairs it with the next trade) | Same; timing of one day ours |
| Optimizer | Adam with the citation of AdamW, learning rate 0.003, weight decay 0.05 on the sequence model, 30 epochs | AdamW with these values | Same |
| Seeds | Several, averaged; standard deviation of the Sharpe ratio 0.13 | One (a second for one run) | Differs, for compute |
| PCA residuals of the benchmarks | "PCA using the past 252 trading days", "in the spirit of" the predecessor | Run both ways: loadings from 60 days (the predecessor's code) and projection | Unknown. The predecessor's recipe cannot be the paper's for its row with 100 factors |
| OU threshold rule | "The same implementation as" the predecessor | AR(1) on the cumulated residuals of 30 days; thresholds 1.25 and R-squared 0.25, the predecessor paper's values; the code is ours, the rule is not in the public code | Cited values; code ours |
| Returns | CRSP daily returns | CRSP daily returns with delisting returns (-30 percent where CRSP has none for a forced delisting); no return on a day with a volume of zero or on the day after; a missing return counts as zero and blocks trading in the security for 30 days; total returns are traded and the risk-free rate enters the Sharpe ratio of the objective | Details ours (the predecessor trades excess returns) |
| Market row of Table 2 | "The equally weighted market portfolio": 8.61 percent mean, 20.37 volatility | The equally weighted universe: 11.61 and 19.62 (9.83 in excess of the risk-free rate) | Not reproduced. Which stocks the row holds, and whether it is in excess of the risk-free rate, is not stated |
| Scope | 1 to 100 factors; Table 3; Figures 2 to 4 | 8 and 30 factors in this round; 100 factors and Table 3 without the past returns in round 1 only (`RESULTS.md`, "Open choices tested") | Partly run |
| Data vintage | Not stated (2025 or earlier) | WRDS files of November 2026 | Differs |

## The market row

<!-- market_row:start -->
Out of sample, 1998 to 2021, from daily returns, percent a year. "Excess": minus the risk-free rate.
The note of the paper's Table 2 calls its row "the equally weighted market portfolio".

| Portfolio | Mean | Volatility | Mean over volatility |
|---|---|---|---|
| **Paper, Table 2, "Market"** | **8.61** | **20.37** | **0.42** |
| Universe of this round, equally weighted | 11.61 | 19.62 | 0.59 |
| ... excess | 9.83 | 19.62 | 0.50 |
| Universe of this round, weighted by the ranking value | 10.26 | 19.38 | 0.53 |
| ... excess | 8.48 | 19.38 | 0.44 |
| Universe of rounds 1 to 3, equally weighted | 11.76 | 20.55 | 0.57 |
| ... excess | 9.98 | 20.55 | 0.49 |
| Universe of rounds 1 to 3, weighted by the ranking value | 10.92 | 19.89 | 0.55 |
| ... excess | 9.14 | 19.89 | 0.46 |
| S&P 500 index, without dividends | 8.57 | 19.60 | 0.44 |
| CRSP value-weighted index, with dividends | 10.22 | 19.50 | 0.52 |
| CRSP equally weighted index, with dividends | 16.16 | 17.40 | 0.93 |
| Fama-French market, excess | 9.01 | 19.76 | 0.46 |
<!-- market_row:end -->

The note of the paper's Table 2 says: "The equally weighted market portfolio is provided for
reference." It does not say which stocks the portfolio holds, nor whether the risk-free rate is
subtracted.

Neither of our universes reproduces the row. Equally weighted, this round's universe earns
11.61 percent a year with a volatility of 19.62, and the universe of rounds 1 to 3 earns 11.76
with 20.55. In excess of the risk-free rate the means are 9.83 and 9.98. The paper's 8.61 is
below all four, by 1.2 points at the least. In volatility the old universe (20.55) is closer to
the paper's 20.37 than the new one (19.62). The history rule of this round lowers the volatility, so the row
speaks against that rule, while the ages of the named firms speak for it. We cannot reconcile
the two.

Two other series are close to the paper's mean: the S&P 500 index without dividends (8.57) and
the value-weighted universe of this round in excess of the risk-free rate (8.48). Neither is
equally weighted, and neither has the paper's volatility (19.60 and 19.38). The CRSP equally
weighted index of all stocks is far off (16.16 and 17.40).

So the named firms say that this round's universe is closer to the paper's, and the market row
does not confirm it.

## Reproduce

From `replication/`. The first three stages need a WRDS token; the rest runs from the files.

```bash
.venv/bin/python scripts/replicate_company_universe.py download   # 8 minutes: Compustat values, names, links
.venv/bin/python scripts/replicate_company_universe.py universe   # 2 minutes: data/company/universe.parquet, rule checks
.venv/bin/python scripts/replicate_company_universe.py data       # 1 minute: daily data of 253 new securities
.venv/bin/python scripts/replicate_company_universe.py build      # 3 minutes: panel, inputs, both sets of PCA residuals
.venv/bin/python scripts/replicate_company_universe.py main --workers 5   # 13 runs, 2.5 hours on 4 cores
.venv/bin/python scripts/replicate_company_universe.py report     # tables from the saved files; no data needed
```

Files: `scripts/replicate_company_universe.py` (the round), `scripts/company_comparison.py`
(tables), `results/company/<run>/` (metrics and daily returns of each run),
`results/company/checks/` (rule comparison, market row, delistings of January 2021), `data/company/` and `runs/company/`
(data and full run output, not committed). The raw-data archive linked in `README.md` does not
hold the files of this round; the `download` and `data` stages fetch them.
