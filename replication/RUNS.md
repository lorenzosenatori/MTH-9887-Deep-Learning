# All runs: settings, results, commits, and the source of each setting

> The commit hashes and branch names below refer to the history of the group's private
> development repository. This public repository starts from a single snapshot of that work.

Status of 25 November 2026. This file lists every run of the replication of Epstein, Wang, Choi
and Pelger (2025), "Attention Factors for Statistical Arbitrage": what the run covers, the
settings in which the runs differ, the result, and the commit that holds it. Two lists follow:
where each setting that changed comes from, and where the settings that never changed come from.

The tables are written by `scripts/runs_table.py` from the `metrics.json` of each run and from the
git history. The two lists of sources are written by hand. The reading of the results is in other
files: `README_ISSUES.md` (rounds 1 to 3), `DIFFERENCES.md` (round 4) and `RESULTS.md` (round 1 and
the six runs called round 1b here).

**How to read the tables**

- **ID**: round, model, number of factors, variant, seed. AF is the Attention Factor model, PCA is
  PCA with the long convolution model, OU is PCA with an OU threshold rule. `3-PCA30-s1` is round 3,
  the PCA benchmark, 30 factors, seed 1. An ID that starts with V is a validation run.
- **Round 1b** is this file's name for the six runs made on the branch `experiments/open-problems`.
  The other documents call them "open choices tested" or "variants". They reached `main` with the
  merge `cc4e0a3` of 25 November.
- **SR** is the annualized Sharpe ratio of the daily returns, before and after the paper's costs
  (5 basis points per unit traded, 1 per unit held short). Out of sample is January 1998 to
  December 2021 (6,040 days). Validation is 1996 to 1997, with six training years before each year.
- **Paper** is the same row of the paper's Table 2, or of its Table 3 for the run without past
  returns.
- **"-"** means that the setting does not act on that model.
- **Commit times** are US Eastern time (UTC-5), as the commits record them. The file times of the
  runs that were never committed are those of the machine that made them, in the same zone.

## The rounds

<!-- rounds:start -->
| Round | What it covers | Script | Results | Runs: out of sample + validation | Commits (hash, time) |
|---|---|---|---|---|---|
| 1 | The first implementation; its open choices were set on the validation years. The numbers on the slides. | `scripts/run.py` | `results/runs/` | 14 + 10 | `5525cf7` 22 Nov 02:04; `7ac23ef` 22 Nov 02:51 |
| 1b | Six more runs on the specification of round 1, made on the branch `experiments/open-problems`: 100 factors, one row of Table 3, the residual history of the decision date, and PCA of the correlation matrix. | `scripts/run.py --pca, --lookback, --drop, --k 100` | `results/runs/` | 6 + 0 | `3c4a3d5` 23 Nov 03:28; `c51740a` 23 Nov 04:45 |
| 2a | Round 1 with the Sharpe ratio per day in the objective, as the paper prints it. Nothing else changed. | `scripts/replicate_daily_sharpe.py` | `results/daily_sharpe/` | 3 + 0 | `98ed050` 23 Nov 02:53 |
| 2b | Round 2a with the objective over the whole training window as one batch and without the bound on the explained-variance term. Not a valid test: 30 optimizer steps a year. | `scripts/replicate_adjusted_loss.py` | `results/adjusted_loss/` | 3 + 3 | `f0d1c88` 23 Nov 10:22 |
| 3 | Choices taken from the sources that the paper cites: Sharpe ratio per day, batches of 125 days, no bound, three inputs as cited, PCA residuals with loadings from 60 days. | `scripts/replicate_as_cited.py` | `results/as_cited/` | 8 + 13 | `ea6bf50` 23 Nov 15:01 |
| 4 | Round 3 on another universe: each month the 500 largest by the value of the whole company, with partnership units and 120 months of history. The rule was inferred from the 108 firms that the paper's figures name; no firm is put in by name. The objective and the PCA residuals are each run both ways. | `scripts/replicate_company_universe.py` | `results/company/` | 13 + 0 | `f128b84` 25 Nov 17:27 |
<!-- rounds:end -->

The commit time of round 4 is not the time of its runs: its result files date from 23 November,
18:10 to 20:04.

## The settings that differ between runs

Each is a column of the tables below. The section "Where each setting that changed comes from"
gives the source of every value.

| Column | Values | What it is |
|---|---|---|
| Number of factors, seed | 1 to 100; 0 or 1 | Part of the ID |
| Sharpe ratio in the objective | per day, annualized | The scale of the Sharpe ratio before 100 times the explained variance is added. Annualized is per day times the square root of 252. It acts on the Attention Factor model only: the objective of the PCA benchmark has one term |
| Batch, days | 64, 125, whole window | Consecutive days of one optimizer step; one Sharpe ratio and one explained variance per batch |
| Bound | -1, none | Lower bound on one security's explained-variance term |
| Inputs | round 1, cited, no past returns | "Cited": Beta, Variance and Resid_Var as the paper's source defines them. "No past returns": the eight inputs of that group left out, a row of the paper's Table 3 |
| Residual history | own, current | The 30 residuals that the trading policy reads. "Own": each day's residual from that day's factor model. "Current": the decision date's residual portfolio applied to the returns of the last 30 days |
| PCA residuals | covariance; correlation; correlation, 60-day loadings; correlation, 252-day loadings | How the benchmarks build residuals. "Covariance": eigenvectors of the covariance matrix as weights and as loadings, an orthogonal projection. "Correlation": eigenvectors V of the correlation matrix, weights V / sigma, loadings sigma V. "60-day loadings": the same weights, loadings from a regression on the last 60 days. "252-day loadings": that regression on all 252 days |
| Universe | security value, company value | "Security value": the 500 largest securities by CRSP capitalization. "Company value": the 500 largest by the value of the whole company, with partnership units and a history of 120 months. Two more things change with it: the market value inside the inputs, and a day without a trade has no return |
| Ridge penalty, weight on explained variance | 1e-4 to 1e-7; 0 or 100 | Changed in validation runs only. Every run out of sample has 1e-5 and 100 |

## Runs out of sample, 1998 to 2021

<!-- main:start -->
| ID | Covers | Sharpe ratio in the objective | Batch, days | Bound | Inputs | Residual history | PCA residuals | Universe | SR before / after costs | Paper | Mean % before / after costs | Volatility % | Turnover a day | Folder under `results/` | Commit (hash, time) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1-AF8-s0** | Attention Factor model, K = 8, seed 0 | annualized | 64 | -1 | round 1 | own | - | security value | 2.66 / 1.15 | 3.35 / 1.94 | 14.38 / 6.21 | 5.41 | 0.56 | `runs/attention_K8_seed0` | `5525cf7` 22 Nov 02:04 |
| **1-AF30-s0** | Attention Factor model, K = 30, seed 0 | annualized | 64 | -1 | round 1 | own | - | security value | 2.97 / 0.94 | 3.97 / 2.28 | 12.62 / 4.00 | 4.26 | 0.59 | `runs/attention_K30_seed0` | `5525cf7` 22 Nov 02:04 |
| **1-AF30-s1** | Attention Factor model, K = 30, seed 1 | annualized | 64 | -1 | round 1 | own | - | security value | 3.04 / 1.03 | 3.97 / 2.28 | 13.01 / 4.41 | 4.27 | 0.58 | `runs/attention_K30_seed1` | `5525cf7` 22 Nov 02:04 |
| **1-PCA8-s0** | PCA with the long convolution model, K = 8, seed 0 | per day | 64 | - | - | own | covariance | security value | 3.07 / 1.35 | 2.64 / 1.50 | 13.87 / 6.10 | 4.52 | 0.53 | `runs/pca_K8_seed0` | `5525cf7` 22 Nov 02:04 |
| **1-PCA30-s0** | PCA with the long convolution model, K = 30, seed 0 | per day | 64 | - | - | own | covariance | security value | 3.47 / 1.26 | 2.79 / 1.57 | 11.15 / 4.02 | 3.21 | 0.48 | `runs/pca_K30_seed0` | `5525cf7` 22 Nov 02:04; `7ac23ef` 22 Nov 02:51 (rerun) |
| **1-PCA30-s1** | PCA with the long convolution model, K = 30, seed 1 | per day | 64 | - | - | own | covariance | security value | 3.46 / 1.27 | 2.79 / 1.57 | 11.09 / 4.02 | 3.20 | 0.47 | `runs/pca_K30_seed1` | `5525cf7` 22 Nov 02:04 |
| **1-OU1** | PCA with an OU threshold rule, K = 1 | - | - | - | - | own | covariance | security value | 0.22 / -2.42 | 0.40 / -2.54 | 1.14 / -12.48 | 5.17 | 0.98 | `runs/ou_K1` | `5525cf7` 22 Nov 02:04; `7ac23ef` 22 Nov 02:51 (rerun) |
| **1-OU3** | PCA with an OU threshold rule, K = 3 | - | - | - | - | own | covariance | security value | 1.04 / -2.43 | 1.26 / -2.72 | 4.05 / -9.48 | 3.90 | 0.97 | `runs/ou_K3` | `5525cf7` 22 Nov 02:04; `7ac23ef` 22 Nov 02:51 (rerun) |
| **1-OU5** | PCA with an OU threshold rule, K = 5 | - | - | - | - | own | covariance | security value | 1.17 / -2.69 | 0.99 / -3.44 | 4.08 / -9.38 | 3.49 | 0.97 | `runs/ou_K5` | `5525cf7` 22 Nov 02:04; `7ac23ef` 22 Nov 02:51 (rerun) |
| **1-OU8** | PCA with an OU threshold rule, K = 8 | - | - | - | - | own | covariance | security value | 1.16 / -3.16 | 0.78 / -4.15 | 3.61 / -9.83 | 3.11 | 0.97 | `runs/ou_K8` | `5525cf7` 22 Nov 02:04; `7ac23ef` 22 Nov 02:51 (rerun) |
| **1-OU10** | PCA with an OU threshold rule, K = 10 | - | - | - | - | own | covariance | security value | 1.02 / -3.49 | 0.80 / -4.32 | 3.04 / -10.38 | 2.98 | 0.97 | `runs/ou_K10` | `5525cf7` 22 Nov 02:04; `7ac23ef` 22 Nov 02:51 (rerun) |
| **1-OU15** | PCA with an OU threshold rule, K = 15 | - | - | - | - | own | covariance | security value | 1.33 / -3.41 | 0.51 / -5.24 | 3.75 / -9.63 | 2.82 | 0.96 | `runs/ou_K15` | `5525cf7` 22 Nov 02:04; `7ac23ef` 22 Nov 02:51 (rerun) |
| **1-OU30** | PCA with an OU threshold rule, K = 30 | - | - | - | - | own | covariance | security value | 1.31 / -3.95 | 0.18 / -6.45 | 3.33 / -10.01 | 2.53 | 0.96 | `runs/ou_K30` | `5525cf7` 22 Nov 02:04; `7ac23ef` 22 Nov 02:51 (rerun) |
| **1-OU100** | PCA with an OU threshold rule, K = 100 | - | - | - | - | own | covariance | security value | 0.65 / -6.60 | -0.35 / -7.05 | 1.19 / -12.17 | 1.84 | 0.96 | `runs/ou_K100` | `5525cf7` 22 Nov 02:04; `7ac23ef` 22 Nov 02:51 (rerun) |
| **1b-AF30cur-s0** | Attention Factor model, K = 30, seed 0 | annualized | 64 | -1 | round 1 | current | - | security value | 2.81 / 0.75 | 3.97 / 2.28 | 12.49 / 3.31 | 4.44 | 0.63 | `runs/attention_K30_look-current_seed0` | `3c4a3d5` 23 Nov 03:28 |
| **1b-AF30nopast-s0** | Attention Factor model, K = 30, seed 0 (Table 3: past returns removed) | annualized | 64 | -1 | round 1, no past returns | own | - | security value | 2.59 / 1.02 | 1.50 / 0.59 | 14.02 / 5.52 | 5.41 | 0.57 | `runs/attention_K30_no-past_returns_seed0` | `3c4a3d5` 23 Nov 03:28; `c51740a` 23 Nov 04:45 (saved settings rewritten, no rerun) |
| **1b-AF100-s0** | Attention Factor model, K = 100, seed 0 | annualized | 64 | -1 | round 1 | own | - | security value | 3.04 / 0.33 | 4.52 / 2.19 | 10.21 / 1.11 | 3.35 | 0.62 | `runs/attention_K100_seed0` | `3c4a3d5` 23 Nov 03:28; `c51740a` 23 Nov 04:45 (saved settings rewritten, no rerun) |
| **1b-PCA30corr-s0** | PCA with the long convolution model, K = 30, seed 0 | per day | 64 | - | - | own | correlation | security value | 2.62 / 0.38 | 2.79 / 1.57 | 9.19 / 1.33 | 3.50 | 0.52 | `runs/pca_K30_corr_seed0` | `3c4a3d5` 23 Nov 03:28 |
| **1b-PCA30corrcur-s0** | PCA with the long convolution model, K = 30, seed 0 | per day | 64 | - | - | current | correlation | security value | 2.63 / 0.43 | 2.79 / 1.57 | 9.40 / 1.52 | 3.57 | 0.52 | `runs/pca_K30_corr_look-current_seed0` | `3c4a3d5` 23 Nov 03:28 |
| **1b-PCA30cur-s0** | PCA with the long convolution model, K = 30, seed 0 | per day | 64 | - | - | current | covariance | security value | 3.52 / 1.33 | 2.79 / 1.57 | 11.40 / 4.29 | 3.24 | 0.47 | `runs/pca_K30_look-current_seed0` | `3c4a3d5` 23 Nov 03:28 |
| **2a-AF8-s0** | Attention Factor model, K = 8, seed 0 | per day | 64 | -1 | round 1 | own | - | security value | 2.10 / 0.42 | 3.35 / 1.94 | 12.13 / 2.45 | 5.78 | 0.67 | `daily_sharpe/attention_K8_seed0` | `98ed050` 23 Nov 02:53 |
| **2a-AF30-s0** | Attention Factor model, K = 30, seed 0 | per day | 64 | -1 | round 1 | own | - | security value | 2.22 / 0.19 | 3.97 / 2.28 | 11.00 / 0.96 | 4.96 | 0.70 | `daily_sharpe/attention_K30_seed0` | `98ed050` 23 Nov 02:53 |
| **2a-AF30-s1** | Attention Factor model, K = 30, seed 1 | per day | 64 | -1 | round 1 | own | - | security value | 1.99 / -0.02 | 3.97 / 2.28 | 10.08 / -0.11 | 5.07 | 0.71 | `daily_sharpe/attention_K30_seed1` | `98ed050` 23 Nov 02:53 |
| **2b-AF8-s0** | Attention Factor model, K = 8, seed 0 | per day | whole window | none | round 1 | own | - | security value | 1.47 / -0.90 | 3.35 / 1.94 | 8.18 / -5.02 | 5.55 | 0.95 | `adjusted_loss/attention_K8_seed0` | `f0d1c88` 23 Nov 10:22 |
| **2b-AF30-s0** | Attention Factor model, K = 30, seed 0 | per day | whole window | none | round 1 | own | - | security value | 1.33 / -1.87 | 3.97 / 2.28 | 5.91 / -8.31 | 4.43 | 1.03 | `adjusted_loss/attention_K30_seed0` | `f0d1c88` 23 Nov 10:22 |
| **2b-AF30-s1** | Attention Factor model, K = 30, seed 1 | per day | whole window | none | round 1 | own | - | security value | 1.61 / -1.58 | 3.97 / 2.28 | 7.16 / -6.99 | 4.44 | 1.02 | `adjusted_loss/attention_K30_seed1` | `f0d1c88` 23 Nov 10:22 |
| **3-AF8-s0** | Attention Factor model, K = 8, seed 0 | per day | 125 | none | cited | own | - | security value | 2.06 / 0.53 | 3.35 / 1.94 | 11.65 / 3.01 | 5.67 | 0.59 | `as_cited/attention_K8_seed0` | `ea6bf50` 23 Nov 15:01 |
| **3-AF30-s0** | Attention Factor model, K = 30, seed 0 | per day | 125 | none | cited | own | - | security value | 1.94 / 0.02 | 3.97 / 2.28 | 9.79 / 0.10 | 5.05 | 0.67 | `as_cited/attention_K30_seed0` | `ea6bf50` 23 Nov 15:01 |
| **3-AF30-s1** | Attention Factor model, K = 30, seed 1 | per day | 125 | none | cited | own | - | security value | 2.15 / 0.16 | 3.97 / 2.28 | 10.45 / 0.76 | 4.86 | 0.67 | `as_cited/attention_K30_seed1` | `ea6bf50` 23 Nov 15:01 |
| **3-PCA8-s0** | PCA with the long convolution model, K = 8, seed 0 | per day | 125 | - | - | own | correlation, 60-day loadings | security value | 2.29 / 0.47 | 2.64 / 1.50 | 9.45 / 1.93 | 4.12 | 0.50 | `as_cited/pca_K8_seed0` | `ea6bf50` 23 Nov 15:01 |
| **3-PCA30-s0** | PCA with the long convolution model, K = 30, seed 0 | per day | 125 | - | - | own | correlation, 60-day loadings | security value | 1.64 / -1.33 | 2.79 / 1.57 | 4.22 / -3.40 | 2.57 | 0.50 | `as_cited/pca_K30_seed0` | `ea6bf50` 23 Nov 15:01 |
| **3-PCA30-s1** | PCA with the long convolution model, K = 30, seed 1 | per day | 125 | - | - | own | correlation, 60-day loadings | security value | 1.82 / -1.20 | 2.79 / 1.57 | 4.59 / -3.00 | 2.52 | 0.50 | `as_cited/pca_K30_seed1` | `ea6bf50` 23 Nov 15:01 |
| **3-OU8** | PCA with an OU threshold rule, K = 8 | - | - | - | - | own | correlation, 60-day loadings | security value | 1.38 / -3.29 | 0.78 / -4.15 | 3.92 / -9.32 | 2.83 | 0.95 | `as_cited/ou_K8` | `ea6bf50` 23 Nov 15:01 |
| **3-OU30** | PCA with an OU threshold rule, K = 30 | - | - | - | - | own | correlation, 60-day loadings | security value | 0.90 / -5.61 | 0.18 / -6.45 | 1.80 / -11.18 | 1.99 | 0.93 | `as_cited/ou_K30` | `ea6bf50` 23 Nov 15:01 |
| **4-AF8-s0** | Attention Factor model, K = 8, seed 0 | per day | 125 | none | cited | own | - | company value | 1.91 / 0.42 | 3.35 / 1.94 | 10.74 / 2.39 | 5.62 | 0.57 | `company/attention_K8_seed0` | `f128b84` 25 Nov 17:27 |
| **4-AF8ann-s0** | Attention Factor model, K = 8, seed 0 | annualized | 125 | none | cited | own | - | company value | 2.58 / 1.14 | 3.35 / 1.94 | 13.86 / 6.12 | 5.36 | 0.52 | `company/attention_annualized_K8_seed0` | `f128b84` 25 Nov 17:27 |
| **4-AF30-s0** | Attention Factor model, K = 30, seed 0 | per day | 125 | none | cited | own | - | company value | 2.03 / -0.03 | 3.97 / 2.28 | 9.10 / -0.15 | 4.49 | 0.63 | `company/attention_K30_seed0` | `f128b84` 25 Nov 17:27 |
| **4-AF30-s1** | Attention Factor model, K = 30, seed 1 | per day | 125 | none | cited | own | - | company value | 2.21 / 0.08 | 3.97 / 2.28 | 9.62 / 0.35 | 4.35 | 0.64 | `company/attention_K30_seed1` | `f128b84` 25 Nov 17:27 |
| **4-AF30ann-s0** | Attention Factor model, K = 30, seed 0 | annualized | 125 | none | cited | own | - | company value | 3.01 / 0.96 | 3.97 / 2.28 | 12.20 / 3.87 | 4.05 | 0.56 | `company/attention_annualized_K30_seed0` | `f128b84` 25 Nov 17:27 |
| **4-PCA8-s0** | PCA with the long convolution model, K = 8, seed 0 | per day | 125 | - | - | own | correlation, 60-day loadings | company value | 2.33 / 0.32 | 2.64 / 1.50 | 8.78 / 1.21 | 3.77 | 0.50 | `company/pca_K8_seed0` | `f128b84` 25 Nov 17:27 |
| **4-PCA8cov-s0** | PCA with the long convolution model, K = 8, seed 0 | per day | 125 | - | - | own | covariance | company value | 2.89 / 1.13 | 2.64 / 1.50 | 12.60 / 4.89 | 4.35 | 0.52 | `company/pcaproj_K8_seed0` | `f128b84` 25 Nov 17:27 |
| **4-PCA30-s0** | PCA with the long convolution model, K = 30, seed 0 | per day | 125 | - | - | own | correlation, 60-day loadings | company value | 1.76 / -1.38 | 2.79 / 1.57 | 4.12 / -3.22 | 2.34 | 0.48 | `company/pca_K30_seed0` | `f128b84` 25 Nov 17:27 |
| **4-PCA30cov-s0** | PCA with the long convolution model, K = 30, seed 0 | per day | 125 | - | - | own | covariance | company value | 3.64 / 1.19 | 2.79 / 1.57 | 11.03 / 3.59 | 3.03 | 0.50 | `company/pcaproj_K30_seed0` | `f128b84` 25 Nov 17:27 |
| **4-OU8** | PCA with an OU threshold rule, K = 8 | - | - | - | - | own | correlation, 60-day loadings | company value | 1.30 / -3.74 | 0.78 / -4.15 | 3.42 / -9.80 | 2.62 | 0.95 | `company/ou_K8` | `f128b84` 25 Nov 17:27 |
| **4-OU8cov** | PCA with an OU threshold rule, K = 8 | - | - | - | - | own | covariance | company value | 1.30 / -3.24 | 0.78 / -4.15 | 3.84 / -9.58 | 2.95 | 0.97 | `company/ouproj_K8` | `f128b84` 25 Nov 17:27 |
| **4-OU30** | PCA with an OU threshold rule, K = 30 | - | - | - | - | own | correlation, 60-day loadings | company value | 1.07 / -5.80 | 0.18 / -6.45 | 2.02 / -10.96 | 1.89 | 0.93 | `company/ou_K30` | `f128b84` 25 Nov 17:27 |
| **4-OU30cov** | PCA with an OU threshold rule, K = 30 | - | - | - | - | own | covariance | company value | 1.66 / -4.10 | 0.18 / -6.45 | 3.84 / -9.47 | 2.31 | 0.96 | `company/ouproj_K30` | `f128b84` 25 Nov 17:27 |
<!-- main:end -->

Means over the seeds of a model, where two were run:

<!-- averages:start -->
| Runs | Covers | Mean SR before / after costs | Paper |
|---|---|---|---|
| **1-AF30-s0** and **1-AF30-s1** | Attention Factor model, K = 30 | 3.00 / 0.99 | 3.97 / 2.28 |
| **1-PCA30-s0** and **1-PCA30-s1** | PCA with the long convolution model, K = 30 | 3.47 / 1.26 | 2.79 / 1.57 |
| **2a-AF30-s0** and **2a-AF30-s1** | Attention Factor model, K = 30 | 2.10 / 0.09 | 3.97 / 2.28 |
| **2b-AF30-s0** and **2b-AF30-s1** | Attention Factor model, K = 30 | 1.47 / -1.73 | 3.97 / 2.28 |
| **3-AF30-s0** and **3-AF30-s1** | Attention Factor model, K = 30 | 2.05 / 0.09 | 3.97 / 2.28 |
| **3-PCA30-s0** and **3-PCA30-s1** | PCA with the long convolution model, K = 30 | 1.73 / -1.26 | 2.79 / 1.57 |
| **4-AF30-s0** and **4-AF30-s1** | Attention Factor model, K = 30 | 2.12 / 0.02 | 3.97 / 2.28 |
<!-- averages:end -->

Notes on single rows:

- **Round 2b is not a valid test of its question.** Its runs take 30 optimizer steps a year where
  the others take about 950 (`README_ISSUES.md`, issue 7).
- **The reruns of round 1.** The nine benchmark runs that predated a speed-up of the PCA code were
  run again with the committed code (`7ac23ef`). The OU runs came out equal to four decimals. The
  PCA run with 30 factors moved in the second decimal of its mean return (11.13 to 11.15 percent)
  and by 0.002 in its Sharpe ratio after costs.
- **"Saved settings rewritten" in round 1b.** Commit `c51740a` took two settings that were at their
  default value out of the saved configuration of two runs. No run was made again and no number
  changed.

## Validation runs, 1996 to 1997

All of them have seed 0, the residual history "own" and the universe by security value.

<!-- validation:start -->
| ID | Covers (what the run tests) | Model | Sharpe ratio in the objective | Batch, days | Bound | Inputs | PCA residuals | Ridge penalty | Weight on explained variance | SR before / after costs | Mean after costs % | Volatility % | Turnover a day | Folder under `results/` | Commit (hash, time) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V1-AF30** | The specification of round 1 | AF, K = 30 | annualized | 64 | -1 | round 1 | - | 1e-05 | 100 | 10.91 / 7.50 | 20.90 | 2.79 | 0.66 | `runs/val_att_K30` | `5525cf7` 22 Nov 02:04 |
| **V1-AF30-day** | ... with the Sharpe ratio per day (the printed objective) | AF, K = 30 | per day | 64 | -1 | round 1 | - | 1e-05 | 100 | 8.24 / 5.17 | 16.49 | 3.19 | 0.68 | `runs/val_att_K30_sr-per-day` | `5525cf7` 22 Nov 02:04 |
| **V1-AF30-lam0** | ... without the explained-variance term | AF, K = 30 | annualized | 64 | -1 | round 1 | - | 1e-05 | 0 | 10.09 / 6.88 | 20.65 | 3.01 | 0.67 | `runs/val_att_K30_lambda0` | `5525cf7` 22 Nov 02:04 |
| **V1-AF30-r4** | ... with a ridge penalty of 1e-4 | AF, K = 30 | annualized | 64 | -1 | round 1 | - | 0.0001 | 100 | 10.59 / 7.45 | 21.95 | 2.95 | 0.64 | `runs/val_att_K30_ridge1e-4` | `5525cf7` 22 Nov 02:04 |
| **V1-AF30-r6** | ... with a ridge penalty of 1e-6 | AF, K = 30 | annualized | 64 | -1 | round 1 | - | 1e-06 | 100 | 10.65 / 7.15 | 19.54 | 2.74 | 0.66 | `runs/val_att_K30_ridge1e-6` | `5525cf7` 22 Nov 02:04 |
| **V1-AF30-r7** | ... with a ridge penalty of 1e-7 | AF, K = 30 | annualized | 64 | -1 | round 1 | - | 1e-07 | 100 | 10.75 / 7.21 | 19.62 | 2.73 | 0.67 | `runs/val_att_K30_ridge1e-7` | `5525cf7` 22 Nov 02:04 |
| **V1-AF8** | The specification of round 1, 8 factors | AF, K = 8 | annualized | 64 | -1 | round 1 | - | 1e-05 | 100 | 8.98 / 6.53 | 23.83 | 3.65 | 0.62 | `runs/val_att_K8` | `5525cf7` 22 Nov 02:04 |
| **V1-PCA30** | The benchmark of round 1 | PCA, K = 30 | per day | 64 | - | - | covariance | - | - | 9.11 / 6.00 | 17.00 | 2.84 | 0.61 | `runs/val_pca_K30` | `5525cf7` 22 Nov 02:04 |
| **V1-PCA8** | The benchmark of round 1, 8 factors | PCA, K = 8 | per day | 64 | - | - | covariance | - | - | 8.70 / 6.04 | 21.56 | 3.58 | 0.66 | `runs/val_pca_K8` | `5525cf7` 22 Nov 02:04 |
| **V1-OU8** | The OU rule of round 1, 8 factors | OU, K = 8 | - | - | - | - | covariance | - | - | 3.92 / -0.49 | -1.52 | 3.13 | 0.99 | `runs/val_ou_K8` | `5525cf7` 22 Nov 02:04 |
| **V2b-AF30-nobound** | Round 2a without the bound | AF, K = 30 | per day | 64 | none | round 1 | - | 1e-05 | 100 | 8.58 / 5.39 | 16.62 | 3.09 | 0.68 | `adjusted_loss/validation/short_uncapped` | `f0d1c88` 23 Nov 10:22 |
| **V2b-AF30-window** | Round 2a with the whole window as one batch | AF, K = 30 | per day | whole window | -1 | round 1 | - | 1e-05 | 100 | 4.89 / -0.98 | -2.70 | 2.75 | 1.18 | `adjusted_loss/validation/full_capped` | `f0d1c88` 23 Nov 10:22 |
| **V2b-AF30-window-nobound** | The specification of round 2b | AF, K = 30 | per day | whole window | none | round 1 | - | 1e-05 | 100 | 4.89 / -0.98 | -2.70 | 2.75 | 1.18 | `adjusted_loss/validation/full_uncapped` | `f0d1c88` 23 Nov 10:22 |
| **V3-AF30-b125** | Round 2a with batches of 125 days | AF, K = 30 | per day | 125 | -1 | round 1 | - | 1e-05 | 100 | 8.71 / 5.56 | 16.67 | 3.00 | 0.65 | `as_cited/validation/att_b125_floor` | `ea6bf50` 23 Nov 15:01 |
| **V3-AF30-b125-nobound** | ... and without the bound | AF, K = 30 | per day | 125 | none | round 1 | - | 1e-05 | 100 | 8.60 / 5.45 | 16.60 | 3.05 | 0.66 | `as_cited/validation/att_b125_nofloor` | `ea6bf50` 23 Nov 15:01 |
| **V3-AF30-cited** | Round 2a with the cited inputs | AF, K = 30 | per day | 64 | -1 | cited | - | 1e-05 | 100 | 8.09 / 5.02 | 15.91 | 3.17 | 0.67 | `as_cited/validation/att_b64_floor_cited` | `ea6bf50` 23 Nov 15:01 |
| **V3-AF30** | The specification of round 3 | AF, K = 30 | per day | 125 | none | cited | - | 1e-05 | 100 | 8.50 / 5.36 | 16.32 | 3.05 | 0.66 | `as_cited/validation/att_b125_nofloor_cited` | `ea6bf50` 23 Nov 15:01 |
| **V3-AF30-ann** | The specification of round 3 with the annualized Sharpe ratio | AF, K = 30 | annualized | 125 | none | cited | - | 1e-05 | 100 | 10.26 / 7.13 | 21.09 | 2.96 | 0.64 | `as_cited/validation/att_b125_nofloor_cited_annualized` | `ea6bf50` 23 Nov 15:01 |
| **V3-AF8** | The specification of round 3, 8 factors | AF, K = 8 | per day | 125 | none | cited | - | 1e-05 | 100 | 7.75 / 5.17 | 18.59 | 3.60 | 0.64 | `as_cited/validation/att_K8_b125_nofloor_cited` | `ea6bf50` 23 Nov 15:01 |
| **V3-PCA30-cov** | The benchmark of round 1 with batches of 125 days | PCA, K = 30 | per day | 125 | - | - | covariance | - | - | 9.02 / 5.90 | 16.48 | 2.81 | 0.60 | `as_cited/validation/pca_b125_projection` | `ea6bf50` 23 Nov 15:01 |
| **V3-PCA30-reg252** | Correlation matrix, loadings from all 252 days | PCA, K = 30 | per day | 125 | - | - | correlation, 252-day loadings | - | - | 9.26 / 6.03 | 16.63 | 2.77 | 0.61 | `as_cited/validation/pca_b125_regression252` | `ea6bf50` 23 Nov 15:01 |
| **V3-PCA30-b64** | Loadings from 60 days, batches of 64 days | PCA, K = 30 | per day | 64 | - | - | correlation, 60-day loadings | - | - | 5.64 / 1.29 | 2.81 | 2.18 | 0.65 | `as_cited/validation/pca_b64_regression` | `ea6bf50` 23 Nov 15:01 |
| **V3-PCA30** | The benchmark of round 3 | PCA, K = 30 | per day | 125 | - | - | correlation, 60-day loadings | - | - | 5.59 / 1.27 | 2.74 | 2.17 | 0.65 | `as_cited/validation/pca_b125_regression` | `ea6bf50` 23 Nov 15:01 |
| **V3-PCA8** | The benchmark of round 3, 8 factors | PCA, K = 8 | per day | 125 | - | - | correlation, 60-day loadings | - | - | 7.53 / 4.65 | 14.56 | 3.14 | 0.62 | `as_cited/validation/pca_K8_b125_regression` | `ea6bf50` 23 Nov 15:01 |
| **V3-OU30** | The OU rule of round 3 | OU, K = 30 | - | - | - | - | correlation, 60-day loadings | - | - | 2.62 / -4.34 | -8.20 | 1.89 | 0.94 | `as_cited/validation/ou_K30_regression` | `ea6bf50` 23 Nov 15:01 |
| **V3-OU8** | The OU rule of round 3, 8 factors | OU, K = 8 | - | - | - | - | correlation, 60-day loadings | - | - | 3.56 / -1.54 | -4.09 | 2.66 | 0.98 | `as_cited/validation/ou_K8_regression` | `ea6bf50` 23 Nov 15:01 |
<!-- validation:end -->

No validation run was made for round 4 or for round 1b. The validation runs of round 3 do not
stand in for round 4: they are on the universe by security value, and round 4 changes the data of
these years too. Of the member-months of round 4's universe, 82 percent are also in the old one
in 1990 to 1995 and 79 percent in 1996 to 1997.

## Runs that were never committed

These exist only under `runs/` on the machine that made them. `runs/` is git-ignored, so this
table is filled only there. None of these numbers is reported in another document.

- **P**: validation runs made on 21 November, before the audit of the code. The audit found five
  bugs (`AI_USAGE.md`), so these numbers are not comparable with the validation table above. The
  committed validation runs of round 1 do not vary the batch; these are the only runs that compare
  batches of 32, 64, 128 and 252 days.
- **B**: the first versions of the nine benchmark runs that commit `7ac23ef` replaced.
- **D**: the 13 runs of round 4 on an earlier build of its data. That build still held share
  classes that hardly trade and took quotes on days without a trade as prices (`AI_USAGE.md`,
  round 4). Five of the runs were stopped before their last year.

<!-- local:start -->
| ID | What it is | Model | Test years | Sharpe ratio in the objective | Batch, days | Ridge penalty | Weight on explained variance | PCA residuals | SR before / after costs | Finished (file time) | Folder under `runs/` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **P-val2_B_K30** | Validation before the code audit of round 1 | AF, K = 30, seed 0 | 1996 to 1997 | annualized | 128 | 1e-05 | 100 | - | 10.03 / 6.94 | 21 Nov 22:14 | `_pre_audit/val2_B_K30` |
| **P-val2_B_batch252** | Validation before the code audit of round 1 | AF, K = 8, seed 0 | 1996 to 1997 | annualized | 252 | 1e-05 | 100 | - | 8.01 / 5.54 | 21 Nov 22:09 | `_pre_audit/val2_B_batch252` |
| **P-val2_B_batch64** | Validation before the code audit of round 1 | AF, K = 8, seed 0 | 1996 to 1997 | annualized | 64 | 1e-05 | 100 | - | 8.84 / 6.25 | 21 Nov 22:07 | `_pre_audit/val2_B_batch64` |
| **P-val2_B_ridge1e-2** | Validation before the code audit of round 1 | AF, K = 8, seed 0 | 1996 to 1997 | annualized | 128 | 0.01 | 100 | - | 5.88 / 4.09 | 21 Nov 22:07 | `_pre_audit/val2_B_ridge1e-2` |
| **P-val2_B_ridge1e-3** | Validation before the code audit of round 1 | AF, K = 8, seed 0 | 1996 to 1997 | annualized | 128 | 0.001 | 100 | - | 7.64 / 5.38 | 21 Nov 22:07 | `_pre_audit/val2_B_ridge1e-3` |
| **P-val2_B_ridge1e-4** | Validation before the code audit of round 1 | AF, K = 8, seed 0 | 1996 to 1997 | annualized | 128 | 0.0001 | 100 | - | 8.25 / 5.80 | 21 Nov 22:04 | `_pre_audit/val2_B_ridge1e-4` |
| **P-val2_B_seed1** | Validation before the code audit of round 1 | AF, K = 8, seed 1 | 1996 to 1997 | annualized | 128 | 1e-05 | 100 | - | 8.73 / 6.25 | 21 Nov 22:04 | `_pre_audit/val2_B_seed1` |
| **P-val2_B_seed2** | Validation before the code audit of round 1 | AF, K = 8, seed 2 | 1996 to 1997 | annualized | 128 | 1e-05 | 100 | - | 8.51 / 6.05 | 21 Nov 22:04 | `_pre_audit/val2_B_seed2` |
| **P-val2_ou_K8** | Validation before the code audit of round 1 | OU, K = 8 | 1996 to 1997 | - | - | - | - | covariance | 3.19 / -1.10 | 21 Nov 22:15 | `_pre_audit/val2_ou_K8` |
| **P-val2_pca_K30** | Validation before the code audit of round 1 | PCA, K = 30, seed 0 | 1996 to 1997 | per day | 128 | - | - | covariance | 9.15 / 5.90 | 21 Nov 22:23 | `_pre_audit/val2_pca_K30` |
| **P-val2_pca_K8_seed1** | Validation before the code audit of round 1 | PCA, K = 8, seed 1 | 1996 to 1997 | per day | 128 | - | - | covariance | 8.61 / 5.88 | 21 Nov 22:15 | `_pre_audit/val2_pca_K8_seed1` |
| **P-val2_pca_K8_seed2** | Validation before the code audit of round 1 | PCA, K = 8, seed 2 | 1996 to 1997 | per day | 128 | - | - | covariance | 8.41 / 5.63 | 21 Nov 22:15 | `_pre_audit/val2_pca_K8_seed2` |
| **P-val3_B_K30_b64** | Validation before the code audit of round 1 | AF, K = 30, seed 0 | 1996 to 1997 | annualized | 64 | 1e-05 | 100 | - | 10.16 / 6.96 | 21 Nov 22:24 | `_pre_audit/val3_B_K30_b64` |
| **P-val3_B_K8_b32** | Validation before the code audit of round 1 | AF, K = 8, seed 0 | 1996 to 1997 | annualized | 32 | 1e-05 | 100 | - | 8.66 / 5.95 | 21 Nov 22:25 | `_pre_audit/val3_B_K8_b32` |
| **P-val3_pca_K30_b64** | Validation before the code audit of round 1 | PCA, K = 30, seed 0 | 1996 to 1997 | per day | 64 | - | - | covariance | 9.13 / 5.90 | 21 Nov 22:23 | `_pre_audit/val3_pca_K30_b64` |
| **P-val3_pca_K8_b32** | Validation before the code audit of round 1 | PCA, K = 8, seed 0 | 1996 to 1997 | per day | 32 | - | - | covariance | 8.30 / 5.42 | 21 Nov 22:26 | `_pre_audit/val3_pca_K8_b32` |
| **P-val3_pca_K8_b64** | Validation before the code audit of round 1 | PCA, K = 8, seed 0 | 1996 to 1997 | per day | 64 | - | - | covariance | 8.77 / 5.98 | 21 Nov 22:23 | `_pre_audit/val3_pca_K8_b64` |
| **P-val_A_default** | Validation before the code audit of round 1 | AF, K = 8, seed 0 | 1996 to 1997 | per day | 128 | 1e-05 | 100 | - | 7.12 / 4.53 | 21 Nov 21:51 | `_pre_audit/val_A_default` |
| **P-val_B_sr_annualized** | Validation before the code audit of round 1 | AF, K = 8, seed 0 | 1996 to 1997 | annualized | 128 | 1e-05 | 100 | - | 8.56 / 6.08 | 21 Nov 21:52 | `_pre_audit/val_B_sr_annualized` |
| **P-val_C_lambda0** | Validation before the code audit of round 1 | AF, K = 8, seed 0 | 1996 to 1997 | per day | 128 | 1e-05 | 0 | - | 7.90 / 5.59 | 21 Nov 21:51 | `_pre_audit/val_C_lambda0` |
| **P-val_D_ridge1e-3** | Validation before the code audit of round 1 | AF, K = 8, seed 0 | 1996 to 1997 | per day | 128 | 0.001 | 100 | - | 7.22 / 4.69 | 21 Nov 21:54 | `_pre_audit/val_D_ridge1e-3` |
| **P-val_E_ridge1e-7** | Validation before the code audit of round 1 | AF, K = 8, seed 0 | 1996 to 1997 | per day | 128 | 1e-07 | 100 | - | 6.83 / 4.09 | 21 Nov 21:54 | `_pre_audit/val_E_ridge1e-7` |
| **P-val_F_batch64** | Validation before the code audit of round 1 | AF, K = 8, seed 0 | 1996 to 1997 | per day | 64 | 1e-05 | 100 | - | 7.25 / 4.54 | 21 Nov 21:55 | `_pre_audit/val_F_batch64` |
| **P-val_G_batch252** | Validation before the code audit of round 1 | AF, K = 8, seed 0 | 1996 to 1997 | per day | 252 | 1e-05 | 100 | - | 7.00 / 4.21 | 21 Nov 21:57 | `_pre_audit/val_G_batch252` |
| **P-val_H_lambda1** | Validation before the code audit of round 1 | AF, K = 8, seed 0 | 1996 to 1997 | per day | 128 | 1e-05 | 1 | - | 8.59 / 6.11 | 21 Nov 21:57 | `_pre_audit/val_H_lambda1` |
| **P-val_pca_K8** | Validation before the code audit of round 1 | PCA, K = 8, seed 0 | 1996 to 1997 | per day | 128 | - | - | covariance | 8.70 / 5.90 | 21 Nov 22:00 | `_pre_audit/val_pca_K8` |
| **B-ou_K1** | First version of a benchmark run of round 1, before its rerun | OU, K = 1 | 1998 to 2021 | - | - | - | - | covariance | 0.22 / -2.42 | 22 Nov 02:48 | `_before_rerun/ou_K1` |
| **B-ou_K10** | First version of a benchmark run of round 1, before its rerun | OU, K = 10 | 1998 to 2021 | - | - | - | - | covariance | 1.02 / -3.49 | 22 Nov 02:48 | `_before_rerun/ou_K10` |
| **B-ou_K100** | First version of a benchmark run of round 1, before its rerun | OU, K = 100 | 1998 to 2021 | - | - | - | - | covariance | 0.65 / -6.60 | 22 Nov 02:48 | `_before_rerun/ou_K100` |
| **B-ou_K15** | First version of a benchmark run of round 1, before its rerun | OU, K = 15 | 1998 to 2021 | - | - | - | - | covariance | 1.33 / -3.41 | 22 Nov 02:48 | `_before_rerun/ou_K15` |
| **B-ou_K3** | First version of a benchmark run of round 1, before its rerun | OU, K = 3 | 1998 to 2021 | - | - | - | - | covariance | 1.04 / -2.43 | 22 Nov 02:48 | `_before_rerun/ou_K3` |
| **B-ou_K30** | First version of a benchmark run of round 1, before its rerun | OU, K = 30 | 1998 to 2021 | - | - | - | - | covariance | 1.31 / -3.95 | 22 Nov 02:48 | `_before_rerun/ou_K30` |
| **B-ou_K5** | First version of a benchmark run of round 1, before its rerun | OU, K = 5 | 1998 to 2021 | - | - | - | - | covariance | 1.17 / -2.69 | 22 Nov 02:48 | `_before_rerun/ou_K5` |
| **B-ou_K8** | First version of a benchmark run of round 1, before its rerun | OU, K = 8 | 1998 to 2021 | - | - | - | - | covariance | 1.16 / -3.16 | 22 Nov 02:48 | `_before_rerun/ou_K8` |
| **B-pca_K30_seed0** | First version of a benchmark run of round 1, before its rerun | PCA, K = 30, seed 0 | 1998 to 2021 | per day | 64 | - | - | covariance | 3.47 / 1.26 | 22 Nov 02:48 | `_before_rerun/pca_K30_seed0.metrics.json` |
| **D-attention_K30_seed0** | Round 4 on an earlier build of its data, discarded | AF, K = 30, seed 0 | 1998 to 2021 | per day | 125 | 1e-05 | 100 | - |  | stopped; last year saved: 2016 | `company_discarded/attention_K30_seed0` |
| **D-attention_K30_seed1** | Round 4 on an earlier build of its data, discarded | AF, K = 30, seed 1 | 1998 to 2021 | per day | 125 | 1e-05 | 100 | - |  | stopped; last year saved: 2000 | `company_discarded/attention_K30_seed1` |
| **D-attention_K8_seed0** | Round 4 on an earlier build of its data, discarded | AF, K = 8, seed 0 | 1998 to 2021 | per day | 125 | 1e-05 | 100 | - | 1.32 / 0.38 | 23 Nov 16:56 | `company_discarded/attention_K8_seed0` |
| **D-attention_annualized_K30_seed0** | Round 4 on an earlier build of its data, discarded | AF, K = 30, seed 0 | 1998 to 2021 | annualized | 125 | 1e-05 | 100 | - |  | stopped; last year saved: 2016 | `company_discarded/attention_annualized_K30_seed0` |
| **D-attention_annualized_K8_seed0** | Round 4 on an earlier build of its data, discarded | AF, K = 8, seed 0 | 1998 to 2021 | annualized | 125 | 1e-05 | 100 | - | 1.83 / 0.87 | 23 Nov 16:56 | `company_discarded/attention_annualized_K8_seed0` |
| **D-ou_K30** | Round 4 on an earlier build of its data, discarded | OU, K = 30 | 1998 to 2021 | - | - | - | - | correlation, 60-day loadings | 1.12 / -5.57 | 23 Nov 17:06 | `company_discarded/ou_K30` |
| **D-ou_K8** | Round 4 on an earlier build of its data, discarded | OU, K = 8 | 1998 to 2021 | - | - | - | - | correlation, 60-day loadings | 1.34 / -3.67 | 23 Nov 17:06 | `company_discarded/ou_K8` |
| **D-ouproj_K30** | Round 4 on an earlier build of its data, discarded | OU, K = 30 | 1998 to 2021 | - | - | - | - | covariance | 1.77 / -3.88 | 23 Nov 17:07 | `company_discarded/ouproj_K30` |
| **D-ouproj_K8** | Round 4 on an earlier build of its data, discarded | OU, K = 8 | 1998 to 2021 | - | - | - | - | covariance | 1.38 / -3.07 | 23 Nov 17:07 | `company_discarded/ouproj_K8` |
| **D-pca_K30_seed0** | Round 4 on an earlier build of its data, discarded | PCA, K = 30, seed 0 | 1998 to 2021 | per day | 125 | - | - | correlation, 60-day loadings | 1.63 / -1.43 | 23 Nov 16:28 | `company_discarded/pca_K30_seed0` |
| **D-pca_K8_seed0** | Round 4 on an earlier build of its data, discarded | PCA, K = 8, seed 0 | 1998 to 2021 | per day | 125 | - | - | correlation, 60-day loadings |  | stopped; last year saved: 2018 | `company_discarded/pca_K8_seed0` |
| **D-pcaproj_K30_seed0** | Round 4 on an earlier build of its data, discarded | PCA, K = 30, seed 0 | 1998 to 2021 | per day | 125 | - | - | covariance | 3.64 / 1.15 | 23 Nov 17:06 | `company_discarded/pcaproj_K30_seed0` |
| **D-pcaproj_K8_seed0** | Round 4 on an earlier build of its data, discarded | PCA, K = 8, seed 0 | 1998 to 2021 | per day | 125 | - | - | covariance |  | stopped; last year saved: 2017 | `company_discarded/pcaproj_K8_seed0` |
<!-- local:end -->

Not kept anywhere: a trial with 100 factors on validation data that was stopped (`README.md`), and
two earlier restarts of round 4 (`AI_USAGE.md` counts three restarts; one set of runs is kept).

## Where each setting that changed comes from

Five labels say what a value rests on:

- **Paper**: the paper states it, in its text, a formula or a table.
- **Cited source**: the paper does not state it, and a source that the paper cites for it does.
- **Inferred**: reconstructed from indirect evidence in the paper.
- **Our choice**: neither the paper nor a cited source gives it.
- **Unknown**: which value the authors used is not known.

"The predecessor" is Guijarro-Ordonez, Pelger and Zanotti, "Deep Learning Statistical Arbitrage",
and its public code `github.com/gregzanotti/dlsa-public`. The paper cites it for its costs, for its
PCA benchmark ("in the spirit of") and for its OU benchmark ("the same implementation").

### 1. The Sharpe ratio in the objective: per day or annualized

- **Values run.** Annualized: rounds 1 and 1b, and two runs of round 4 (`4-AF8ann-s0`, `4-AF30ann-s0`).
  Per day: rounds 2a, 2b and 3, and the other runs of round 4.
- **The paper.** Section 3.3 prints the objective as the mean net return minus the risk-free rate,
  over the standard deviation of the net returns, plus the weight times the explained variance.
  No annualization factor is printed. The weight is 100 (Section 4.2 and Table 4). "The tradeoff
  between these two objectives is selected optimally on the validation data."
- **The predecessor's code.** `train_test.py` has `loss = -mean_ret/std` on the returns of one
  batch. The square root of 252 appears only in a line that logs the result.
- **The predecessor's paper.** Its appendix on the estimation says the objective is computed by
  "replacing the mean and variance by their annualized sample counterpart over a training set".
  That objective has one term, so its scale changes nothing there.
- **Where each value comes from.** Per day: **Paper**, as printed, and the predecessor's code.
  Annualized: **Our choice** in round 1, made on validation (7.50 after costs against 5.17), and a
  departure from the printed formula. The predecessor's paper describes annualized moments, which
  `README_ISSUES.md` does not mention. Which scale the authors used with the weight of 100 is
  **Unknown**. No other setting moves the Attention Factor model as much: 3.00 and 0.99
  (annualized) against 2.10 and 0.09 (per day) with 30 factors, two seeds each.

### 2. The batch: 64 days, 125 days or the whole training window

- **Values run.** 64: rounds 1, 1b and 2a. The whole window: round 2b. 125: rounds 3 and 4.
- **The paper.** No batch is given. Table 4 has "Epochs 30: Number of passes over the data".
- **The predecessor's paper.** Its appendix splits the training window into batches, each "a
  subwindow of 125 days of the original training window, with the subwindows being consecutive and
  non-overlapping".
- **The predecessor's code.** `configs/ouffn-full.yaml` has `batch_size: 125`; one loss and one
  optimizer step per batch. The other three configuration files were not read.
- **Where each value comes from.** 64: **Our choice**, made on validation runs that predate the
  code audit (the P runs below tried 32, 64, 128 and 252 days). 125: **Cited source**, the predecessor's
  value; whether the paper kept it is **Unknown**. The whole window: a reading of "passes over
  the data" tried in round 2b, which the run could not test.

### 3. The bound on one security's explained-variance term: -1 or none

- **Values run.** -1: rounds 1, 1b and 2a. None: rounds 2b, 3 and 4.
- **The paper.** The term is the average over the securities of 1 - Var(residual) / Var(return)
  (Section 3.3). No bound is printed.
- **Cited sources.** None has this term: the predecessor's objective is the Sharpe ratio alone.
- **Where each value comes from.** None: **Paper**, as printed. -1: **Our choice** in round 1,
  added because a stock whose price hardly moves in a batch took over the gradient.

### 4. The inputs: three characteristics, and the group of past returns

- **Values run.** The definitions of round 1: rounds 1, 1b, 2a and 2b. The cited definitions:
  rounds 3 and 4. Without the eight inputs of the past-return group: `1b-AF30nopast-s0`.
- **The paper.** "39 firm-specific characteristics" from Chen, Pelger and Zhu; "the full details on
  the construction of these variables are in the Internet Appendix of" that paper (Section 4.1).
  Table 1 names the characteristics and their six groups. Table 3 removes one group at a time.
- **The cited source.** The appendix of Chen, Pelger and Zhu, read in the arXiv versions 1 to 3 by
  AI research agents for round 3 (`README_ISSUES.md`, issue 4). Beta: correlation from overlapping
  three-day log returns over five years, times a ratio of volatilities over one year. Variance and
  Resid_Var: "in the past two months". The final Internet Appendix was not read.
- **Where each value comes from.** Cited: **Cited source**, with gaps (r2_1 and ST_Rev are one
  series here and two in the authors' data; the window of SUV and the formula of Spread are not
  given). Round 1: **Our choice**, reconstructed from standard definitions (Beta as a regression
  slope on 252 days; Variance and Resid_Var on 60 trading days). Without past returns: **Paper**
  (Table 3); the members of the group are those of Table 1.

### 5. The residual history: own or current

- **Values run.** Current: `1b-AF30cur-s0`, `1b-PCA30cur-s0` and `1b-PCA30corrcur-s0`. Own: every
  other run.
- **The paper.** The policy reads "the past s residuals" (Section 3.3), and the residual of day t
  is defined with the weights of day t-1 (Section 3.2). "All sequence models use a look-back
  window of the past 30 daily residual returns" (Section 4.2).
- **The predecessor's code.** `factor_models/pca.py` computes the residual of each day with
  loadings from windows that end the day before.
- **Where each value comes from.** Own: **Paper**, read literally, and the predecessor's code.
  Current: **Our choice**, an alternative tested in round 1b. `README.md` says that the paper's
  notation fits both readings.

### 6. The PCA residuals of the two benchmarks

- **Values run.** Covariance: rounds 1 and 1b, and four runs of round 4 (`4-PCA8cov-s0`,
  `4-PCA30cov-s0`, `4-OU8cov`, `4-OU30cov`). Correlation: `1b-PCA30corr-s0` and `1b-PCA30corrcur-s0`.
  Correlation with 60-day loadings: round 3 and the other benchmark runs of round 4. Correlation
  with 252-day loadings: one validation run, `V3-PCA30-reg252`.
- **The paper.** "The latent factors are estimated with PCA using the past 252 trading days";
  "essentially a benchmark in the spirit of" the predecessor; for the OU rule, "the same
  implementation as in" the predecessor (Section 4.2). Section 3.1 describes PCA in general with
  weights that "are the eigenvectors of the top K eigenvalues of the return covariance matrix".
  Table 2 has a row with 100 factors.
- **The predecessor's paper.** "We use the last 252 days, or roughly one trading year, to estimate
  the correlation matrix from which we extract the PCA factors. Then, we use the last 60 days to
  estimate the loadings on the latent factors using linear regressions, and compute residuals for
  the current day out-of-sample."
- **The predecessor's code.** `factor_models/pca.py`: returns standardized over 252 days
  (`sizeCovarianceWindow=252`), loadings from `LinearRegression(fit_intercept=False)` on the last 60
  days (`sizeWindow=60`), residual equal to the return minus loadings times factor returns.
- **Where each value comes from.** Covariance: **Our choice** in round 1; it fits the general
  description of Section 3.1. Correlation with 60-day loadings: **Cited source**. It cannot be the
  paper's recipe for every row: a regression on 60 days cannot estimate loadings on 100 factors.
  Correlation: **Our choice** in round 1b, after Avellaneda and Lee (2010), whom the predecessor
  names for its procedure. Which construction the authors used is **Unknown**. This setting
  decides the ranking of the model against the benchmark. Out of sample with 30 factors the three
  constructions give 3.47 and 1.26 (covariance, two seeds), 2.62 and 0.38 (correlation, one seed),
  and 1.73 and -1.26 (60-day loadings, two seeds, batches of 125 days where the others have 64); the
  paper has 2.79 and 1.57.
- **Two documents disagree here.** On validation the correlation matrix with 252-day loadings
  scores like the covariance (`V3-PCA30-reg252`: 9.26 and 6.03; `V3-PCA30-cov`: 9.02 and 5.90), and
  `README_ISSUES.md` concludes that standardizing changes nothing and only the 60-day regression
  does. Out of sample the correlation construction of round 1b, which differs from the one with
  252-day loadings only in details (no risk-free rate subtracted, returns demeaned), is 0.85 lower
  before costs than the covariance. No run has both constructions on the same years.

### 7. The universe

- **Values run.** By security value: rounds 1 to 3 and 1b. By company value: round 4.
- **The paper.** "We collect daily equity return data for the securities on CRSP"; "we consider
  only the 500 largest stocks based on the previous month market capitalization" (Section 4.1).
  Figures 3 and 4 name 108 firms among "the 500 equities with the largest market cap on the first
  trading day of 2021". The note of Table 2 calls its market row "the equally weighted market
  portfolio".
- **The predecessor.** It uses another rule: stocks with more than 0.01 percent of the total
  market capitalization, about 550 stocks. The paper does not point to it for the universe.
- **Where each value comes from.** By security value: **Our choice** of how to read the sentence
  (which securities count as stocks, and the value of the security). By company value, with
  partnership units and 120 months of history: **Inferred** from the 108 named firms; 91 of them
  are members in January 2021, against 68. The paper's rule is **Unknown**: 16 named firms are
  left out for a reason that is not known, and neither universe gives the paper's market row
  (8.61 percent a year; ours 11.6 and 11.8). The rule for days without a trade, which only
  round 4 has, is **Our choice**.

### 8. The number of factors

- **Values run.** The Attention Factor model: 8 and 30 in every round, 100 once (`1b-AF100-s0`).
  PCA with the long convolution model: 8 and 30. The OU rule: 1, 3, 5, 8, 10, 15, 30 and 100 in
  round 1; 8 and 30 in rounds 3 and 4.
- **Where it comes from.** **Paper**: "We consider 1, 3, 5, 8, 10, 15, 30, and 100 latent
  factors" (Section 4.2). Not run for the two trained models: 1, 3, 5 and 15, and 10 and 100 for
  the benchmark.

### 9. The seeds

- **Values run.** Seed 0 in every run. Seed 1 as well for the trained models with 30 factors in
  rounds 1, 2a, 2b and 3, and for one run of round 4 (`4-AF30-s1`).
- **The paper.** The note of Table 3: "All values are averaged across multiple starting seeds for
  model estimation of the neural networks, and the value in parentheses for the Sharpe Ratio is
  the standard deviation across seeds" (0.13 for the model with 30 factors). The number of seeds
  is not given, and the note of Table 2 does not mention seeds.
- **Where it comes from.** The number of seeds: **Unknown**; one or two here, for compute. How a
  seed is used (the generator starts from the seed times 100003 plus the test year): **Our
  choice**.

### 10. Changed on validation only: the ridge penalty and the weight on the explained variance

- **The ridge penalty of the loadings.** Run with 1e-4, 1e-5, 1e-6 and 1e-7 (`V1-AF30` and its
  three variants); 1e-2 and 1e-3 before the audit (the P runs). The paper adds "a ridge penalty
  for stability" (Section 3.2) and gives no value; Table 4 does not list it. The predecessor has
  no such penalty. 1e-5 is **Our choice**, made on validation (7.50 after costs, against 7.45,
  7.15 and 7.21). The authors' value is **Unknown**.
- **The weight on the explained variance.** 100 is **Paper** (Section 4.2 and Table 4). One
  validation run has 0 (`V1-AF30-lam0`: 6.88 after costs, against 7.50 with 100).

## Settings that are the same in every run

| Setting | Value | Where it comes from | Label |
|---|---|---|---|
| Sample | CRSP daily returns, January 1990 to December 2021 | Section 4.1. The daily data here start in 1988, to fill the first lookbacks | Paper |
| Protocol | A rolling window of 8 years, retrained every year; out of sample January 1998 to December 2021 | Section 4.2: "a rolling window of 8 years, where we retrain the models every year" | Paper |
| Validation years | 1996 and 1997, with the six years before each as training data | Section 4.2: "the last two years of the first training data to select tuning parameters". The six training years are ours | Paper; split ours |
| Size of the universe | 500, ranked on the capitalization of the previous month | Section 4.1. What is ranked differs between rounds (setting 7) | Paper |
| Exchanges | NYSE, AMEX, Nasdaq | The paper says "the securities on CRSP" | Our choice |
| The 39 characteristics | 35 monthly and 4 daily, in six groups | Table 1 and Section 4.1. Three definitions differ between rounds (setting 4) | Paper |
| The 4 daily inputs | Return of the day, return and standard deviation of five days, dollar volume of five days | Section 4.1: "the previous day and week return and volatility"; Table 1: "weekly trading volume". The windows are ours | Paper; details ours |
| Normalization | Ranks in (-0.5, 0.5) within the universe; tied values share their average rank | Section 4.1: "normalized to rank quantiles". Scale and ties are ours | Paper; details ours |
| Missing values | The last value, for at most 12 months or 5 days; else the cross-sectional median | Section 4.1: "imputed with last observed values if available and by the cross-sectional median otherwise". The limits are ours | Paper; limits ours |
| The 40 inputs equal for all stocks | Medians of the characteristics and the risk-free rate; 79 inputs in all | Section 4.1. They cancel in the softmax of the paper's equation and change nothing (`README.md`) | Paper |
| Accounting data | Fama-French timing: at least six months old | Not stated | Our choice |
| Factor weights, loadings, residuals | Softmax of scaled inner products; loadings from the weights with a ridge penalty; residual equal to the return minus loadings times factor returns | Equations of Sections 3.1 and 3.2 | Paper |
| Attention dimension, hidden dimension, layers | 32, 32, 1 | Section 4.2 and Table 4 | Paper |
| Lookback | 30 days of residuals | Section 4.2 and the note of Table 2. The predecessor: "a local lookback window of L = 30 days" | Paper |
| Dropout, Squash strength, kernel initialization | 0.1, 0.001, geometric decay | Table 4 and Appendix A | Paper |
| Epochs, learning rate, weight decay | 30, 0.003, 0.05 on the sequence model only | Table 4 | Paper |
| Optimizer | AdamW (decoupled weight decay) | Section 4.2 says "Adam" and cites the paper that introduces AdamW | Paper; reading ours |
| Weight on the explained variance | 100 in every run out of sample | Section 4.2 and Table 4 | Paper |
| Ridge penalty | 1e-5 in every run out of sample | Not reported. Chosen on validation (setting 10) | Our choice |
| Sequence model, input and output | The residual series enters the 32 channels through an affine lift, in percent; GELU; dropout on the 32 channels; a linear read-out at the last position | The paper gives the layer, not how a scalar series enters and leaves it | Our choice |
| Initialization of W and Q | Gaussian | Not stated. A row that is commented out in the paper's TeX source reads "Q-init: Market. The first factor is initialized to the market factor" | Our choice |
| Batches | Consecutive days; the first cut at a random offset; random order; every date in one batch per epoch | Not stated. Consecutive, non-overlapping batches are the predecessor's convention | Our choice |
| Costs | 0.0005 times the L1 change of the weights plus 0.0001 times the short positions | Section 3.3, "as in" the predecessor, whose Section III.J has the same formula | Paper |
| Timing of the cost | The return of a weight is charged with the trade that established it | The printed formula pairs the return with the next trade; a shift of one day | Our choice |
| Gross exposure | The weights of a day are divided by the sum of their absolute values | Section 3.3: the L1 norm of the weights is one. The predecessor's code does the same | Paper |
| Risk-free rate | Subtracted inside the Sharpe ratio of the objective; total returns are traded | The objective of Section 3.3 subtracts it. The predecessor trades excess returns | Paper; returns ours |
| Sharpe ratio of the tables | Mean over volatility of the daily returns, annualized, no risk-free rate subtracted | The paper says "annualized" and gives no formula | Our choice |
| Explained variance, which securities | Those with a return on half of the days of the batch and a return variance above 1e-10 | Not stated | Our choice |
| Who gets an allocation | A security in the universe on the decision date and on the 30 dates before, with a return on each day of the lookback | Not stated | Our choice |
| PCA window | The last 252 trading days | Section 4.2 and the note of Table 2 | Paper |
| Securities of the PCA benchmarks | In the universe, with a return on each of the 252 days | The predecessor: "we only consider the stocks with no missing observations in the daily returns during the rolling window" | Cited source |
| OU threshold rule | An AR(1) model on the cumulated residuals of 30 days; -1 above 1.25 standard deviations, +1 below -1.25, 0 otherwise or when the R-squared is below 0.25 | The predecessor's paper, Appendix B: "Our benchmark model has the values c_thresh = 1.25 and c_crit = 0.25". Its public code has no threshold rule, so the code is ours | Cited source; code ours |
| Returns | CRSP daily returns with delisting returns; -30 percent where CRSP has none for a delisting for cause; a missing return counts as zero | Not stated | Our choice |
| Seeding | The generators start from the seed times 100003 plus the test year | Not stated | Our choice |
| Evaluation | The weights of all test years are joined before turnover and costs are computed | Not stated | Our choice |
| Data vintage | WRDS files of November 2026 | The paper is of 2025 | Differs |
| Hardware | A laptop CPU, two threads per run | The paper: five GPUs | Differs |

## How the sources were checked

This file was written with Claude Code (model Claude Fable 5.2) on 25 November 2026, at the
direction of a group member. No person has repeated the checks below.

- **Settings and results.** Read from the saved `metrics.json` of each run by
  `scripts/runs_table.py`, not copied from the other documents. A setting that a run did not
  record is shown with the value that the code had at the time (the bound of -1 in rounds 1, 1b
  and 2a).
- **The paper.** Quotations are from version 1 on arXiv (a copy is `paper/2510.11616v1.pdf`
  in this repository).
- **The predecessor's paper.** The text of arXiv 2106.04028, version 2, was searched for every
  sentence quoted here: the PCA, the batches, the OU constants, the objective, the universe, the
  costs, the normalization and the lookback.
- **The predecessor's code.** An AI agent read `train_test.py`, `factor_models/pca.py`,
  `preprocess.py`, `run_train_test.py` and `configs/ouffn-full.yaml` of the repository on
  25 November. One difference from what `README_ISSUES.md` implies: the shipped configuration has
  both cost constants at 0, with 0.0005 and 0.0001 only as comments.
- **Chen, Pelger and Zhu.** Not read again for this file. The statements are those of
  `README_ISSUES.md`, issue 4.
- **The merged code.** After the merge of `experiments/open-problems`, the code gave back the saved
  weights of rounds 1, 2a, 3 and 4 bit for bit: the OU rule and the Attention Factor model for
  1998, 2010 and 2021, and one year of the PCA benchmark trained again in rounds 1, 3 and 4.

## Statements in other files that the merge made out of date

The merge of `experiments/open-problems` brought the six runs of round 1b and new slides. These
statements were written without them and no longer hold. They are listed here and not changed.

- **`README.md`, "What the code covers and what was run".** It says that Table 3 and the Attention
  Factor model with 100 factors were not run and that the `--drop` option is untested.
  `1b-AF100-s0` and `1b-AF30nopast-s0` are those runs.
- **`README_ISSUES.md`, "What is still open", item 6**: "no run with 100 factors, no Table 3".
  The 100 factors and one row of Table 3 were run in round 1b, with the specification of round 1
  only.
- **`README_ISSUES.md`, "What this changes on the slides".** The table quotes eleven statements
  from the slides. The slides were rewritten on the branch: four of the quoted statements are gone
  (the lead over the baseline, the volatility line, "Validation only", "The level reproduces; the
  ranking does not") and two are reworded (the count of tests, the net Sharpe gap). The slides now
  say "24 tests"; there are 34.
- **`README_ISSUES.md`, issue 6**, on standardizing: see setting 6 above.
- **`RESULTS.md`, "Open choices tested", item 5**, calls the annualized Sharpe ratio a choice that
  "holds on the test years". `README_ISSUES.md`, issue 1, calls the same choice a departure from the
  printed objective on which the headline rests. The numbers are the same; the reading differs.
