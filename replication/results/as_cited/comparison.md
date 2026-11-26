# Three rounds of runs beside the paper

Written by `scripts/as_cited_comparison.py` from the saved result files. The reading of these tables is in
`README_ISSUES.md`. Out of sample means January 1998 to December 2021 (6,040 days); all numbers are annualized.
A cell of a model with two seeds is the average of the two runs.

## Out of sample, 1998 to 2021

### Attention Factor model, K = 30

| | Round 1: annualized Sharpe (slides) (2 seeds) | Round 2a: Sharpe per day (2 seeds) | Round 2b: whole window, 30 steps (not a valid test) (2 seeds) | Round 3: as cited (2 seeds) | Paper, Table 2 |
|---|---|---|---|---|---|
| Sharpe ratio | 3.00 | 2.10 | 1.47 | 2.05 | 3.97 |
| Sharpe ratio after costs | 0.99 | 0.09 | -1.73 | 0.09 | 2.28 |
| Mean, % a year | 12.81 | 10.54 | 6.54 | 10.12 | 16.66 |
| Mean after costs, % a year | 4.20 | 0.42 | -7.65 | 0.43 | 9.52 |
| Volatility, % a year | 4.26 | 5.02 | 4.43 | 4.95 | 4.20 |
| Market beta | 0.05 | 0.03 | 0.02 | 0.04 | 0.05 |
| Turnover a day | 0.59 | 0.70 | 1.03 | 0.67 |  |
| Cost, points a year | 8.61 | 10.11 | 14.19 | 9.69 |  |

### Benchmarks, K = 30

| Model | Residuals | Seeds | Sharpe ratio | after costs | Mean % | after costs | Volatility % | Turnover a day |
|---|---|---|---|---|---|---|---|---|
| PCA with the long convolution model | Round 1: projection | 2 | 3.47 | 1.26 | 11.12 | 4.02 | 3.21 | 0.47 |
| PCA with the long convolution model | Round 3: loadings from 60 days | 2 | 1.73 | -1.26 | 4.41 | -3.20 | 2.55 | 0.50 |
| PCA with the long convolution model | Paper, Table 2 | | 2.79 | 1.57 | 15.15 | 8.47 | 5.42 | |
| PCA with an OU threshold rule | Round 1: projection | - | 1.31 | -3.95 | 3.33 | -10.01 | 2.53 | 0.96 |
| PCA with an OU threshold rule | Round 3: loadings from 60 days | - | 0.90 | -5.61 | 1.80 | -11.18 | 1.99 | 0.93 |
| PCA with an OU threshold rule | Paper, Table 2 | | 0.18 | -6.45 | 0.40 | -14.74 | 2.24 | |

### Attention Factor model, K = 8

| | Round 1: annualized Sharpe (slides) (1 seed) | Round 2a: Sharpe per day (1 seed) | Round 2b: whole window, 30 steps (not a valid test) (1 seed) | Round 3: as cited (1 seed) | Paper, Table 2 |
|---|---|---|---|---|---|
| Sharpe ratio | 2.66 | 2.10 | 1.47 | 2.06 | 3.35 |
| Sharpe ratio after costs | 1.15 | 0.42 | -0.90 | 0.53 | 1.94 |
| Mean, % a year | 14.38 | 12.13 | 8.18 | 11.65 | 15.70 |
| Mean after costs, % a year | 6.21 | 2.45 | -5.02 | 3.01 | 9.05 |
| Volatility, % a year | 5.41 | 5.78 | 5.55 | 5.67 | 4.68 |
| Market beta | 0.08 | 0.05 | 0.03 | 0.07 | 0.07 |
| Turnover a day | 0.56 | 0.67 | 0.95 | 0.59 |  |
| Cost, points a year | 8.18 | 9.68 | 13.20 | 8.65 |  |

### Benchmarks, K = 8

| Model | Residuals | Seeds | Sharpe ratio | after costs | Mean % | after costs | Volatility % | Turnover a day |
|---|---|---|---|---|---|---|---|---|
| PCA with the long convolution model | Round 1: projection | 1 | 3.07 | 1.35 | 13.87 | 6.10 | 4.52 | 0.53 |
| PCA with the long convolution model | Round 3: loadings from 60 days | 1 | 2.29 | 0.47 | 9.45 | 1.93 | 4.12 | 0.50 |
| PCA with the long convolution model | Paper, Table 2 | | 2.64 | 1.50 | 14.88 | 8.42 | 5.63 | |
| PCA with an OU threshold rule | Round 1: projection | - | 1.16 | -3.16 | 3.61 | -9.83 | 3.11 | 0.97 |
| PCA with an OU threshold rule | Round 3: loadings from 60 days | - | 1.38 | -3.29 | 3.92 | -9.32 | 2.83 | 0.95 |
| PCA with an OU threshold rule | Paper, Table 2 | | 0.78 | -4.15 | 2.04 | -10.83 | 2.61 | |

### Attention Factor model minus PCA with the long convolution model, K = 30

The Sharpe ratios of this table are those of the daily returns averaged over the seeds of a model. With two
seeds they are a little larger in size than the averages of the seeds' Sharpe ratios in the tables above.

| Model | Benchmark | | Sharpe ratios | Difference | Bootstrap s.e., blocks of 1 / 21 / 63 days | Difference in s.e. |
|---|---|---|---|---|---|---|
| Round 1: annualized Sharpe (slides) | PCA, projection (round 1) | before costs | 3.11 and 3.49 | -0.38 | 0.22 / 0.28 / 0.31 | 1.8 / 1.4 / 1.2 |
| Round 1: annualized Sharpe (slides) | PCA, projection (round 1) | after costs | 1.02 and 1.27 | -0.25 | 0.20 / 0.23 / 0.25 | 1.3 / 1.1 / 1.0 |
| Round 1: annualized Sharpe (slides) | PCA, loadings from 60 days (round 3) | before costs | 3.11 and 1.84 | +1.27 | 0.24 / 0.27 / 0.30 | 5.3 / 4.6 / 4.2 |
| Round 1: annualized Sharpe (slides) | PCA, loadings from 60 days (round 3) | after costs | 1.02 and -1.34 | +2.36 | 0.23 / 0.25 / 0.28 | 10.3 / 9.4 / 8.6 |
| Round 2a: Sharpe per day | PCA, projection (round 1) | before costs | 2.19 and 3.49 | -1.30 | 0.22 / 0.26 / 0.28 | 6.0 / 4.9 / 4.7 |
| Round 2a: Sharpe per day | PCA, projection (round 1) | after costs | 0.09 and 1.27 | -1.18 | 0.21 / 0.23 / 0.24 | 5.6 / 5.1 / 4.9 |
| Round 2a: Sharpe per day | PCA, loadings from 60 days (round 3) | before costs | 2.19 and 1.84 | +0.35 | 0.22 / 0.24 / 0.26 | 1.6 / 1.4 / 1.3 |
| Round 2a: Sharpe per day | PCA, loadings from 60 days (round 3) | after costs | 0.09 and -1.34 | +1.43 | 0.23 / 0.24 / 0.27 | 6.3 / 5.9 / 5.4 |
| Round 2b: whole window, 30 steps (not a valid test) | PCA, projection (round 1) | before costs | 1.73 and 3.49 | -1.76 | 0.24 / 0.29 / 0.28 | 7.3 / 6.1 / 6.3 |
| Round 2b: whole window, 30 steps (not a valid test) | PCA, projection (round 1) | after costs | -2.02 and 1.27 | -3.29 | 0.25 / 0.27 / 0.27 | 13.1 / 12.4 / 12.1 |
| Round 2b: whole window, 30 steps (not a valid test) | PCA, loadings from 60 days (round 3) | before costs | 1.73 and 1.84 | -0.11 | 0.24 / 0.26 / 0.27 | 0.5 / 0.4 / 0.4 |
| Round 2b: whole window, 30 steps (not a valid test) | PCA, loadings from 60 days (round 3) | after costs | -2.02 and -1.34 | -0.68 | 0.26 / 0.26 / 0.26 | 2.6 / 2.7 / 2.6 |
| Round 3: as cited | PCA, projection (round 1) | before costs | 2.13 and 3.49 | -1.36 | 0.22 / 0.29 / 0.30 | 6.1 / 4.8 / 4.5 |
| Round 3: as cited | PCA, projection (round 1) | after costs | 0.09 and 1.27 | -1.18 | 0.21 / 0.24 / 0.25 | 5.7 / 5.0 / 4.8 |
| Round 3: as cited | PCA, loadings from 60 days (round 3) | before costs | 2.13 and 1.84 | +0.29 | 0.23 / 0.26 / 0.28 | 1.3 / 1.1 / 1.0 |
| Round 3: as cited | PCA, loadings from 60 days (round 3) | after costs | 0.09 and -1.34 | +1.43 | 0.22 / 0.24 / 0.26 | 6.5 / 6.0 / 5.5 |

Paper, Table 2: 3.97 and 2.79 before costs (+1.18), 2.28 and 1.57 after (+0.71).

### Attention Factor model minus PCA with the long convolution model, K = 8

The Sharpe ratios of this table are those of the daily returns averaged over the seeds of a model. With two
seeds they are a little larger in size than the averages of the seeds' Sharpe ratios in the tables above.

| Model | Benchmark | | Sharpe ratios | Difference | Bootstrap s.e., blocks of 1 / 21 / 63 days | Difference in s.e. |
|---|---|---|---|---|---|---|
| Round 1: annualized Sharpe (slides) | PCA, projection (round 1) | before costs | 2.66 and 3.07 | -0.41 | 0.19 / 0.23 / 0.23 | 2.1 / 1.8 / 1.8 |
| Round 1: annualized Sharpe (slides) | PCA, projection (round 1) | after costs | 1.15 and 1.35 | -0.20 | 0.18 / 0.20 / 0.18 | 1.1 / 1.0 / 1.1 |
| Round 1: annualized Sharpe (slides) | PCA, loadings from 60 days (round 3) | before costs | 2.66 and 2.29 | +0.37 | 0.20 / 0.24 / 0.24 | 1.8 / 1.5 / 1.5 |
| Round 1: annualized Sharpe (slides) | PCA, loadings from 60 days (round 3) | after costs | 1.15 and 0.47 | +0.68 | 0.19 / 0.21 / 0.21 | 3.6 / 3.2 / 3.2 |
| Round 2a: Sharpe per day | PCA, projection (round 1) | before costs | 2.10 and 3.07 | -0.97 | 0.20 / 0.21 / 0.21 | 4.9 / 4.5 / 4.6 |
| Round 2a: Sharpe per day | PCA, projection (round 1) | after costs | 0.42 and 1.35 | -0.93 | 0.19 / 0.20 / 0.19 | 4.8 / 4.8 / 4.9 |
| Round 2a: Sharpe per day | PCA, loadings from 60 days (round 3) | before costs | 2.10 and 2.29 | -0.19 | 0.22 / 0.24 / 0.23 | 0.9 / 0.8 / 0.8 |
| Round 2a: Sharpe per day | PCA, loadings from 60 days (round 3) | after costs | 0.42 and 0.47 | -0.05 | 0.21 / 0.21 / 0.22 | 0.2 / 0.2 / 0.2 |
| Round 2b: whole window, 30 steps (not a valid test) | PCA, projection (round 1) | before costs | 1.47 and 3.07 | -1.60 | 0.23 / 0.28 / 0.29 | 7.0 / 5.7 / 5.5 |
| Round 2b: whole window, 30 steps (not a valid test) | PCA, projection (round 1) | after costs | -0.90 and 1.35 | -2.26 | 0.22 / 0.26 / 0.27 | 10.4 / 8.6 / 8.4 |
| Round 2b: whole window, 30 steps (not a valid test) | PCA, loadings from 60 days (round 3) | before costs | 1.47 and 2.29 | -0.82 | 0.21 / 0.27 / 0.28 | 3.8 / 3.0 / 2.9 |
| Round 2b: whole window, 30 steps (not a valid test) | PCA, loadings from 60 days (round 3) | after costs | -0.90 and 0.47 | -1.37 | 0.21 / 0.26 / 0.27 | 6.5 / 5.3 / 5.1 |
| Round 3: as cited | PCA, projection (round 1) | before costs | 2.06 and 3.07 | -1.01 | 0.20 / 0.24 / 0.24 | 5.0 / 4.2 / 4.2 |
| Round 3: as cited | PCA, projection (round 1) | after costs | 0.53 and 1.35 | -0.82 | 0.19 / 0.21 / 0.20 | 4.4 / 3.9 / 4.1 |
| Round 3: as cited | PCA, loadings from 60 days (round 3) | before costs | 2.06 and 2.29 | -0.24 | 0.20 / 0.23 / 0.24 | 1.2 / 1.0 / 1.0 |
| Round 3: as cited | PCA, loadings from 60 days (round 3) | after costs | 0.53 and 0.47 | +0.06 | 0.19 / 0.21 / 0.22 | 0.3 / 0.3 / 0.3 |

Paper, Table 2: 3.35 and 2.64 before costs (+0.71), 1.94 and 1.50 after (+0.44).

### Round 3 by period

Sharpe ratio before costs / after costs / mean after costs in % a year.

| Run | 1998-2002 | 2003-2009 | 2010-2021 | 2003-2021 |
|---|---|---|---|---|
| attention_K30_seed0 | 3.83 / 2.34 / 16.1 | 1.54 / -0.21 / -1.2 | 1.02 / -1.68 / -5.8 | 1.24 / -0.93 / -4.1 |
| attention_K30_seed1 | 4.06 / 2.54 / 17.1 | 1.96 / -0.15 / -0.7 | 1.07 / -1.37 / -5.2 | 1.44 / -0.84 / -3.5 |
| attention_K8_seed0 | 3.78 / 2.50 / 19.2 | 1.40 / -0.24 / -1.4 | 1.48 / -0.26 / -1.1 | 1.43 / -0.25 / -1.2 |
| pca_K30_seed0 | 3.93 / 1.25 / 4.5 | 1.75 / -1.49 / -3.9 | -0.04 / -3.33 / -6.4 | 0.74 / -2.48 / -5.5 |
| pca_K30_seed1 | 3.78 / 1.19 / 4.3 | 2.12 / -1.35 / -3.4 | 0.23 / -3.12 / -5.8 | 1.05 / -2.31 / -4.9 |
| pca_K8_seed0 | 4.98 / 3.24 / 17.4 | 1.80 / -0.14 / -0.6 | 0.98 / -0.99 / -3.0 | 1.33 / -0.58 / -2.1 |

## Validation, 1996 to 1997 (six training years)

One setting changes from row to row where the label starts with dots. "Bound": one security's
explained-variance term is bounded below at -1.

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
