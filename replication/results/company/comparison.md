# Round 4 beside the paper

Written by `scripts/company_comparison.py` from the saved result files; the reading is in `DIFFERENCES.md`.
Out of sample, January 1998 to December 2021, annualized. One seed for every trained model, a second for one of them.

## Sharpe ratios before / after costs

| | 8 factors: paper | this round | 30 factors: paper | this round |
|---|---|---|---|---|
| Attention Factor model, objective as printed | 3.35 / 1.94 | 1.91 / 0.42 | 3.97 / 2.28 | 2.03 / -0.03 |
| Attention Factor model, annualized Sharpe ratio in the objective | 3.35 / 1.94 | 2.58 / 1.14 | 3.97 / 2.28 | 3.01 / 0.96 |
| PCA with the long convolution model, loadings from 60 days | 2.64 / 1.50 | 2.33 / 0.32 | 2.79 / 1.57 | 1.76 / -1.38 |
| PCA with the long convolution model, projection | 2.64 / 1.50 | 2.89 / 1.13 | 2.79 / 1.57 | 3.64 / 1.19 |
| PCA with an OU threshold rule, loadings from 60 days | 0.78 / -4.15 | 1.30 / -3.74 | 0.18 / -6.45 | 1.07 / -5.80 |
| PCA with an OU threshold rule, projection | 0.78 / -4.15 | 1.30 / -3.24 | 0.18 / -6.45 | 1.66 / -4.10 |

Second seed of the Attention Factor model with 30 factors, objective as printed: 2.21 / 0.08.

## Every number of the subset

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

## Mean, volatility and cost

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

## The same specification on both universes

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

## Model minus benchmark

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

## By period

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

## The forced delistings of January 2021

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
