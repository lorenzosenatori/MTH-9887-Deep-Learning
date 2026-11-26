Attention Factor model minus PCA with the long convolution model, out of sample 1998 to 2021, on the same days.

In a row "seed average" the daily returns of each model are averaged over its seeds first, and the Sharpe
ratios are those of that average. Table 2 shows the average of the seeds' Sharpe ratios instead, which is
lower, most of all for the Attention Factor model, whose seeds are less alike. So the difference of a
seed-average row is smaller than the difference in Table 2. The t-statistic is that of the mean of the daily
differences. The standard error is that of the difference of the two Sharpe ratios, from a circular block
bootstrap that draws the same days for both models (4,000 draws; blocks of 1, 21 and 63 days).

### K = 30

| | Mean difference, % a year | t-statistic | Sharpe ratios | Difference | Bootstrap s.e., blocks of 1 / 21 / 63 days | Difference in s.e. |
|---|---|---|---|---|---|---|
| Before costs, seed average (2 and 2 seeds) | +1.70 | 2.28 | 3.11 and 3.49 | -0.38 | 0.22 / 0.28 / 0.31 | 1.8 / 1.4 / 1.2 |
| Before costs, seed 0 against seed 0 | +1.47 | 1.87 | 2.97 and 3.47 | -0.51 | 0.23 / 0.29 / 0.31 | 2.2 / 1.7 / 1.6 |
| Before costs, seed 1 against seed 1 | +1.92 | 2.48 | 3.04 and 3.46 | -0.42 | 0.21 / 0.25 / 0.28 | 2.0 / 1.7 / 1.5 |
| After costs, seed average (2 and 2 seeds) | +0.18 | 0.25 | 1.02 and 1.27 | -0.25 | 0.20 / 0.23 / 0.25 | 1.3 / 1.1 / 1.0 |
| After costs, seed 0 against seed 0 | -0.02 | -0.02 | 0.94 and 1.26 | -0.32 | 0.21 / 0.24 / 0.24 | 1.5 / 1.3 / 1.3 |
| After costs, seed 1 against seed 1 | +0.39 | 0.50 | 1.03 and 1.27 | -0.23 | 0.20 / 0.23 / 0.25 | 1.2 / 1.0 / 0.9 |

Years in which the Attention Factor model has the higher Sharpe ratio after costs (seed average (2 and 2 seeds)): 13 of 24.

### K = 8

| | Mean difference, % a year | t-statistic | Sharpe ratios | Difference | Bootstrap s.e., blocks of 1 / 21 / 63 days | Difference in s.e. |
|---|---|---|---|---|---|---|
| Before costs, one seed each | +0.51 | 0.58 | 2.66 and 3.07 | -0.41 | 0.19 / 0.23 / 0.23 | 2.1 / 1.8 / 1.8 |
| After costs, one seed each | +0.11 | 0.12 | 1.15 and 1.35 | -0.20 | 0.18 / 0.20 / 0.18 | 1.1 / 1.0 / 1.1 |

Years in which the Attention Factor model has the higher Sharpe ratio after costs (one seed each): 12 of 24.

### After costs, by run

Sums of the daily returns after costs (not compounded, as in the figure), and the Sharpe ratio after costs.

| Run | Cumulated to the end of 2002, % | Cumulated to the end of 2021, % | Share earned by the end of 2002 | Sharpe ratio 2003-2021 |
|---|---|---|---|---|
| attention_K30_seed0 | 107.5 | 95.9 | 1.12 | -0.16 |
| attention_K30_seed1 | 110.5 | 105.7 | 1.05 | -0.07 |
| attention_K8_seed0 | 126.6 | 148.7 | 0.85 | 0.23 |
| pca_K30_seed0 | 95.4 | 96.2 | 0.99 | 0.02 |
| pca_K30_seed1 | 94.2 | 96.4 | 0.98 | 0.04 |
| pca_K8_seed0 | 125.8 | 146.1 | 0.86 | 0.27 |
