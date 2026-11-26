# Daily Sharpe in the Attention Factor objective

With 30 factors, the annualized net Sharpe moved from **0.99** in the slides to **0.09** after refitting. The PCA + LongConv benchmark earned **1.26** on the same dates. The paper reports **2.28** for its attention model.

## Out-of-sample results, January 1998 through December 2021

The slides trained on √252 × daily net Sharpe + 100 × explained variance. The new fits use daily net Sharpe + 100 × explained variance, following the objective as printed. This gives explained variance about 15.87 times its former weight relative to trading performance. All performance Sharpe ratios below are annualized.

Each test year uses the preceding eight calendar years for training, with 30 epochs, a batch size of 64 and ridge of 10⁻⁵. The 30-factor rows average the metrics for seeds 0 and 1. The ± values give half their range. The eight-factor rows use seed 0. Changes use unrounded values. Return, volatility and explained variance changes are in percentage points.

| Factors | Attention Factor metric | Slides¹ | Daily objective | Change | Paper, Table 2 |
|---:|---|---:|---:|---:|---:|
| 30 | Gross Sharpe | 3.00 ± 0.04 | 2.10 ± 0.11 | -0.90 | 3.97 |
| 30 | Net Sharpe | 0.99 ± 0.05 | 0.09 ± 0.11 | -0.90 | 2.28 |
| 30 | Gross mean return (%) | 12.81 | 10.54 | -2.27 | 16.66 |
| 30 | Net mean return (%) | 4.20 | 0.42 | -3.78 | 9.52 |
| 30 | Gross volatility (%) | 4.26 | 5.02 | +0.75 | 4.20 |
| 30 | Market beta | 0.05 | 0.03 | -0.02 | 0.05 |
| 30 | Turnover per day | 0.59 | 0.70 | +0.12 | — |
| 30 | Explained variance (%) | 42.85 | 44.24 | +1.39 | — |
| 8 | Gross Sharpe | 2.66 | 2.10 | -0.56 | 3.35 |
| 8 | Net Sharpe | 1.15 | 0.42 | -0.73 | 1.94 |
| 8 | Gross mean return (%) | 14.38 | 12.13 | -2.26 | 15.70 |
| 8 | Net mean return (%) | 6.21 | 2.45 | -3.76 | 9.05 |
| 8 | Gross volatility (%) | 5.41 | 5.78 | +0.38 | 4.68 |
| 8 | Market beta | 0.08 | 0.05 | -0.03 | 0.07 |
| 8 | Turnover per day | 0.56 | 0.67 | +0.11 | — |
| 8 | Explained variance (%) | 37.45 | 38.26 | +0.82 | — |

¹ The slides column uses the archived runs behind the presentation. It includes additional diagnostics from those same runs. [Slide tables](../../../slides/attention-factors-slides.pdf) · [Historical outputs](../runs/) · [Paper, trading objective and Table 2](../../../paper/2510.11616v1.pdf)

The slides also report a separate 1996–1997 validation result. On that earlier sample, 30-factor net Sharpe was 7.50 with the annualized objective and 5.17 with the daily objective. [Validation with annualized Sharpe](../runs/val_att_K30/metrics.json) · [Validation with daily Sharpe](../runs/val_att_K30_sr-per-day/metrics.json)

## Benchmarks and the slide’s Sharpe gap

PCA + LongConv already optimized daily net Sharpe on fixed PCA residuals. The OU threshold rule has no fitted trading objective. Both columns therefore use the archived benchmark portfolios.

| Factors | Model | Slides gross Sharpe | New gross Sharpe | Slides net Sharpe | New net Sharpe |
|---:|---|---:|---:|---:|---:|
| 30 | Attention Factor | 3.00 | 2.10 | 0.99 | 0.09 |
| 30 | PCA + LongConv | 3.47 | 3.47 | 1.26 | 1.26 |
| 30 | PCA + OU | 1.31 | 1.31 | -3.95 | -3.95 |
| 8 | Attention Factor | 2.66 | 2.10 | 1.15 | 0.42 |
| 8 | PCA + LongConv | 3.07 | 3.07 | 1.35 | 1.35 |
| 8 | PCA + OU | 1.16 | 1.16 | -3.16 | -3.16 |

The slide’s 30-factor gap averages daily returns across seeds first, then subtracts the PCA + LongConv Sharpe from the Attention Factor Sharpe. Standard errors use 4,000 paired circular block-bootstrap draws for each of the three block lengths.

| 30-factor comparison | Slides gap | New gap | Slides s.e. (1 / 21 / 63 days) | New s.e. (1 / 21 / 63 days) |
|---|---:|---:|---|---|
| Before costs | -0.38 | -1.30 | 0.22 / 0.28 / 0.31 | 0.22 / 0.26 / 0.28 |
| After costs | -0.25 | -1.18 | 0.20 / 0.23 / 0.25 | 0.21 / 0.23 / 0.24 |

## When the attention portfolios earned their returns

The first three periods match the implementation chapter. The last period matches the post-2002 comparison in the critique. Each 30-factor cell lists seed 0 followed by seed 1.

| Factors | Test years | Slides gross Sharpe | New gross Sharpe | Slides net Sharpe | New net Sharpe | Slides net return (%/yr) | New net return (%/yr) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 30 | 1998–2002 | 5.69, 5.87 | 4.49, 4.10 | 3.90, 4.08 | 2.78, 2.48 | 21.56, 22.16 | 17.54, 15.88 |
| 30 | 2003–2009 | 2.37, 2.57 | 1.81, 1.76 | 0.39, 0.38 | -0.19, -0.40 | 1.76, 1.59 | -0.96, -1.96 |
| 30 | 2010–2021 | 1.79, 1.78 | 1.17, 0.94 | -0.61, -0.37 | -1.18, -1.28 | -2.00, -1.33 | -4.81, -5.68 |
| 30 | 2003–2021 | 2.02, 2.09 | 1.43, 1.26 | -0.16, -0.07 | -0.75, -0.93 | -0.61, -0.25 | -3.39, -4.31 |
| 8 | 1998–2002 | 5.30 | 4.39 | 3.86 | 2.86 | 25.39 | 19.95 |
| 8 | 2003–2009 | 1.94 | 1.59 | 0.30 | -0.28 | 1.69 | -1.61 |
| 8 | 2010–2021 | 1.72 | 1.20 | 0.19 | -0.47 | 0.86 | -2.46 |
| 8 | 2003–2021 | 1.80 | 1.35 | 0.23 | -0.40 | 1.17 | -2.15 |

## Reproduction

From the repository root, run `replication/.venv/bin/python replication/scripts/replicate_daily_sharpe.py` with the local data panel. Pass `--report-only` to rebuild this comparison from the saved portfolio outputs. The three run directories beside this report contain each portfolio’s daily returns and aggregate metrics.

The training Sharpe subtracts the daily risk-free rate. For comparability with the slides, reported Sharpe uses mean portfolio return divided by sample volatility and scaled by √252. Yearly weights are joined before charging turnover, including trades across year boundaries. The experiment isolates the Sharpe scale. It retains the reconstructed characteristics and universe, cost timing, and per-stock explained-variance filters of the original implementation. [Replication conventions](../../README.md)

OpenAI Codex wrote the rerun and comparison scripts and executed the experiment using the existing replication code and local panel. [Original implementation disclosure](../../AI_USAGE.md)
