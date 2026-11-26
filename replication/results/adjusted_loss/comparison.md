# Adjusting the Attention Factor training loss

At 30 factors the adjusted fit earned an annualized net Sharpe of **-1.73**, compared with **0.09** for daily Sharpe with 64-day batches and **0.99** in the slides. The paper reports **2.28**.

## January 1998 through December 2021

All three replications use the same 6,040 trading days. Performance Sharpe ratios are annualized. The 30-factor rows average seed-level metrics. The ± values are half the range of seeds 0 and 1. The eight-factor rows use seed 0.

| Factors | Metric | Slides | Daily Sharpe | Adjusted loss | Adjusted − daily | Paper, Table 2 |
|---:|---|---:|---:|---:|---:|---:|
| 30 | Gross Sharpe | 3.00 ± 0.04 | 2.10 ± 0.11 | 1.47 ± 0.14 | -0.63 | 3.97 |
| 30 | Net Sharpe | 0.99 ± 0.05 | 0.09 ± 0.11 | -1.73 ± 0.15 | -1.81 | 2.28 |
| 30 | Gross annual return (%) | 12.81 | 10.54 | 6.54 | -4.00 | 16.66 |
| 30 | Net annual return (%) | 4.20 | 0.42 | -7.65 | -8.07 | 9.52 |
| 30 | Annual volatility (%) | 4.26 | 5.02 | 4.43 | -0.58 | 4.20 |
| 30 | Market beta | 0.05 | 0.03 | 0.02 | -0.01 | 0.05 |
| 30 | Annual cost drag (pp) | 8.61 | 10.11 | 14.19 | +4.07 | — |
| 30 | Daily turnover | 0.59 | 0.70 | 1.03 | +0.32 | — |
| 30 | Explained variance, common diagnostic (%) | 42.85 | 44.24 | 41.45 | -2.79 | — |
| 8 | Gross Sharpe | 2.66 | 2.10 | 1.47 | -0.62 | 3.35 |
| 8 | Net Sharpe | 1.15 | 0.42 | -0.90 | -1.33 | 1.94 |
| 8 | Gross annual return (%) | 14.38 | 12.13 | 8.18 | -3.95 | 15.70 |
| 8 | Net annual return (%) | 6.21 | 2.45 | -5.02 | -7.46 | 9.05 |
| 8 | Annual volatility (%) | 5.41 | 5.78 | 5.55 | -0.23 | 4.68 |
| 8 | Market beta | 0.08 | 0.05 | 0.03 | -0.02 | 0.07 |
| 8 | Annual cost drag (pp) | 8.18 | 9.68 | 13.20 | +3.51 | — |
| 8 | Daily turnover | 0.56 | 0.67 | 0.95 | +0.28 | — |
| 8 | Explained variance, common diagnostic (%) | 37.45 | 38.26 | 35.91 | -2.35 | — |

The slides scale training Sharpe by √252. The daily-Sharpe run removes that multiplier. Both historical fits use 64-day batches and floor each stock’s explained variance at −1. The adjusted run maximizes daily net Sharpe plus 100 times uncapped explained variance over the entire preceding eight-year window. Each of 30 epochs gives one AdamW update, compared with roughly 32 updates per epoch with 64-day batches. The paper leaves batch size unspecified.

All fits apply the same eligibility rule to their respective loss window. A stock must have a valid return on at least half its dates and return variance above 10⁻¹⁰. The longer window thus changes which stocks qualify. For comparability, the explained-variance row uses the historical, capped 64-day out-of-sample diagnostic for all three fits. It is distinct from the adjusted training loss.

## Validation on 1996–1997

These four 30-factor, seed-0 fits use daily Sharpe and six preceding years for each test year. The two-by-two comparison varies the floor and loss window independently. The full, uncapped specification was fixed before evaluating the 1998–2021 returns.

| Loss window | Per-stock floor | Gross Sharpe | Net Sharpe | Net annual return (%) | Daily turnover |
|---|---|---:|---:|---:|---:|
| 64-day batches | −1 | 8.24 | 5.17 | 16.49 | 0.68 |
| 64-day batches | None | 8.58 | 5.39 | 16.62 | 0.68 |
| Six years | −1 | 4.89 | -0.98 | -2.70 | 1.18 |
| Six years | None | 4.89 | -0.98 | -2.70 | 1.18 |

The two full-window validation runs produced identical daily portfolio series. The validation difference between full-window and minibatch training also includes the change in optimizer updates per year.

## Net Sharpe by seed and subperiod

The final column uses the archived PCA + LongConv portfolio for the matching factor count and seed.

| Factors | Seed | Test years | Slides | Daily Sharpe | Adjusted loss | PCA + LongConv |
|---:|---:|---|---:|---:|---:|---:|
| 30 | 0 | 1998–2021 | 0.94 | 0.19 | -1.87 | 1.26 |
| 30 | 0 | 1998–2002 | 3.90 | 2.78 | 1.00 | 4.15 |
| 30 | 0 | 2003–2009 | 0.39 | -0.19 | -1.65 | 0.68 |
| 30 | 0 | 2010–2021 | -0.61 | -1.18 | -4.24 | -0.50 |
| 30 | 1 | 1998–2021 | 1.03 | -0.02 | -1.58 | 1.27 |
| 30 | 1 | 1998–2002 | 4.08 | 2.48 | 1.40 | 4.15 |
| 30 | 1 | 2003–2009 | 0.38 | -0.40 | -2.22 | 0.81 |
| 30 | 1 | 2010–2021 | -0.37 | -1.28 | -3.67 | -0.57 |
| 8 | 0 | 1998–2021 | 1.15 | 0.42 | -0.90 | 1.35 |
| 8 | 0 | 1998–2002 | 3.86 | 2.86 | 0.74 | 4.24 |
| 8 | 0 | 2003–2009 | 0.30 | -0.28 | -0.80 | 0.53 |
| 8 | 0 | 2010–2021 | 0.19 | -0.47 | -2.57 | 0.06 |

## Reproduction

From the repository root, run `replication/.venv/bin/python replication/scripts/replicate_adjusted_loss.py`. With `--report-only`, the script rebuilds this table from the saved public outputs. Each new run has a daily portfolio CSV and a metrics JSON with settings and annual optimizer-update counts.

The full-window implementation computes model outputs in 64-day chunks, aligns each stock across chunks, and then computes one loss. Turnover includes transitions across chunks. As in the earlier training code, the first decision date supplies the starting holdings. Evaluation joins all annual portfolios before charging trading costs. Data, characteristics, factor initialization, ridge, learning rate, cost timing and benchmark portfolios follow the daily-Sharpe replication.

[Slide tables](../../../slides/attention-factors-slides.pdf) · [Daily-Sharpe comparison](../daily_sharpe/comparison.md) · [Paper objective](../../../paper/2510.11616v1.pdf) · [Adjusted training script](../../scripts/replicate_adjusted_loss.py)

OpenAI Codex implemented and ran this experiment using the existing replication panel and model.
