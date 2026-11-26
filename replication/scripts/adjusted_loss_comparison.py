"""Generate the adjusted-loss comparison from aggregate and daily portfolio results."""
from pathlib import Path

import numpy as np
import pandas as pd

from daily_sharpe_comparison import estimate, read_run
from replicate_daily_sharpe import CASES, ORIGINAL, RESULTS as DAILY
from report import PAPER2, _sr

ADJUSTED = ORIGINAL.parent / "adjusted_loss"


def dates_from(run: Path, count: int) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.read_csv(run / "daily.csv", parse_dates=["date"]).date)
    if len(dates) != count or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError(f"Unexpected trading dates for {run}")
    return dates


def verify_settings(adjusted, daily, full, capped):
    cfg = adjusted["config"]
    for key, value in daily["config"].items():
        expected = None if full and key == "batch" else value
        if key != "name" and cfg[key] != expected:
            raise ValueError(f"Unexpected {key} in {cfg['name']}")
    expected = dict(loss_window="full training window" if full else "minibatch",
                    ev_floor=-1 if capped else None, min_valid_share=0.5,
                    min_return_variance=1e-10, sr_scale=1.0, lambda_var=100.0)
    if any(cfg[k] != v for k, v in expected.items()):
        raise ValueError(f"Unexpected objective for {cfg['name']}")
    years = [str(year) for year in range(cfg["first_year"], cfg["last_year"] + 1)]
    steps = adjusted["optimizer_steps"]
    if set(steps) != set(years) or any((steps[y] != cfg["epochs"] if full else steps[y] <= cfg["epochs"]) for y in years):
        raise ValueError(f"Incomplete training for {cfg['name']}")


def report():
    dates = dates_from(ORIGINAL / CASES[0], 6040)
    runs = [[read_run(base / case, dates) for case in CASES] for base in (ORIGINAL, DAILY, ADJUSTED)]
    for i in range(len(CASES)):
        verify_settings(runs[2][i][0], runs[1][i][0], True, False)
        for stage in runs[1:]:
            np.testing.assert_array_equal(runs[0][i][1].market, stage[i][1].market)
    metrics = [
        ("SR", "Gross Sharpe", 0), ("SR_net", "Net Sharpe", 3),
        ("mu", "Gross annual return (%)", 1), ("mu_net", "Net annual return (%)", 4),
        ("sigma", "Annual volatility (%)", 2), ("beta", "Market beta", 6),
        ("cost_drag", "Annual cost drag (pp)", None), ("turnover_per_day", "Daily turnover", None),
        ("explained_variance", "Explained variance, common diagnostic (%)", None),
    ]
    net30 = [estimate([m["SR_net"] for m, _ in stage if m["config"]["k"] == 30]) for stage in runs]
    lines = ["# Adjusting the Attention Factor training loss", "",
             f"At 30 factors the adjusted fit earned an annualized net Sharpe of **{net30[2]}**, "
             f"compared with **{net30[1]}** for daily Sharpe with 64-day batches and **{net30[0]}** "
             "in the slides. The paper reports **2.28**.", "",
             "## January 1998 through December 2021", "",
             "All three replications use the same 6,040 trading days. Performance Sharpe ratios are annualized. "
             "The 30-factor rows average seed-level metrics. The ± values are half the range of seeds 0 and 1. "
             "The eight-factor rows use seed 0.", "",
             "| Factors | Metric | Slides | Daily Sharpe | Adjusted loss | Adjusted − daily | Paper, Table 2 |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for k in (30, 8):
        for key, label, paper_col in metrics:
            scale = 100 if key == "explained_variance" else 1
            values = [[m[key] * scale for m, _ in stage if m["config"]["k"] == k] for stage in runs]
            cells = [estimate(v, spread=key in ("SR", "SR_net")) for v in values]
            paper = "—" if paper_col is None else f"{PAPER2['attention'][k][paper_col]:.2f}"
            delta = np.mean(values[2]) - np.mean(values[1])
            lines.append(f"| {k} | {label} | " + " | ".join(cells) + f" | {delta:+.2f} | {paper} |")
    lines += ["", "The slides scale training Sharpe by √252. The daily-Sharpe run removes that multiplier. "
              "Both historical fits use 64-day batches and floor each stock’s explained variance at −1. "
              "The adjusted run maximizes daily net Sharpe plus 100 times uncapped explained variance over the "
              "entire preceding eight-year window. Each of 30 epochs gives one AdamW update, compared with "
              "roughly 32 updates per epoch with 64-day batches. The paper leaves batch size unspecified.", "",
              "All fits apply the same eligibility rule to their respective loss window. A stock must have a valid "
              "return on at least half its dates and return variance above 10⁻¹⁰. The longer window thus changes "
              "which stocks qualify. For comparability, the explained-variance row uses the historical, capped "
              "64-day out-of-sample diagnostic for all three fits. It is distinct from the adjusted training loss.", "",
              "## Validation on 1996–1997", "",
              "These four 30-factor, seed-0 fits use daily Sharpe and six preceding years for each test year. "
              "The two-by-two comparison varies the floor and loss window independently. The full, uncapped "
              "specification was fixed before evaluating the 1998–2021 returns.", "",
              "| Loss window | Per-stock floor | Gross Sharpe | Net Sharpe | Net annual return (%) | Daily turnover |",
              "|---|---|---:|---:|---:|---:|"]
    historical = ORIGINAL / "val_att_K30_sr-per-day"
    val_dates = dates_from(historical, 507)
    reference, reference_daily = read_run(historical, val_dates)
    validation_series = {}
    for spec in ("short_capped", "short_uncapped", "full_capped", "full_uncapped"):
        full = spec.startswith("full_")
        capped = spec.endswith("_capped")
        if spec == "short_capped":
            m, d = reference, reference_daily
        else:
            m, d = read_run(ADJUSTED / "validation" / spec, val_dates)
            np.testing.assert_array_equal(reference_daily.market, d.market)
            verify_settings(m, reference, full, capped)
        validation_series[spec] = d
        lines.append(f"| {'Six years' if full else '64-day batches'} | {'−1' if capped else 'None'} | "
                     f"{m['SR']:.2f} | {m['SR_net']:.2f} | {m['mu_net']:.2f} | {m['turnover_per_day']:.2f} |")
    if validation_series["full_capped"].equals(validation_series["full_uncapped"]):
        lines += ["", "The two full-window validation runs produced identical daily portfolio series. "
                  "The validation difference between full-window and minibatch training also includes "
                  "the change in optimizer updates per year."]
    lines += ["", "## Net Sharpe by seed and subperiod", "",
              "The final column uses the archived PCA + LongConv portfolio for the matching factor count and seed.",
              "", "| Factors | Seed | Test years | Slides | Daily Sharpe | Adjusted loss | PCA + LongConv |",
              "|---:|---:|---|---:|---:|---:|---:|"]
    for i, case in enumerate(CASES):
        m = runs[2][i][0]
        _, pca = read_run(ORIGINAL / case.replace("attention", "pca"), dates)
        for start, end in ((1998, 2021), (1998, 2002), (2003, 2009), (2010, 2021)):
            mask = (dates.year >= start) & (dates.year <= end)
            values = [_sr(d.loc[mask, "net"].to_numpy()) for d in [stage[i][1] for stage in runs] + [pca]]
            lines.append(f"| {m['config']['k']} | {m['config']['seed']} | {start}–{end} | "
                         + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines += ["", "## Reproduction", "",
              "From the repository root, run `replication/.venv/bin/python replication/scripts/replicate_adjusted_loss.py`. "
              "With `--report-only`, the script rebuilds this table from the saved public outputs. Each new run "
              "has a daily portfolio CSV and a metrics JSON with settings and annual optimizer-update counts.", "",
              "The full-window implementation computes model outputs in 64-day chunks, aligns each stock across "
              "chunks, and then computes one loss. Turnover includes transitions across chunks. As in the earlier "
              "training code, the first decision date supplies the starting holdings. Evaluation joins all annual "
              "portfolios before charging trading costs. Data, characteristics, factor initialization, ridge, "
              "learning rate, cost timing and benchmark portfolios follow the daily-Sharpe replication.", "",
              "[Slide tables](../../../slides/attention-factors-slides.pdf) · "
              "[Daily-Sharpe comparison](../daily_sharpe/comparison.md) · "
              "[Paper objective](../../../paper/2510.11616v1.pdf) · "
              "[Adjusted training script](../../scripts/replicate_adjusted_loss.py)", "",
              "OpenAI Codex implemented and ran this experiment using the existing replication panel and model."]
    (ADJUSTED / "comparison.md").write_text("\n".join(lines) + "\n")
