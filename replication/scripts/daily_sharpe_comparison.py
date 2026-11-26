"""Build the slide comparison from the saved out-of-sample portfolio series."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from report import PAPER2, _gap_se, _sr


def read_run(directory: Path, dates: pd.DatetimeIndex) -> tuple[dict, pd.DataFrame]:
    metrics = json.loads((directory / "metrics.json").read_text())
    daily = pd.read_csv(directory / "daily.csv", parse_dates=["date"]).set_index("date")
    if not daily.index.equals(dates) or len(daily) != metrics["days"]:
        raise ValueError(f"Trading dates differ in {directory}")
    if not np.isfinite(daily.to_numpy()).all():
        raise ValueError(f"Nonfinite daily result in {directory}")
    expected_net = daily.gross - 0.0005 * daily.turnover - 0.0001 * daily.short
    np.testing.assert_allclose(daily.net, expected_net, rtol=2e-6, atol=2e-9)
    for column, suffix in (("gross", ""), ("net", "_net")):
        x = daily[column].to_numpy()
        actual = [_sr(x), x.mean() * 25200, x.std(ddof=1) * np.sqrt(252) * 100]
        saved = [metrics[key + suffix] for key in ("SR", "mu", "sigma")]
        np.testing.assert_allclose(actual, saved, rtol=2e-6, atol=2e-6)
    return metrics, daily


def estimate(values: list[float], spread: bool = False) -> str:
    mean = f"{np.mean(values):.2f}"
    if spread and len(values) == 2:
        return f"{mean} ± {np.ptp(values) / 2:.2f}"
    return mean


def write_comparison(original: Path, results: Path, cases: tuple[str, ...]) -> None:
    dates = pd.DatetimeIndex(pd.read_csv(original / cases[0] / "daily.csv", parse_dates=["date"])["date"])
    if len(dates) != 6040 or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Expected 6,040 distinct chronological trading dates")
    if (dates[0], dates[-1]) != (pd.Timestamp("1998-01-02"), pd.Timestamp("2021-12-31")):
        raise ValueError("Unexpected out-of-sample period")
    old, new, pca = {}, {}, {}
    for case in cases:
        old[case] = read_run(original / case, dates)
        new[case] = read_run(results / case, dates)
        pca[case] = read_run(original / case.replace("attention", "pca"), dates)
        expected = dict(old[case][0]["config"], sr_scale=1.0, name=f"daily_sharpe/{case}")
        if new[case][0]["config"] != expected or expected["lambda_var"] != 100:
            raise ValueError(f"Training settings differ for {case}")
        np.testing.assert_array_equal(old[case][1]["market"], new[case][1]["market"])

    groups = {k: [c for c in cases if old[c][0]["config"]["k"] == k] for k in (30, 8)}
    net_sharpes = [estimate([runs[c][0]["SR_net"] for c in groups[30]]) for runs in (old, new, pca)]
    lines = [
        "# Daily Sharpe in the Attention Factor objective",
        "",
        f"With 30 factors, the annualized net Sharpe moved from **{net_sharpes[0]}** in the slides "
        f"to **{net_sharpes[1]}** after refitting. The PCA + LongConv benchmark earned **{net_sharpes[2]}** "
        f"on the same dates. The paper reports **{PAPER2['attention'][30][3]:.2f}** for its attention model.",
        "",
        "## Out-of-sample results, January 1998 through December 2021",
        "",
        "The slides trained on √252 × daily net Sharpe + 100 × explained variance. "
        "The new fits use daily net Sharpe + 100 × explained variance, following the objective as printed. "
        "This gives explained variance about 15.87 times its former weight relative to trading performance. "
        "All performance Sharpe ratios below are annualized.",
        "",
        "Each test year uses the preceding eight calendar years for training, with 30 epochs, "
        "a batch size of 64 and ridge of 10⁻⁵. The 30-factor rows average the metrics for seeds 0 and 1. "
        "The ± values give half their range. The eight-factor rows use seed 0. "
        "Changes use unrounded values. Return, volatility and explained variance changes are in percentage points.",
        "",
        "| Factors | Attention Factor metric | Slides¹ | Daily objective | Change | Paper, Table 2 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    metrics = (
        ("SR", "Gross Sharpe", 0), ("SR_net", "Net Sharpe", 3),
        ("mu", "Gross mean return (%)", 1), ("mu_net", "Net mean return (%)", 4),
        ("sigma", "Gross volatility (%)", 2), ("beta", "Market beta", 6),
        ("turnover_per_day", "Turnover per day", None),
        ("explained_variance", "Explained variance (%)", None),
    )
    for k, group in groups.items():
        for key, label, paper_index in metrics:
            scale = 100 if key == "explained_variance" else 1
            a = [old[c][0][key] * scale for c in group]
            b = [new[c][0][key] * scale for c in group]
            spread = key in ("SR", "SR_net")
            paper = "—" if paper_index is None else f"{PAPER2['attention'][k][paper_index]:.2f}"
            lines.append(f"| {k} | {label} | {estimate(a, spread)} | {estimate(b, spread)} "
                         f"| {np.mean(b) - np.mean(a):+.2f} | {paper} |")
    lines += [
        "",
        "¹ The slides column uses the archived runs behind the presentation. It includes additional "
        "diagnostics from those same runs. [Slide tables](../../../slides/attention-factors-slides.pdf) "
        "· [Historical outputs](../runs/) · [Paper, trading objective and Table 2](../../../paper/2510.11616v1.pdf)",
        "",
        "The slides also report a separate 1996–1997 validation result. On that earlier sample, "
        "30-factor net Sharpe was 7.50 with the annualized objective and 5.17 with the daily objective. "
        "[Validation with annualized Sharpe](../runs/val_att_K30/metrics.json) · "
        "[Validation with daily Sharpe](../runs/val_att_K30_sr-per-day/metrics.json)",
        "",
        "## Benchmarks and the slide’s Sharpe gap",
        "",
        "PCA + LongConv already optimized daily net Sharpe on fixed PCA residuals. The OU threshold "
        "rule has no fitted trading objective. Both columns therefore use the archived benchmark portfolios.",
        "",
        "| Factors | Model | Slides gross Sharpe | New gross Sharpe | Slides net Sharpe | New net Sharpe |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for k, group in groups.items():
        ou, _ = read_run(original / f"ou_K{k}", dates)
        for label, before, after in (
            ("Attention Factor", [old[c][0] for c in group], [new[c][0] for c in group]),
            ("PCA + LongConv", [pca[c][0] for c in group], [pca[c][0] for c in group]),
            ("PCA + OU", [ou], [ou]),
        ):
            cells = [estimate([m[key] for m in runs]) for key in ("SR", "SR_net") for runs in (before, after)]
            lines.append(f"| {k} | {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "The slide’s 30-factor gap averages daily returns across seeds first, then subtracts the PCA + "
        "LongConv Sharpe from the Attention Factor Sharpe. Standard errors use 4,000 paired circular "
        "block-bootstrap draws for each of the three block lengths.",
        "",
        "| 30-factor comparison | Slides gap | New gap | Slides s.e. (1 / 21 / 63 days) | New s.e. (1 / 21 / 63 days) |",
        "|---|---:|---:|---|---|",
    ]
    group = groups[30]
    for col, label in (("gross", "Before costs"), ("net", "After costs")):
        arrays = [np.mean([runs[c][1][col].to_numpy() for c in group], axis=0) for runs in (old, new, pca)]
        baseline = arrays[-1]
        gaps = [_sr(x) - _sr(baseline) for x in arrays[:2]]
        errors = [" / ".join(f"{_gap_se(x, baseline, block):.2f}" for block in (1, 21, 63)) for x in arrays[:2]]
        lines.append(f"| {label} | {gaps[0]:+.2f} | {gaps[1]:+.2f} | {errors[0]} | {errors[1]} |")
    lines += [
        "",
        "## When the attention portfolios earned their returns",
        "",
        "The first three periods match the implementation chapter. The last period matches the "
        "post-2002 comparison in the critique. Each 30-factor cell lists seed 0 followed by seed 1.",
        "",
        "| Factors | Test years | Slides gross Sharpe | New gross Sharpe | Slides net Sharpe | New net Sharpe "
        "| Slides net return (%/yr) | New net return (%/yr) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k, group in groups.items():
        for start, end in ((1998, 2002), (2003, 2009), (2010, 2021), (2003, 2021)):
            mask = (dates.year >= start) & (dates.year <= end)
            cells = []
            for col, is_mean in (("gross", False), ("net", False), ("net", True)):
                for runs in (old, new):
                    series = [runs[c][1].loc[mask, col].to_numpy() for c in group]
                    values = [s.mean() * 25200 if is_mean else _sr(s) for s in series]
                    cells.append(", ".join(f"{v:.2f}" for v in values))
            lines.append(f"| {k} | {start}–{end} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Reproduction",
        "",
        "From the repository root, run "
        "`replication/.venv/bin/python replication/scripts/replicate_daily_sharpe.py` with the local data panel. "
        "Pass `--report-only` to rebuild this comparison from the saved portfolio outputs. "
        "The three run directories beside this report contain each portfolio’s daily returns and aggregate metrics.",
        "",
        "The training Sharpe subtracts the daily risk-free rate. For comparability with the slides, "
        "reported Sharpe uses mean portfolio return divided by sample volatility and scaled by √252. "
        "Yearly weights are joined before charging turnover, including trades across year boundaries. "
        "The experiment isolates the Sharpe scale. It retains the reconstructed characteristics and universe, "
        "cost timing, and per-stock explained-variance filters of the original implementation. "
        "[Replication conventions](../../README.md)",
        "",
        "OpenAI Codex wrote the rerun and comparison scripts and executed the experiment using the "
        "existing replication code and local panel. [Original implementation disclosure](../../AI_USAGE.md)",
    ]
    (results / "comparison.md").write_text("\n".join(lines) + "\n")
