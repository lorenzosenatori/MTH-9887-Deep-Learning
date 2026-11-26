"""Collect the finished runs into tables and figures, beside the numbers of the paper.

    python scripts/report.py

Writes results/table2.md, results/table3.md, results/runs.csv, results/subperiods.md,
results/comparison.md (the model against the PCA benchmark on the same days) and
results/cumulative_returns.png and .svg. Only aggregate performance numbers and portfolio return
series are written: no CRSP or Compustat data.

The script reads runs/<name>/metrics.json and daily.csv. It also copies these two small files of
every finished run to results/runs/<name>/, which is committed. Without a runs/ folder (a fresh
clone, no data, no training) it reads results/runs/ instead, so the tables and the figure can be
rebuilt from the repository alone:

    python scripts/report.py --from-saved
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SAVED = RES / "runs"                       # metrics.json and daily.csv of the finished runs, committed
LIVE = ROOT / "runs"                       # full output of scripts/run.py, not committed
RUNS = SAVED if ("--from-saved" in sys.argv or not any(LIVE.glob("*/metrics.json"))) else LIVE


def save_small_files() -> int:
    """Copy metrics.json and daily.csv of every finished run from runs/ to results/runs/."""
    import shutil
    n = 0
    for f in sorted(LIVE.glob("*/metrics.json")):
        out = SAVED / f.parent.name
        out.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.json", "daily.csv"):
            if (f.parent / name).exists():
                shutil.copy2(f.parent / name, out / name)
        n += 1
    return n

# Table 2 of Epstein, Wang, Choi and Pelger (2025): SR, mu, sigma, SR net, mu net, sigma net, beta
PAPER2 = {
    "attention": {1: (3.05, 14.45, 4.74, 1.68, 7.94, 4.72, 0.05), 3: (3.05, 14.91, 4.89, 1.69, 8.25, 4.87, 0.06),
                  5: (2.92, 14.21, 4.87, 1.58, 7.66, 4.85, 0.07), 8: (3.35, 15.70, 4.68, 1.94, 9.05, 4.66, 0.07),
                  15: (3.81, 16.66, 4.37, 2.25, 9.78, 4.35, 0.06), 30: (3.97, 16.66, 4.20, 2.28, 9.52, 4.18, 0.05),
                  100: (4.52, 16.45, 3.64, 2.19, 7.93, 3.62, 0.05)},
    "ou": {1: (0.40, 1.85, 4.57, -2.54, -11.62, 4.57, 0.02), 3: (1.26, 4.18, 3.33, -2.72, -9.04, 3.33, 0.01),
           5: (0.99, 2.91, 2.93, -3.44, -10.11, 2.93, 0.00), 8: (0.78, 2.04, 2.61, -4.15, -10.83, 2.61, 0.00),
           10: (0.80, 1.99, 2.50, -4.32, -10.80, 2.50, 0.00), 15: (0.51, 1.12, 2.20, -5.24, -11.53, 2.20, 0.00),
           30: (0.18, 0.40, 2.24, -6.45, -14.74, 2.29, -0.00), 100: (-0.35, -0.66, 1.87, -7.05, -13.23, 1.88, -0.00)},
    "pca": {1: (2.26, 13.10, 5.79, 1.19, 6.98, 5.78, 0.10), 3: (2.76, 14.61, 5.30, 1.57, 8.29, 5.28, 0.07),
            5: (2.41, 14.10, 5.86, 1.30, 7.62, 5.84, 0.10), 8: (2.64, 14.88, 5.63, 1.50, 8.42, 5.61, 0.09),
            10: (2.66, 14.94, 5.61, 1.52, 8.48, 5.59, 0.09), 15: (2.56, 14.74, 5.75, 1.41, 8.08, 5.73, 0.09),
            30: (2.79, 15.15, 5.42, 1.57, 8.47, 5.40, 0.09), 100: (2.66, 14.36, 5.40, 1.44, 7.75, 5.38, 0.09)},
}
# Table 3 (K = 30): SR, mu, sigma, SR net, mu net, beta
PAPER3 = {"": (3.97, 16.66, 4.20, 2.28, 9.52, 0.05), "past_returns": (1.50, 7.82, 5.23, 0.59, 3.09, 0.08),
          "investment": (3.88, 17.93, 4.63, 2.19, 10.06, 0.06), "profitability": (3.94, 18.39, 4.67, 2.26, 10.48, 0.05),
          "intangibles": (3.91, 18.18, 4.65, 2.24, 10.34, 0.06), "value": (4.08, 18.45, 4.53, 2.32, 10.44, 0.04),
          "trading_frictions": (2.90, 13.36, 4.61, 1.34, 6.14, 0.06)}
# the specification selected on 1996-1997; rows of the tables use it and nothing else
MAIN = {"sr_scale": 15.874507866387544, "lambda_var": 100.0, "ridge": 1e-5, "batch": 64}
LABEL = {"attention": "Attention Factor model", "pca": "PCA with the long convolution model", "ou": "PCA with an OU threshold rule"}


def load_runs() -> pd.DataFrame:
    rows = []
    for f in sorted(RUNS.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        c = m.pop("config")
        full = c["first_year"] == 1998 and c["last_year"] == 2021
        rows.append({"run": f.parent.name, "kind": c["kind"], "k": c["k"], "seed": c["seed"], "drop": "-".join(c["drop_groups"]),
                     "full_sample": full, "ridge": c["ridge"], "lambda_var": c["lambda_var"], "sr_scale": c["sr_scale"],
                     "use_common": c["use_common"], "batch": c["batch"], "first_year": c["first_year"], "last_year": c["last_year"],
                     "pca_mode": c.get("pca", "cov"), "lookback": c.get("lookback", "own"),
                     "main_spec": abs(c["sr_scale"] - MAIN["sr_scale"]) < 1e-6 and c["lambda_var"] == MAIN["lambda_var"]
                     and c["ridge"] == MAIN["ridge"] and c["batch"] == MAIN["batch"], **m})
    return pd.DataFrame(rows)


def fmt(x: float, nd: int = 2) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def table2(df: pd.DataFrame) -> str:
    base = df[df["full_sample"] & df["main_spec"] & (df["drop"] == "") & df["run"].str.match(r"^(attention|pca|ou)_K\d+(_seed\d+)?$")]
    out = ["| Model | K | SR | SR (paper) | mean % | mean (paper) | vol % | vol (paper) | net SR | net SR (paper) | net mean % | net mean (paper) | beta | beta (paper) | turnover/day | effective factors | seeds |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for kind in ["attention", "pca", "ou"]:
        for k in sorted(set(base[base.kind == kind].k) | set(PAPER2[kind])):
            g = base[(base.kind == kind) & (base.k == k)]
            p = PAPER2[kind].get(k)
            if g.empty:
                continue
            a = g[["SR", "mu", "sigma", "SR_net", "mu_net", "beta", "turnover_per_day"]].mean()
            eff = fmt(float(g["effective_factors"].mean()), 1) if "effective_factors" in g and g["effective_factors"].notna().any() else ""
            pp = [fmt(v) for v in p] if p else [""] * 7
            out.append(f"| {LABEL[kind]} | {k} | {fmt(a.SR)} | {pp[0]} | {fmt(a.mu)} | {pp[1]} | {fmt(a.sigma)} | {pp[2]} | "
                       f"{fmt(a.SR_net)} | {pp[3]} | {fmt(a.mu_net)} | {pp[4]} | {fmt(a.beta)} | {pp[6]} | {fmt(a.turnover_per_day)} | {eff} | {len(g)} |")
    # the market of this replication: the equally weighted universe, from any finished run
    any_run = next(iter(sorted(RUNS.glob("*_seed0/daily.csv"))), None) or next(iter(sorted(RUNS.glob("ou_K*/daily.csv"))), None)
    if any_run is not None:
        d = pd.read_csv(any_run)
        if len(d) > 6000:
            mu, sd = d.market.mean() * 25200, d.market.std(ddof=1) * np.sqrt(252) * 100
            out.append(f"| Equally weighted universe | | {fmt(mu / sd)} | 0.42 | {fmt(mu)} | 8.61 | {fmt(sd)} | 20.37 | | | | | 1.00 | 1.00 | | | |")
    return "\n".join(out)


def table3(df: pd.DataFrame) -> str:
    g = df[df["full_sample"] & df["main_spec"] & (df.kind == "attention") & (df.k == 30) & df["run"].str.match(r"^attention_K30(_no-[a-z_]+)?_seed\d+$")]
    out = ["| Group removed | SR | SR (paper) | mean % | mean (paper) | net SR | net SR (paper) | net mean % | net mean (paper) | seeds |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for drop, p in PAPER3.items():
        x = g[g["drop"] == drop]
        if x.empty:
            continue
        a = x[["SR", "mu", "SR_net", "mu_net"]].mean()
        out.append(f"| {drop.replace('_', ' ') or 'none (baseline)'} | {fmt(a.SR)} | {fmt(p[0])} | {fmt(a.mu)} | {fmt(p[1])} | "
                   f"{fmt(a.SR_net)} | {fmt(p[3])} | {fmt(a.mu_net)} | {fmt(p[4])} | {len(x)} |")
    return "\n".join(out)


def variants(df: pd.DataFrame) -> str:
    """Runs with a variant of an open choice (--pca, --lookback) beside the default run of the same
    model and K, and the paper's row. Empty when no variant has been run."""
    full = df[df["full_sample"] & df["main_spec"] & (df["drop"] == "")]
    var = full[(full.pca_mode != "cov") | (full.lookback != "own")]
    if var.empty:
        return ""
    out = ["| Model | K | PCA | Lookback | Seeds | SR | mean % | vol % | net SR | net mean % | turnover/day |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (kind, k), g in var.groupby(["kind", "k"]):
        base = full[(full.kind == kind) & (full.k == k) & (full.pca_mode == "cov") & (full.lookback == "own")
                    & full["run"].str.fullmatch(rf"{kind}_K{k}(_seed\d+)?")]
        groups = [("cov", "own", base)] + [(p, lb, x) for (p, lb), x in g.groupby(["pca_mode", "lookback"])]
        for p, lb, x in groups:
            if x.empty:
                continue
            a = x[["SR", "mu", "sigma", "SR_net", "mu_net", "turnover_per_day"]].mean()
            out.append(f"| {LABEL[kind]} | {k} | {p} | {lb} | {len(x)} | {fmt(a.SR)} | {fmt(a.mu)} | {fmt(a.sigma)} | "
                       f"{fmt(a.SR_net)} | {fmt(a.mu_net)} | {fmt(a.turnover_per_day)} |")
        p2 = PAPER2[kind].get(k)
        if p2:
            out.append(f"| {LABEL[kind]}, paper | {k} | | | | {fmt(p2[0])} | {fmt(p2[1])} | {fmt(p2[2])} | {fmt(p2[3])} | {fmt(p2[4])} | |")
    return "\n".join(out)


def subperiods(names: list[str]) -> str:
    out = ["| Run | Period | SR | mean % | net SR | net mean % |", "|---|---|---|---|---|---|"]
    for name in names:
        f = RUNS / name / "daily.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, parse_dates=["date"])
        for a, b in [(1998, 2002), (2003, 2009), (2010, 2021), (1998, 2021)]:
            x = d[(d.date.dt.year >= a) & (d.date.dt.year <= b)]
            if len(x) < 100:
                continue
            s = lambda v: v.mean() / v.std(ddof=1) * np.sqrt(252)  # noqa: E731
            out.append(f"| {name} | {a}-{b} | {s(x.gross):.2f} | {x.gross.mean() * 25200:.2f} | {s(x.net):.2f} | {x.net.mean() * 25200:.2f} |")
    return "\n".join(out)


def _sr(v: np.ndarray) -> float:
    return float(v.mean() / v.std(ddof=1) * np.sqrt(252))


def _seeds(df: pd.DataFrame, kind: str, k: int) -> dict[int, pd.DataFrame]:
    """Daily returns of the runs of one model that enter Table 2: full sample, main specification."""
    g = df[df["full_sample"] & df["main_spec"] & (df["drop"] == "") & (df.kind == kind) & (df.k == k)
           & df["run"].str.fullmatch(rf"{kind}_K{k}_seed\d+")]
    return {int(r.seed): pd.read_csv(RUNS / r.run / "daily.csv", parse_dates=["date"]).set_index("date")[["gross", "net"]]
            for r in g.sort_values("seed").itertuples()}


def _gap_se(a: np.ndarray, b: np.ndarray, block: int, draws: int = 4000) -> float:
    """Standard error of SR(a) - SR(b) from a circular block bootstrap of the same days in both."""
    rng = np.random.default_rng(0)
    n = len(a)
    reps = -(-n // block)
    out = np.empty(draws)
    for i in range(draws):
        idx = ((rng.integers(0, n, reps)[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        out[i] = _sr(a[idx]) - _sr(b[idx])
    return float(out.std(ddof=1))


def comparison(df: pd.DataFrame) -> str:
    """The Attention Factor model against PCA with the long convolution model, on the same days."""
    out = ["Attention Factor model minus PCA with the long convolution model, out of sample 1998 to 2021, on the same days.",
           "",
           "In a row \"seed average\" the daily returns of each model are averaged over its seeds first, and the Sharpe",
           "ratios are those of that average. Table 2 shows the average of the seeds' Sharpe ratios instead, which is",
           "lower, most of all for the Attention Factor model, whose seeds are less alike. So the difference of a",
           "seed-average row is smaller than the difference in Table 2. The t-statistic is that of the mean of the daily",
           "differences. The standard error is that of the difference of the two Sharpe ratios, from a circular block",
           "bootstrap that draws the same days for both models (4,000 draws; blocks of 1, 21 and 63 days).",
           ""]
    for k in (30, 8):
        sa, sb = _seeds(df, "attention", k), _seeds(df, "pca", k)
        if not sa or not sb:
            continue
        idx = next(iter(sa.values())).index
        assert all(d.index.equals(idx) for d in list(sa.values()) + list(sb.values())), "runs with different dates"
        a, b = sum(sa.values()) / len(sa), sum(sb.values()) / len(sb)
        pairs = [(f"seed average ({len(sa)} and {len(sb)} seeds)" if len(sa) + len(sb) > 2 else "one seed each", a, b)]
        if len(sa) > 1 and sorted(sa) == sorted(sb):
            pairs += [(f"seed {s} against seed {s}", sa[s], sb[s]) for s in sorted(sa)]
        out += [f"### K = {k}", "",
                "| | Mean difference, % a year | t-statistic | Sharpe ratios | Difference | Bootstrap s.e., blocks of 1 / 21 / 63 days | Difference in s.e. |",
                "|---|---|---|---|---|---|---|"]
        for col, label in (("gross", "Before costs"), ("net", "After costs")):
            for name, pa, pb in pairs:
                x, y = pa[col].to_numpy(), pb[col].to_numpy()
                d = x - y
                t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                gap = _sr(x) - _sr(y)
                ses = [_gap_se(x, y, blk) for blk in (1, 21, 63)]
                out.append(f"| {label}, {name} | {d.mean() * 25200:+.2f} | {t:.2f} | {_sr(x):.2f} and {_sr(y):.2f} | {gap:+.2f} | "
                           + " / ".join(f"{v:.2f}" for v in ses) + " | " + " / ".join(f"{abs(gap) / v:.1f}" for v in ses) + " |")
        years = a.index.year
        ahead = sum(_sr(a.net[years == y].to_numpy()) > _sr(b.net[years == y].to_numpy()) for y in sorted(set(years)))
        out += ["", f"Years in which the Attention Factor model has the higher Sharpe ratio after costs ({pairs[0][0]}): "
                    f"{ahead} of {len(set(years))}.", ""]
    out += ["### After costs, by run", "",
            "Sums of the daily returns after costs (not compounded, as in the figure), and the Sharpe ratio after costs.", "",
            "| Run | Cumulated to the end of 2002, % | Cumulated to the end of 2021, % | Share earned by the end of 2002 | Sharpe ratio 2003-2021 |",
            "|---|---|---|---|---|"]
    for kind in ("attention", "pca"):
        for k in (30, 8):
            for seed, d in _seeds(df, kind, k).items():
                early, late = d.net[d.index.year <= 2002], d.net[d.index.year >= 2003]
                out.append(f"| {kind}_K{k}_seed{seed} | {100 * early.sum():.1f} | {100 * d.net.sum():.1f} | "
                           f"{early.sum() / d.net.sum():.2f} | {_sr(late.to_numpy()):.2f} |")
    return "\n".join(out)


def run_label(name: str) -> str:
    kind, k = name.split("_")[0], name.split("_")[1][1:]
    return {"attention": "Attention Factor model", "pca": "PCA + long convolution", "ou": "PCA + OU threshold rule"}[kind] + f", K = {k}"


def figure(names: list[str]) -> None:
    """Cumulated daily returns (sums, not compounded) at unit gross exposure, before and after costs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"attention": ("#1f5fa8", "-"), "pca": ("#c8641e", "-"), "ou": ("#6b6b6b", "-")}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True, sharey=True)
    seen: dict[str, int] = {}
    for name in names:
        f = RUNS / name / "daily.csv"
        if not f.exists():
            continue
        kind = name.split("_")[0]
        color, _ = style[kind]
        ls = ["-", "--", ":"][seen.get(kind, 0) % 3]
        seen[kind] = seen.get(kind, 0) + 1
        d = pd.read_csv(f, parse_dates=["date"])
        axes[0].plot(d.date, 100 * np.cumsum(d.gross), label=run_label(name), lw=1.3, color=color, ls=ls)
        axes[1].plot(d.date, 100 * np.cumsum(d.net), label=run_label(name), lw=1.3, color=color, ls=ls)
    axes[0].set_title("Before trading costs", loc="left", fontsize=11)
    axes[1].set_title("After trading costs (5 bp per trade, 1 bp short)", loc="left", fontsize=11)
    axes[0].set_ylabel("cumulated daily return, percent of gross exposure")
    for ax in axes:
        ax.axhline(0, color="#999999", lw=0.6)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Out of sample 1998-2021, this replication: almost all of the profit after costs is earned before 2003", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(RES / "cumulative_returns.png", dpi=150)
    # a fixed salt and no date: the file changes only when the data change
    plt.rcParams["svg.hashsalt"] = "attnfactors"
    fig.savefig(RES / "cumulative_returns.svg", metadata={"Date": None})


def main() -> None:
    RES.mkdir(exist_ok=True)
    df = load_runs()
    if df.empty:
        print("no finished runs")
        return
    df.to_csv(RES / "runs.csv", index=False)
    t2, t3 = table2(df), table3(df)
    (RES / "table2.md").write_text(t2 + "\n")
    (RES / "table3.md").write_text(t3 + "\n")
    names = [n for n in ["attention_K30_seed0", "attention_K8_seed0", "pca_K30_seed0", "pca_K8_seed0", "ou_K30", "ou_K8"]
             if (RUNS / n / "metrics.json").exists()]
    sens = df[df["full_sample"] & ~df["main_spec"] & (df.kind == "attention")]
    if not sens.empty:
        lines = ["| Run | Sharpe scale | lambda_Var | ridge | SR | net SR | mean % | net mean % |", "|---|---|---|---|---|---|---|---|"]
        for r in sens.itertuples():
            lines.append(f"| {r.run} | {'per day' if abs(r.sr_scale - 1) < 1e-9 else fmt(r.sr_scale)} | {r.lambda_var:g} | {r.ridge:g} | "
                         f"{fmt(r.SR)} | {fmt(r.SR_net)} | {fmt(r.mu)} | {fmt(r.mu_net)} |")
        (RES / "sensitivity.md").write_text("\n".join(lines) + "\n")
    # the second seeds are in the table of subperiods, not in the figure
    sp = subperiods(names + [n for n in ("attention_K30_seed1", "pca_K30_seed1") if (RUNS / n / "metrics.json").exists()])
    (RES / "subperiods.md").write_text(sp + "\n")
    (RES / "comparison.md").write_text(comparison(df) + "\n")
    tv = variants(df)
    if tv:
        (RES / "variants.md").write_text(tv + "\n")
        print(tv, "\n")
    figure(names)
    if RUNS == LIVE:
        print(f"copied the small result files of {save_small_files()} runs to results/runs/")
    print(t2, "\n\n", t3, "\n\n", sp)


if __name__ == "__main__":
    sys.exit(main())
