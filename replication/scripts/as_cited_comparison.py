"""Tables of all three rounds of runs, from the saved result files only (no data, no training).

    python scripts/replicate_as_cited.py report

Reads results/runs/ (round 1, the slides), results/daily_sharpe/ and results/adjusted_loss/
(round 2) and results/as_cited/ (round 3). Writes results/as_cited/comparison.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from report import PAPER2, _gap_se, _sr

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "as_cited"
ROUNDS = {  # label -> folder of the Attention Factor runs
    "Round 1: annualized Sharpe (slides)": RES / "runs",
    "Round 2a: Sharpe per day": RES / "daily_sharpe",
    "Round 2b: whole window, 30 steps (not a valid test)": RES / "adjusted_loss",
    "Round 3: as cited": OUT,
}
METRICS = [("SR", "Sharpe ratio", 2), ("SR_net", "Sharpe ratio after costs", 2), ("mu", "Mean, % a year", 2),
           ("mu_net", "Mean after costs, % a year", 2), ("sigma", "Volatility, % a year", 2), ("beta", "Market beta", 2),
           ("turnover_per_day", "Turnover a day", 2), ("cost_drag", "Cost, points a year", 2)]
PAPER_COL = {"SR": 0, "mu": 1, "sigma": 2, "SR_net": 3, "mu_net": 4, "beta": 6}


def metrics(folder: Path, run: str) -> dict | None:
    f = folder / run / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else None


def daily(folder: Path, run: str) -> pd.DataFrame | None:
    f = folder / run / "daily.csv"
    return pd.read_csv(f, parse_dates=["date"]).set_index("date") if f.exists() else None


def seeds(folder: Path, kind: str, k: int) -> list[str]:
    return [f"{kind}_K{k}_seed{s}" for s in (0, 1) if (folder / f"{kind}_K{k}_seed{s}" / "metrics.json").exists()]


def mean_metric(folder: Path, runs: list[str], key: str) -> float | None:
    vals = [metrics(folder, r)[key] for r in runs]
    return float(np.mean(vals)) if vals else None


def fmt(x, nd=2) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def attention_table(k: int) -> list[str]:
    cols = [(label, folder, seeds(folder, "attention", k)) for label, folder in ROUNDS.items()]
    cols = [c for c in cols if c[2]]
    out = [f"### Attention Factor model, K = {k}", "",
           "| | " + " | ".join(f"{label} ({len(runs)} seed{'s' if len(runs) > 1 else ''})" for label, _, runs in cols) + " | Paper, Table 2 |",
           "|---|" + "---|" * (len(cols) + 1)]
    for key, name, nd in METRICS:
        paper = fmt(PAPER2["attention"][k][PAPER_COL[key]], nd) if key in PAPER_COL else ""
        out.append(f"| {name} | " + " | ".join(fmt(mean_metric(folder, runs, key), nd) for _, folder, runs in cols) + f" | {paper} |")
    return out + [""]


def benchmark_table(k: int) -> list[str]:
    rows = [("PCA with the long convolution model", "pca", [("Round 1: projection", RES / "runs"), ("Round 3: loadings from 60 days", OUT)]),
            ("PCA with an OU threshold rule", "ou", [("Round 1: projection", RES / "runs"), ("Round 3: loadings from 60 days", OUT)])]
    out = [f"### Benchmarks, K = {k}", "",
           "| Model | Residuals | Seeds | Sharpe ratio | after costs | Mean % | after costs | Volatility % | Turnover a day |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, kind, variants in rows:
        for label, folder in variants:
            runs = seeds(folder, kind, k) if kind == "pca" else ([f"ou_K{k}"] if (folder / f"ou_K{k}" / "metrics.json").exists() else [])
            if not runs:
                continue
            g = lambda key: fmt(mean_metric(folder, runs, key))  # noqa: E731
            out.append(f"| {name} | {label} | {len(runs) if kind == 'pca' else '-'} | {g('SR')} | {g('SR_net')} | {g('mu')} | {g('mu_net')} | {g('sigma')} | {g('turnover_per_day')} |")
        p = PAPER2[kind][k]
        out.append(f"| {name} | Paper, Table 2 | | {p[0]:.2f} | {p[3]:.2f} | {p[1]:.2f} | {p[4]:.2f} | {p[2]:.2f} | |")
    return out + [""]


def average(folder: Path, runs: list[str]) -> pd.DataFrame | None:
    ds = [daily(folder, r)[["gross", "net"]] for r in runs]
    if not ds:
        return None
    assert all(d.index.equals(ds[0].index) for d in ds)
    return sum(ds) / len(ds)


def gap_table(k: int) -> list[str]:
    """Sharpe ratio of the model minus that of the benchmark on the seed-averaged daily returns."""
    models = [(label, average(folder, seeds(folder, "attention", k))) for label, folder in ROUNDS.items()]
    benches = [("PCA, projection (round 1)", average(RES / "runs", seeds(RES / "runs", "pca", k))),
               ("PCA, loadings from 60 days (round 3)", average(OUT, seeds(OUT, "pca", k)))]
    out = [f"### Attention Factor model minus PCA with the long convolution model, K = {k}", "",
           "The Sharpe ratios of this table are those of the daily returns averaged over the seeds of a model. With two",
           "seeds they are a little larger in size than the averages of the seeds' Sharpe ratios in the tables above.", "",
           "| Model | Benchmark | | Sharpe ratios | Difference | Bootstrap s.e., blocks of 1 / 21 / 63 days | Difference in s.e. |",
           "|---|---|---|---|---|---|---|"]
    for ml, a in models:
        for bl, b in benches:
            if a is None or b is None or not a.index.equals(b.index):
                continue
            for col, label in (("gross", "before costs"), ("net", "after costs")):
                x, y = a[col].to_numpy(), b[col].to_numpy()
                gap = _sr(x) - _sr(y)
                ses = [_gap_se(x, y, blk) for blk in (1, 21, 63)]
                out.append(f"| {ml} | {bl} | {label} | {_sr(x):.2f} and {_sr(y):.2f} | {gap:+.2f} | "
                           + " / ".join(f"{v:.2f}" for v in ses) + " | " + " / ".join(f"{abs(gap) / v:.1f}" for v in ses) + " |")
    p = PAPER2
    out += ["", f"Paper, Table 2: {p['attention'][k][0]:.2f} and {p['pca'][k][0]:.2f} before costs "
                f"({p['attention'][k][0] - p['pca'][k][0]:+.2f}), {p['attention'][k][3]:.2f} and {p['pca'][k][3]:.2f} after "
                f"({p['attention'][k][3] - p['pca'][k][3]:+.2f}).", ""]
    return out


def subperiods() -> list[str]:
    out = ["### Round 3 by period", "",
           "Sharpe ratio before costs / after costs / mean after costs in % a year.", "",
           "| Run | 1998-2002 | 2003-2009 | 2010-2021 | 2003-2021 |", "|---|---|---|---|---|"]
    for run in ["attention_K30_seed0", "attention_K30_seed1", "attention_K8_seed0", "pca_K30_seed0", "pca_K30_seed1", "pca_K8_seed0"]:
        d = daily(OUT, run)
        if d is None:
            continue
        cells = []
        for a, b in [(1998, 2002), (2003, 2009), (2010, 2021), (2003, 2021)]:
            x = d[(d.index.year >= a) & (d.index.year <= b)]
            cells.append(f"{_sr(x.gross.to_numpy()):.2f} / {_sr(x.net.to_numpy()):.2f} / {x.net.mean() * 25200:.1f}")
        out.append(f"| {run} | " + " | ".join(cells) + " |")
    return out + [""]


def validation() -> list[str]:
    v, r1, r2 = OUT / "validation", RES / "runs", RES / "adjusted_loss" / "validation"
    att = [("Round 1 specification: annualized Sharpe, batch 64, bound", r1, "val_att_K30"),
           ("Sharpe per day (the printed loss), batch 64, bound", r1, "val_att_K30_sr-per-day"),
           ("... and no bound (run of round 2b)", r2, "short_uncapped"),
           ("... and batch 125, bound", v, "att_b125_floor"),
           ("... and batch 125, no bound", v, "att_b125_nofloor"),
           ("... and cited inputs (batch 64, bound)", v, "att_b64_floor_cited"),
           ("Round 3 specification: per day, batch 125, no bound, cited inputs", v, "att_b125_nofloor_cited"),
           ("Round 3 specification with the annualized Sharpe", v, "att_b125_nofloor_cited_annualized"),
           ("Whole six-year window as one batch, 30 optimizer steps a year (round 2b)", r2, "full_uncapped")]
    pca = [("PCA with the long convolution model: projection, batch 64 (round 1)", r1, "val_pca_K30"),
           ("... projection, batch 125", v, "pca_b125_projection"),
           ("... correlation matrix, loadings from all 252 days, batch 125", v, "pca_b125_regression252"),
           ("... correlation matrix, loadings from 60 days, batch 64", v, "pca_b64_regression"),
           ("... correlation matrix, loadings from 60 days, batch 125 (round 3)", v, "pca_b125_regression"),
           ("PCA with an OU threshold rule: projection (K = 8, round 1)", r1, "val_ou_K8"),
           ("... loadings from 60 days (K = 8)", v, "ou_K8_regression"),
           ("... loadings from 60 days (K = 30)", v, "ou_K30_regression")]
    k8 = [("Attention Factor model, round 1 specification", r1, "val_att_K8"),
          ("Attention Factor model, round 3 specification", v, "att_K8_b125_nofloor_cited"),
          ("PCA with the long convolution model, projection (round 1)", r1, "val_pca_K8"),
          ("PCA with the long convolution model, loadings from 60 days (round 3)", v, "pca_K8_b125_regression")]
    out = []
    for title, rows in (("Attention Factor model, K = 30, seed 0", att), ("Benchmarks, K = 30 unless noted", pca), ("K = 8", k8)):
        out += [f"### {title}", "", "| Configuration | Sharpe ratio | after costs | Mean after costs % | Volatility % | Turnover a day |", "|---|---|---|---|---|---|"]
        for label, folder, run in rows:
            m = metrics(folder, run)
            if m:
                out.append(f"| {label} | {m['SR']:.2f} | {m['SR_net']:.2f} | {m['mu_net']:.2f} | {m['sigma']:.2f} | {m['turnover_per_day']:.2f} |")
        out.append("")
    return out


def summary() -> list[str]:
    """The table of README_ISSUES.md: every round beside the paper, one block per number of factors."""
    def line(label: str, folder: Path, runs: list[str]) -> str | None:
        if not runs:
            return None
        g = lambda key: fmt(mean_metric(folder, runs, key))  # noqa: E731
        return f"| {label} | {g('SR')} | {g('SR_net')} | {g('mu')} | {g('mu_net')} | {g('sigma')} | {g('turnover_per_day')} |"

    def paper(kind: str, k: int) -> str:
        x = PAPER2[kind][k]
        return f"| **Paper, Table 2** | **{x[0]:.2f}** | **{x[3]:.2f}** | **{x[1]:.2f}** | **{x[4]:.2f}** | **{x[2]:.2f}** | |"

    out = []
    for k in (30, 8):
        n = {1: "one seed", 2: "two seeds"}[len(seeds(OUT, "attention", k)) or len(seeds(RES / "runs", "attention", k))]
        out += [f"**{k} factors** ({n} for the trained models)", "",
                "| | Sharpe ratio | after costs | Mean % | after costs | Volatility % | Turnover a day |", "|---|---|---|---|---|---|---|",
                "| *Attention Factor model* | | | | | | |", paper("attention", k)]
        for label, folder in ROUNDS.items():
            out.append(line(label, folder, seeds(folder, "attention", k)))
        out += ["| *PCA with the long convolution model* | | | | | | |", paper("pca", k),
                line("Round 1: projection on the eigenvectors", RES / "runs", seeds(RES / "runs", "pca", k)),
                line("Round 3: loadings from a regression on 60 days", OUT, seeds(OUT, "pca", k)),
                "| *PCA with an OU threshold rule* | | | | | | |", paper("ou", k),
                line("Round 1: projection on the eigenvectors", RES / "runs", [f"ou_K{k}"] if (RES / "runs" / f"ou_K{k}" / "metrics.json").exists() else []),
                line("Round 3: loadings from a regression on 60 days", OUT, [f"ou_K{k}"] if (OUT / f"ou_K{k}" / "metrics.json").exists() else []),
                ""]
    return [x for x in out if x is not None]


def update_readme() -> None:
    """Replace the generated blocks of README_ISSUES.md, so that its tables are those of the files."""
    f = RES.parent / "README_ISSUES.md"
    if not f.exists():
        return
    s = f.read_text()
    for name, lines in (("results", summary()), ("validation", validation())):
        a, b = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
        if a in s and b in s:
            s = s[:s.index(a) + len(a)] + "\n" + "\n".join(lines).rstrip() + "\n" + s[s.index(b):]
    f.write_text(s)
    print(f"updated the generated tables of {f.name}")


def report() -> None:
    out = ["# Three rounds of runs beside the paper", "",
           "Written by `scripts/as_cited_comparison.py` from the saved result files. The reading of these tables is in",
           "`README_ISSUES.md`. Out of sample means January 1998 to December 2021 (6,040 days); all numbers are annualized.",
           "A cell of a model with two seeds is the average of the two runs.", "",
           "## Out of sample, 1998 to 2021", ""]
    for k in (30, 8):
        out += attention_table(k) + benchmark_table(k)
    for k in (30, 8):
        out += gap_table(k)
    out += subperiods()
    out += ["## Validation, 1996 to 1997 (six training years)", "",
            "One setting changes from row to row where the label starts with dots. \"Bound\": one security's",
            "explained-variance term is bounded below at -1.", ""] + validation()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.md").write_text("\n".join(out).rstrip() + "\n")
    print(f"wrote {OUT / 'comparison.md'}")
    update_readme()


if __name__ == "__main__":
    report()
