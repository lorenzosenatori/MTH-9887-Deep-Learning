"""Tables of the fourth round (scripts/replicate_company_universe.py) beside the paper, from saved files.

    python scripts/replicate_company_universe.py report

Writes results/company/comparison.md and fills the generated blocks of DIFFERENCES.md.
"""
from __future__ import annotations

from pathlib import Path

from as_cited_comparison import daily, fmt, metrics
from report import PAPER2, _gap_se, _sr

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "company"
KEYS = [("SR", "Sharpe ratio", 0), ("SR_net", "Sharpe ratio after costs", 3), ("mu", "Mean, % a year", 1),
        ("mu_net", "Mean after costs, % a year", 4), ("sigma", "Volatility, % a year", 2), ("beta", "Market beta", 6)]
# (label, kind in the paper's Table 2, run of round 4 for K, the same specification on the universe of rounds 1 to 3)
ROWS = [
    ("Attention Factor model, objective as printed", "attention", "attention_K{k}_seed0", (RES / "as_cited", "attention_K{k}_seed0")),
    ("Attention Factor model, annualized Sharpe ratio in the objective", "attention", "attention_annualized_K{k}_seed0", None),
    ("PCA with the long convolution model, loadings from 60 days", "pca", "pca_K{k}_seed0", (RES / "as_cited", "pca_K{k}_seed0")),
    ("PCA with the long convolution model, projection", "pca", "pcaproj_K{k}_seed0", (RES / "runs", "pca_K{k}_seed0")),
    ("PCA with an OU threshold rule, loadings from 60 days", "ou", "ou_K{k}", (RES / "as_cited", "ou_K{k}")),
    ("PCA with an OU threshold rule, projection", "ou", "ouproj_K{k}", (RES / "runs", "ou_K{k}")),
]


def results(k: int) -> list[str]:
    """One block per model: the paper, this round, the difference, and the same specification on the old universe."""
    out = [f"**{k} factors**", "",
           "| | Paper | This round | This round minus paper | Same specification, universe of rounds 1 to 3 |", "|---|---|---|---|---|"]
    for label, kind, run, old in ROWS:
        m = metrics(OUT, run.format(k=k))
        if m is None:
            continue
        o = metrics(old[0], old[1].format(k=k)) if old else None
        out.append(f"| *{label}* | | | | |")
        for key, name, col in KEYS:
            p = PAPER2[kind][k][col]
            out.append(f"| {name} | {p:.2f} | {m[key]:.2f} | {m[key] - p:+.2f} | {fmt(o[key]) if o else ''} |")
        out.append(f"| Turnover a day | | {m['turnover_per_day']:.2f} | | {fmt(o['turnover_per_day']) if o else ''} |")
    return out + [""]


def summary() -> list[str]:
    """Sharpe ratios only: before / after costs, both numbers of factors, every reading."""
    out = ["| | 8 factors: paper | this round | 30 factors: paper | this round |", "|---|---|---|---|---|"]
    for label, kind, run, _ in ROWS:
        cells = []
        for k in (8, 30):
            m = metrics(OUT, run.format(k=k))
            p = PAPER2[kind][k]
            cells += [f"{p[0]:.2f} / {p[3]:.2f}", f"{m['SR']:.2f} / {m['SR_net']:.2f}" if m else "not run"]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    s1 = metrics(OUT, "attention_K30_seed1")
    if s1:
        out += ["", f"Second seed of the Attention Factor model with 30 factors, objective as printed: {s1['SR']:.2f} / {s1['SR_net']:.2f}."]
    return out + [""]


def decomposition() -> list[str]:
    """Mean, volatility and cost of a year (mean minus mean after costs): the paper / this round."""
    out = ["Each cell: the paper / this round. Cost: the mean minus the mean after costs.", "",
           "| | Mean, % a year | Volatility, % a year | Cost, % a year |", "|---|---|---|---|"]
    for k in (8, 30):
        for label, kind, run, _ in ROWS:
            m, p = metrics(OUT, run.format(k=k)), PAPER2[kind][k]
            if m is not None:
                out.append(f"| {label}, K = {k} | {p[1]:.2f} / {m['mu']:.2f} | {p[2]:.2f} / {m['sigma']:.2f} | "
                           f"{p[1] - p[4]:.2f} / {m['mu'] - m['mu_net']:.2f} |")
    return out + [""]


def universe_effect() -> list[str]:
    """The same specification and seed on both universes: Sharpe ratio before / after costs."""
    out = ["| | 8 factors: universe of rounds 1 to 3 | this round | 30 factors: universe of rounds 1 to 3 | this round |",
           "|---|---|---|---|---|"]
    for label, _, run, old in ROWS:
        if old is None:
            continue
        cells = []
        for k in (8, 30):
            m, o = metrics(OUT, run.format(k=k)), metrics(old[0], old[1].format(k=k))
            cells += [f"{o['SR']:.2f} / {o['SR_net']:.2f}" if o else "", f"{m['SR']:.2f} / {m['SR_net']:.2f}" if m else "not run"]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return out + [""] + OLD_NOTE + [""]


OLD_NOTE = ["The runs on the universe of rounds 1 to 3 are those of round 3 (`results/as_cited/`), with the same settings and the",
            "same seed. For the two rows with a projection they are the runs of round 1 (`results/runs/`), where the residuals",
            "are built the same way. The trained one of the two used batches of 64 days and an annualized Sharpe ratio in its",
            "objective; that objective has one term, so the scale only rescales it."]


def gaps() -> list[str]:
    out = ["| | | Sharpe ratios | Difference | Paper | Bootstrap s.e., blocks of 1 / 21 / 63 days |", "|---|---|---|---|---|---|"]
    for k in (30, 8):
        for ml, mrun in (("as printed", "attention_K{k}_seed0"), ("annualized", "attention_annualized_K{k}_seed0")):
            for bl, brun in (("loadings from 60 days", "pca_K{k}_seed0"), ("projection", "pcaproj_K{k}_seed0")):
                a, b = daily(OUT, mrun.format(k=k)), daily(OUT, brun.format(k=k))
                if a is None or b is None:
                    continue
                for col, name, pc in (("gross", "before costs", 0), ("net", "after costs", 3)):
                    x, y = a[col].to_numpy(), b[col].to_numpy()
                    paper = PAPER2["attention"][k][pc] - PAPER2["pca"][k][pc]
                    out.append(f"| K = {k}: model {ml}, benchmark {bl} | {name} | {_sr(x):.2f} and {_sr(y):.2f} | {_sr(x) - _sr(y):+.2f} | "
                               f"{paper:+.2f} | " + " / ".join(f"{_gap_se(x, y, blk):.2f}" for blk in (1, 21, 63)) + " |")
    return out + [""]


def periods() -> list[str]:
    out = ["Sharpe ratio before costs / after costs / mean after costs in % a year.", "",
           "| Run | 1998-2002 | 2003-2009 | 2010-2021 | 2003-2021 |", "|---|---|---|---|---|"]
    for label, _, run, _ in ROWS[:4]:
        for k in (30, 8):
            d = daily(OUT, run.format(k=k))
            if d is None:
                continue
            cells = []
            for a, b in [(1998, 2002), (2003, 2009), (2010, 2021), (2003, 2021)]:
                x = d[(d.index.year >= a) & (d.index.year <= b)]
                cells.append(f"{_sr(x.gross.to_numpy()):.2f} / {_sr(x.net.to_numpy()):.2f} / {x.net.mean() * 25200:.1f}")
            out.append(f"| {label}, K = {k} | " + " | ".join(cells) + " |")
    return out + [""]


def report() -> None:
    out = ["# Round 4 beside the paper", "",
           "Written by `scripts/company_comparison.py` from the saved result files; the reading is in `DIFFERENCES.md`.",
           "Out of sample, January 1998 to December 2021, annualized. One seed for every trained model, a second for one of them.", "",
           "## Sharpe ratios before / after costs", ""] + summary() + ["## Every number of the subset", ""]
    for k in (8, 30):
        out += results(k)
    out += OLD_NOTE + ["", "## Mean, volatility and cost", ""] + decomposition()
    out += ["## The same specification on both universes", ""] + universe_effect()
    out += ["## Model minus benchmark", ""] + gaps() + ["## By period", ""] + periods()
    g = OUT / "checks" / "delisting_2021.md"
    if g.exists():
        out += [g.read_text().rstrip(), ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.md").write_text("\n".join(out).rstrip() + "\n")
    print(f"wrote {OUT / 'comparison.md'}")
    f = RES.parent / "DIFFERENCES.md"
    if f.exists():
        s = f.read_text()
        blocks = {"summary": summary(), "results": results(8) + results(30) + OLD_NOTE, "gaps": gaps(), "periods": periods(),
                  "decomposition": decomposition(), "universe_effect": universe_effect()}
        for name in ("universe_rules", "market_row", "delisting_2021"):
            g = OUT / "checks" / f"{name}.md"
            if g.exists():
                blocks[name] = g.read_text().split("\n", 2)[2].splitlines()      # without its own heading
        for name, lines in blocks.items():
            a, b = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
            if a in s and b in s:
                s = s[:s.index(a) + len(a)] + "\n" + "\n".join(lines).rstrip() + "\n" + s[s.index(b):]
        f.write_text(s)
        print(f"updated the generated tables of {f.name}")


if __name__ == "__main__":
    report()
