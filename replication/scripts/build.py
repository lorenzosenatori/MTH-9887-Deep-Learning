"""Build the arrays the models read from the raw downloads, then print data checks.

    python scripts/build.py                 # panel, chars, inputs, pca, check
    python scripts/build.py chars inputs    # selected steps
    python scripts/build.py pca_corr        # correlation PCA for `run.py --pca corr` (not built by default)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attnfactors import chars, features, panel  # noqa: E402


def check() -> None:
    """Compare the equally weighted universe with the 'Market' row of Table 2 of the paper."""
    p = panel.Panel()
    ret = p.daily("ret")
    years = p.dates.astype("datetime64[Y]").astype(int) + 1970
    mem = p.member[:-1]
    r_next = np.nan_to_num(ret[1:])
    mkt = (r_next * mem).sum(1) / np.maximum(mem.sum(1), 1)
    oos = (years[1:] >= 1998) & (years[1:] <= 2021)
    x = mkt[oos]
    mu, sd = x.mean() * 252 * 100, x.std(ddof=1) * np.sqrt(252) * 100
    rf = p.rf[1:][oos].mean() * 252 * 100
    print(f"equally weighted universe, 1998-2021: mean {mu:.2f}%  volatility {sd:.2f}%  mean/vol {mu / sd:.2f}  "
          f"(excess of rf: {mu - rf:.2f}%, {(mu - rf) / sd:.2f});  paper, Table 2: 8.61%, 20.37%, 0.42")
    missing = (~np.isfinite(ret[1:]) & mem)[oos].mean()
    print(f"share of (day, member) cells without a return, 1998-2021: {missing:.5f}")
    print(f"days in the out-of-sample period: {int(oos.sum())}; members per day: {int(mem[oos].sum(1).min())} to {int(mem[oos].sum(1).max())}")
    xm = np.load(panel.PANEL / "chars_monthly.npy")
    meta = np.load(panel.PANEL / "chars_meta.npz", allow_pickle=False)
    fm = np.load(panel.PANEL / "feature_meta.npz", allow_pickle=False)
    rows = fm["month_row"]
    first = {int(r): int(np.where(rows == r)[0][0]) for r in np.unique(rows) if r >= 0}
    cov = np.zeros(xm.shape[2])
    n = 0
    for r, d in first.items():
        if meta["dec_months"][r] < "1990-01" or not p.member[d].any():      # a month without a universe has no members
            continue
        cov += np.isfinite(xm[r, p.member[d]]).mean(0)
        n += 1
    print("share of universe members with a value, per characteristic (decision months 1990-2021):")
    print("  " + ", ".join(f"{c} {v / n:.2f}" for c, v in zip(meta["monthly"], cov)))


STEPS = {"panel": panel.build, "chars": chars.build, "inputs": features.build_inputs, "pca": features.build_pca,
         "pca_corr": lambda: features.build_pca("corr"), "check": check}
DEFAULT = ["panel", "chars", "inputs", "pca", "check"]

if __name__ == "__main__":
    for name in (sys.argv[1:] or DEFAULT):
        t0 = time.time()
        print(f"== {name}", flush=True)
        STEPS[name]()
        print(f"   ({time.time() - t0:.0f}s)", flush=True)
