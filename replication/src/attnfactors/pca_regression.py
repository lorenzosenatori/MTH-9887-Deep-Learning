"""PCA residuals built the way the public code of the predecessor paper builds them.

The paper calls its PCA benchmark "in the spirit of" Guijarro-Ordonez, Pelger and Zanotti (Deep
Learning Statistical Arbitrage) and says the OU benchmark uses "the same implementation". That
paper and its public code (github.com/gregzanotti/dlsa-public, factor_models/pca.py) do three
things that ``features.build_pca`` does not:

1. The returns of the last 252 days are standardized: the PCA is of the correlation matrix.
2. The loadings of a stock are estimated by a regression without intercept of its returns of the
   last 60 days on the factor returns of those days.
3. The residual of the next day is the return minus loadings times the factor returns of that
   day, so the residual portfolios are not an orthogonal projection.

Returns are in excess of the risk-free rate, as in that code. For decision date t the windows
end on day t and the residual is that of day t+1 (the strictly out-of-sample variant of the
code, in this replication's dating). The securities are those of ``features.build_pca``: in the
universe on t with a return on each of the 252 days.

With a = V / vol (eigenvectors over the volatilities) and b the regression loadings, the residual
is eps = x - b (a^T x), so an allocation alloc over residuals is the asset portfolio
alloc - a (b^T alloc).

Files under the panel folder: pcareg_a.npy float32 [T, 500, max K], pcareg_beta_K{k}.npy
float32 [T, 500, k] and pcareg_eps_K{k}.npy float32 [T, N] (residual of day t+1, dated t).

A second set, with the prefix pcareg252, estimates the loadings on all 252 days instead of 60.
Then the loadings are close to the eigenvectors times the volatilities (not equal: nothing is
demeaned in the regression) and the residual is close to an orthogonal projection of the
standardized returns. It separates the two steps in the validation runs: the
change from covariance to correlation, and the change to loadings from 60 days. A regression on
60 days cannot be what the paper used for its row with 100 factors (100 regressors, 60 days).
"""
from __future__ import annotations

import numpy as np
import torch

from .features import N_SLOTS, PCA_WINDOW
from .panel import PANEL, Panel
from .train import Data, PcaData

LOADING_WINDOW = 60
KS = (8, 30)


def prefix(loading_window: int = LOADING_WINDOW) -> str:
    return "pcareg" if loading_window == LOADING_WINDOW else f"pcareg{loading_window}"


def residual_parts(x: np.ndarray, ks: tuple[int, ...] = KS, loading_window: int = LOADING_WINDOW
                   ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """x [252, n]: excess returns of the window. Returns a [n, max(ks)] and {k: b [n, k]}."""
    mean = x.mean(0)
    vol = np.sqrt(((x - mean) ** 2).mean(0))
    vol = np.maximum(vol, 1e-12)
    z = (x - mean) / vol
    kmax = max(ks)
    # eigenvectors of the correlation matrix z^T z through the 252 x 252 matrix z z^T (same subspace)
    lam, u = np.linalg.eigh(z @ z.T)
    top = np.argsort(lam)[::-1][:kmax]
    v = (z.T @ u[:, top]) / np.sqrt(np.maximum(lam[top], 1e-300))
    a = v / vol[:, None]
    x60 = x[-loading_window:]
    betas = {}
    for k in ks:
        f = x60 @ a[:, :k]                                           # factor returns of the last 60 days, not demeaned
        betas[k] = np.linalg.solve(f.T @ f + 1e-18 * np.eye(k), f.T @ x60).T   # regression without intercept
    return a, betas


def build(ks: tuple[int, ...] = KS, loading_window: int = LOADING_WINDOW) -> None:
    pre = prefix(loading_window)
    panel = Panel()
    ret = panel.daily("ret").astype(np.float64)
    rf = panel.rf.astype(np.float64)
    cols_all = np.load(PANEL / "pca_cols.npy")
    t_n, n_all = ret.shape
    a_all = np.lib.format.open_memmap(PANEL / f"{pre}_a.npy", mode="w+", dtype=np.float32, shape=(t_n, N_SLOTS, max(ks)))
    b_all = {k: np.lib.format.open_memmap(PANEL / f"{pre}_beta_K{k}.npy", mode="w+", dtype=np.float32, shape=(t_n, N_SLOTS, k))
             for k in ks}
    eps = {k: np.full((t_n, n_all), np.nan, dtype=np.float32) for k in ks}
    for t in range(PCA_WINDOW - 1, t_n - 1):
        c = cols_all[t]
        n = int((c >= 0).sum())
        if n == 0:
            continue
        c = c[:n]
        x = ret[t - PCA_WINDOW + 1:t + 1][:, c] - rf[t - PCA_WINDOW + 1:t + 1, None]
        if not np.isfinite(x).all():
            raise ValueError(f"missing return inside the PCA window of day {t}")
        a, betas = residual_parts(x, ks, loading_window)
        a_all[t, :n] = a
        nxt = ret[t + 1, c]
        xn = np.where(np.isfinite(nxt), nxt - rf[t + 1], 0.0)            # a missing return counts as zero, as elsewhere
        for k in ks:
            b_all[k][t, :n] = betas[k]
            eps[k][t, c] = xn - betas[k] @ (a[:, :k].T @ xn)
        if t % 2000 == 0:
            print(f"pca (loadings from {loading_window} days): day", t, "of", t_n, "securities", n, flush=True)
    a_all.flush()
    for k in ks:
        b_all[k].flush()
        np.save(PANEL / f"{pre}_eps_K{k}.npy", eps[k])
    print(f"pca (loadings from {loading_window} days): done")


class PcaRegressionData(PcaData):
    """Drop-in for ``PcaData`` in train_pca, apply_pca and apply_ou, with the residuals of this module."""

    def __init__(self, data: Data, loading_window: int = LOADING_WINDOW):
        self.data = data
        self.pre = prefix(loading_window)
        self.cols = np.load(PANEL / "pca_cols.npy")
        # the files of this module refer to the slots of pca_cols.npy: refuse a set that is older than its sources
        newest = max((PANEL / n).stat().st_mtime for n in ("pca_cols.npy", "daily_ret.npy", "ff.npy"))
        if (PANEL / f"{self.pre}_a.npy").stat().st_mtime < newest:
            raise RuntimeError(f"{self.pre}_*.npy is older than the panel or the PCA it was built from: build it again")
        self.a = np.load(PANEL / f"{self.pre}_a.npy", mmap_mode="r")
        self._b: dict[int, np.ndarray] = {}
        self._eps: dict[int, torch.Tensor] = {}

    def eps(self, k: int) -> torch.Tensor:
        if k not in self._eps:
            self._eps[k] = torch.from_numpy(np.load(PANEL / f"{self.pre}_eps_K{k}.npy"))
        return self._eps[k]

    def weights(self, alloc: torch.Tensor, j0: int, j1: int, k: int, cs: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
        """w = alloc - a (b^T alloc) on the securities of each day's PCA, scaled to an L1 norm of one."""
        if k not in self._b:
            self._b[k] = np.load(PANEL / f"{self.pre}_beta_K{k}.npy", mmap_mode="r")
        a = torch.from_numpy(np.asarray(self.a[j0:j1, :, :k]))
        b = torch.from_numpy(np.asarray(self._b[k][j0:j1]))
        x = torch.gather(alloc, 1, cs) * ok
        ws = x - torch.einsum("bnk,bk->bn", a, torch.einsum("bnk,bn->bk", b, x))
        w = torch.zeros_like(alloc).scatter_add(1, cs, ws * ok)
        return w / (w.abs().sum(1, keepdim=True) + 1e-12)


if __name__ == "__main__":
    build()
