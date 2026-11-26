"""Model inputs from the raw characteristics, and the rolling PCA of the two-step benchmarks.

Rank normalization (Section 4.1 of the paper): on each decision date every characteristic is
replaced by its rank among the securities of the universe, scaled to (-0.5, 0.5). Choices: tied
values share their average rank; a missing value is first replaced by the last observed value
of the security (at most 12 months back for a monthly characteristic, 5 days for a daily one)
and otherwise set to the cross-sectional median, which is 0 after ranking; a characteristic
that fewer than half of the universe has in a month is set to 0 for all (this only concerns
the bid-ask spread before 1993, when CRSP has no closing quotes for NYSE and AMEX stocks).

The "level" inputs (Section 4.1): the cross-sectional median of each raw characteristic and
the risk-free rate, 40 numbers per date that are equal for all securities.

Outputs under data/panel/:
  rank_monthly.npy  float32 [Mc, N, 35]   rank_daily.npy  float32 [T, N, 4]
  common_raw.npy    float32 [T, 40]       (medians in the order of chars.ALL, then rf)
  feature_meta.npz  names (order of the 39 firm inputs), month_row [T] (row of rank_monthly per day, -1 if none)
  pca_v.npy         float32 [T, 500, KMAX]   pca_cols.npy int32 [T, 500]  (security column per slot, -1 = empty)
"""
from __future__ import annotations

import numpy as np

from .chars import ALL, DAILY, MONTHLY
from .panel import PANEL, Panel

KMAX = 100
PCA_WINDOW = 252
N_SLOTS = 500


def _ffill(a: np.ndarray, limit: int) -> np.ndarray:
    """Forward fill along axis 0, at most ``limit`` steps."""
    out = a.copy()
    for _ in range(limit):
        prev = np.roll(out, 1, axis=0)
        prev[0] = np.nan
        hole = np.isnan(out)
        if not hole.any():
            break
        out[hole] = prev[hole]
    return out


def rank_row(v: np.ndarray) -> np.ndarray:
    """Ranks of the values v in (-0.5, 0.5); tied values share their average rank."""
    n = len(v)
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0            # average 1-based rank of each distinct value
    return (avg[inv] / (n + 1) - 0.5).astype(np.float32)


def _rank_rows(x: np.ndarray, member: np.ndarray, min_share: float = 0.0) -> np.ndarray:
    """x [R, N], member [R, N] -> ranks among member cells with a value; 0 elsewhere."""
    out = np.zeros_like(x, dtype=np.float32)
    for r in range(x.shape[0]):
        ok = member[r] & np.isfinite(x[r])
        n = int(ok.sum())
        if n < 2 or n < min_share * max(int(member[r].sum()), 1):
            continue
        out[r, ok] = rank_row(x[r, ok])
    return out


def build_inputs() -> None:
    panel = Panel()
    meta = np.load(PANEL / "chars_meta.npz", allow_pickle=False)
    dec_months = list(meta["dec_months"])
    t_n, n = panel.member.shape
    # the decision month of date t is the month of the return it earns (see panel.py)
    month_row = np.array([dec_months.index(m) if m in dec_months else -1 for m in panel.month_of_ret], dtype=np.int32)

    # ---- monthly: membership is constant within a decision month
    xm = _ffill(np.load(PANEL / "chars_monthly.npy"), 12)
    first_day_of = {r: int(np.where(month_row == r)[0][0]) for r in np.unique(month_row) if r >= 0}
    member_m = np.zeros((len(dec_months), n), dtype=bool)
    for r, d in first_day_of.items():
        member_m[r] = panel.member[d]
    rank_m = np.stack([_rank_rows(xm[:, :, c], member_m, min_share=0.5) for c in range(xm.shape[2])], -1)
    med_m = np.full((len(dec_months), xm.shape[2]), np.nan, dtype=np.float32)
    for r in range(len(dec_months)):
        if member_m[r].any():
            with np.errstate(all="ignore"):
                med_m[r] = np.nanmedian(xm[r, member_m[r]], axis=0)
    np.save(PANEL / "rank_monthly.npy", rank_m)

    # ---- daily
    xd = _ffill(np.load(PANEL / "chars_daily.npy"), 5)
    rank_d = np.stack([_rank_rows(xd[:, :, c], panel.member) for c in range(xd.shape[2])], -1)
    med_d = np.full((t_n, xd.shape[2]), np.nan, dtype=np.float32)
    for t in range(t_n):
        if panel.member[t].any():
            with np.errstate(all="ignore"):
                med_d[t] = np.nanmedian(xd[t, panel.member[t]], axis=0)
    np.save(PANEL / "rank_daily.npy", rank_d)

    # ---- the 40 inputs that are equal for all securities, in the order of ALL, then rf
    common = np.full((t_n, len(ALL) + 1), np.nan, dtype=np.float32)
    for j, c in enumerate(ALL):
        if c in MONTHLY:
            col = med_m[:, MONTHLY.index(c)]
            common[:, j] = np.where(month_row >= 0, col[np.maximum(month_row, 0)], np.nan)
        else:
            common[:, j] = med_d[:, DAILY.index(c)]
    common[:, -1] = panel.rf
    np.save(PANEL / "common_raw.npy", common)
    np.savez(PANEL / "feature_meta.npz", names=np.array(ALL), month_row=month_row)
    print("inputs: rank_monthly", rank_m.shape, "rank_daily", rank_d.shape, "common", common.shape)


def pca_files(mode: str) -> tuple[str, str | None]:
    """File names of the stored eigenvectors and, for the correlation PCA, the volatilities."""
    if mode == "cov":
        return "pca_v.npy", None
    if mode == "corr":
        return "pca_v_corr.npy", "pca_sd_corr.npy"
    raise ValueError(mode)


def pca_window(x: np.ndarray, corr: bool, kmax: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Eigenvectors [n, kmax] of the covariance matrix of the returns x [L, n] (largest eigenvalue
    first), or of their correlation matrix when ``corr``; then also the volatilities [n]."""
    x = x.astype(np.float64)
    x = x - x.mean(0)
    sd = None
    if corr:
        sd = np.maximum(x.std(0), 1e-8)
        x = x / sd                               # the correlation matrix is that of the standardized returns
    # eigenvectors of x^T x through the L x L matrix x x^T (same subspace, cheaper when L < n)
    lam, u = np.linalg.eigh(x @ x.T)
    top = np.argsort(lam)[::-1][:kmax]
    return (x.T @ u[:, top]) / np.sqrt(np.maximum(lam[top], 1e-300)), sd


def build_pca(mode: str = "cov") -> None:
    """Rolling PCA of the last 252 daily returns (Section 4.2 of the paper).

    Choices. (1) mode "cov" (the default, used by every reported run): the PCA is of the
    covariance matrix, as Section 3.1 of the paper describes PCA factors. The eigenvectors V
    serve as factor weights and as loadings, so the residual of the next day is (I - V V^T) R;
    no separate regression estimates the loadings. (2) mode "corr", the construction of
    Avellaneda and Lee (2010): the PCA is of the correlation matrix; the factor weights are the
    eigenportfolios V / sigma, and the loadings are those of an OLS regression of each return on
    the factor returns over the same 252 days. On that window the OLS loadings are exactly
    sigma * V (the factor returns are uncorrelated there, with variances equal to the
    eigenvalues), so the residual of the next day is R - (sigma V)(V / sigma)^T R. The
    volatilities sigma are stored beside V. (3) On decision date t the PCA uses the returns of
    days t-251 .. t of the securities that are in the universe on t and have a return on each of
    those days. The eigenvectors of the KMAX largest eigenvalues are stored; a model with K
    factors uses the first K. pca_cols.npy (which securities, in which slots) is the same for
    both modes.
    """
    panel = Panel()
    v_name, sd_name = pca_files(mode)
    # residuals of an older PCA. pcareg*: the files of pca_regression.py, which refer to the slots of
    # pca_cols.npy; only the default mode writes that file again
    patterns = ("pca_eps_K*.npy", "pcareg*.npy") if mode == "cov" else (f"pca_eps_{mode}_K*.npy",)
    for pattern in patterns:
        for f in PANEL.glob(pattern):
            f.unlink()
    ret = panel.daily("ret")
    t_n, n = ret.shape
    v_all = np.lib.format.open_memmap(PANEL / v_name, mode="w+", dtype=np.float32, shape=(t_n, N_SLOTS, KMAX))
    sd_all = np.zeros((t_n, N_SLOTS), dtype=np.float32) if sd_name else None
    cols_all = np.full((t_n, N_SLOTS), -1, dtype=np.int32)
    finite = np.isfinite(ret)
    run = np.zeros(n, dtype=np.int32)       # length of the current run of days with a return
    for t in range(t_n):
        run = np.where(finite[t], run + 1, 0)
        if t < PCA_WINDOW - 1 or not panel.member[t].any():
            continue
        cols = np.where(panel.member[t] & (run >= PCA_WINDOW))[0]
        if len(cols) <= KMAX:
            continue
        v, sd = pca_window(ret[t - PCA_WINDOW + 1:t + 1][:, cols], mode == "corr", KMAX)
        v_all[t, :len(cols)] = v.astype(np.float32)
        if sd_all is not None:
            sd_all[t, :len(cols)] = sd.astype(np.float32)
        cols_all[t, :len(cols)] = cols
        if t % 2000 == 0:
            print(f"pca ({mode}): day", t, "of", t_n, "securities", len(cols), flush=True)
    v_all.flush()
    if sd_all is not None:
        np.save(PANEL / sd_name, sd_all)
    cols_file = PANEL / "pca_cols.npy"
    if mode == "cov" or not cols_file.exists():
        np.save(cols_file, cols_all)
    elif not np.array_equal(np.load(cols_file), cols_all):
        raise RuntimeError("pca_cols.npy differs from the securities of this PCA: rebuild with scripts/build.py pca")
    print(f"pca ({mode}): done")


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["inputs", "pca"]
    if "inputs" in what:
        build_inputs()
    if "pca" in what:
        build_pca()
    if "pca_corr" in what:
        build_pca("corr")
