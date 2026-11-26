"""Rolling-window estimation and out-of-sample evaluation (Section 4.2 of the paper).

For each test year y the model is trained on the eight calendar years before y and then
applied, unchanged, to every trading day of y. The out-of-sample weights of all years are
joined; returns, trading costs and performance numbers are computed from the joined series.

Three model classes:
  attention   the Attention Factor model, factors and trading policy trained in one step
  pca         PCA with the long convolution model (two steps)
  ou          PCA with an OU threshold rule (two steps, nothing trained)
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .chars import ALL, DAILY, GROUPS, MONTHLY
from .features import KMAX, N_SLOTS, pca_files
from .model import (COST_SHORT, COST_TRADE, AttentionFactorModel, LongConvPolicy, explained_variance, objective, sharpe,
                    trading_cost)
from .panel import PANEL, ROOT, Panel

RUNS = ROOT / "runs"
FIRM = MONTHLY + DAILY          # order of the 39 firm inputs inside the model


@dataclass
class Config:
    kind: str = "attention"          # attention | pca | ou
    k: int = 30
    seed: int = 0
    first_year: int = 1998
    last_year: int = 2021
    train_years: int = 8
    s: int = 30                      # lookback (paper)
    d: int = 32                      # attention dimension (paper)
    d_h: int = 32                    # hidden dimension of the sequence model (paper)
    dropout: float = 0.1             # paper
    squash: float = 0.001            # paper
    lambda_var: float = 100.0        # paper
    lr: float = 0.003                # paper
    weight_decay: float = 0.05       # paper, on the sequence model; choice: decoupled (AdamW), the reading of its citation
    epochs: int = 30                 # paper
    ridge: float = 1e-5              # choice: the paper does not report it; selected on 1996-1997
    batch: int = 64                  # choice: consecutive decision dates per gradient step; selected on 1996-1997
    # choice: scale of the Sharpe ratio inside the objective. The paper gives lambda_Var = 100 and
    # not the unit of its Sharpe ratio. sqrt(252) = annualized; selected on 1996-1997 against 1 (per day).
    sr_scale: float = 15.874507866387544
    # choice: lower bound of one security's explained-variance term in the objective; None = no bound,
    # the formula as printed (scripts/replicate_as_cited.py). The diagnostics always use -1.
    ev_floor: float | None = -1.0
    use_common: bool = True          # the 40 inputs that are equal for all securities
    drop_groups: list[str] = field(default_factory=list)   # characteristic groups left out (Table 3)
    ou_threshold: float = 1.25       # choice, see README
    ou_r2: float = 0.25              # choice, see README
    # variants of open choices (README, "Variants of open choices"); the defaults are those of every reported run
    pca: str = "cov"                 # cov | corr: PCA of the covariance or of the correlation matrix (pca, ou)
    lookback: str = "own"            # own | current: residual history with each day's own factor model, or today's
    name: str = ""

    def __post_init__(self) -> None:
        if self.pca not in ("cov", "corr") or self.lookback not in ("own", "current"):
            raise ValueError(f"pca={self.pca!r}, lookback={self.lookback!r}")
        if self.kind == "attention" and self.pca != "cov":
            raise ValueError("--pca applies to the two PCA benchmarks, not to the Attention Factor model")

    def run_name(self) -> str:
        if self.name:
            return self.name
        base = f"{self.kind}_K{self.k}"
        if self.pca != "cov":
            base += f"_{self.pca}"
        if self.lookback != "own":
            base += f"_look-{self.lookback}"
        if self.drop_groups:
            base += "_no-" + "-".join(self.drop_groups)
        return base + (f"_seed{self.seed}" if self.kind != "ou" else "")


class Data:
    """All arrays the three model classes read, as CPU tensors indexed by (day, security)."""

    def __init__(self, drop_groups: list[str] | None = None):
        p = Panel()
        self.dates = p.dates
        self.years = self.dates.astype("datetime64[Y]").astype(int) + 1970
        # a security that has stopped trading (no price at the close of t) is no longer in the cross-section
        member = p.member & np.isfinite(p.daily("prc"))
        self.member = torch.from_numpy(member)
        ret = p.daily("ret")
        r_next = np.zeros_like(ret)
        r_next[:-1] = np.nan_to_num(ret[1:])
        self.r_next = torch.from_numpy(r_next)
        has = np.zeros(ret.shape, dtype=bool)
        has[:-1] = np.isfinite(ret[1:])
        self.has_ret_next = torch.from_numpy(has)
        rf_next = np.zeros(len(self.dates), dtype=np.float32)
        rf_next[:-1] = p.rf[1:]
        self.rf_next = torch.from_numpy(rf_next)
        fm = np.load(PANEL / "feature_meta.npz", allow_pickle=False)
        self.month_row = torch.from_numpy(fm["month_row"].astype(np.int64))
        drop = {c for g in (drop_groups or []) for c in GROUPS[g]}
        self.firm_names = [c for c in FIRM if c not in drop]
        keep_m = [i for i, c in enumerate(MONTHLY) if c not in drop]
        keep_d = [i for i, c in enumerate(DAILY) if c not in drop]
        self.rank_m = torch.from_numpy(np.load(PANEL / "rank_monthly.npy")[:, :, keep_m])
        self.rank_d = torch.from_numpy(np.load(PANEL / "rank_daily.npy")[:, :, keep_d])
        common = np.load(PANEL / "common_raw.npy")
        keep_c = [i for i, c in enumerate(ALL) if c not in drop] + [len(ALL)]
        self.common_raw = torch.from_numpy(common[:, keep_c])
        self.m_firm = len(self.firm_names)
        self.m_common = len(keep_c)
        # equally weighted return of the universe, the "market" of Table 2
        mem = (member & has).astype(np.float32)
        self.mkt_next = torch.from_numpy((r_next * mem).sum(1) / np.maximum(mem.sum(1), 1))

    # decision-date ranges ------------------------------------------------
    def last_day_of_year(self, y: int) -> int:
        return int(np.where(self.years == y)[0][-1])

    def test_range(self, y: int) -> tuple[int, int]:
        """Decision dates whose next-day return falls in year y: [lo, hi)."""
        return self.last_day_of_year(y - 1), self.last_day_of_year(y)

    def train_range(self, y: int, train_years: int) -> tuple[int, int]:
        """Decision dates whose next-day return falls in the train_years years before y."""
        return self.last_day_of_year(y - train_years - 1), self.last_day_of_year(y - 1)

    # inputs ---------------------------------------------------------------
    def x_firm(self, a: int, b: int, cols: torch.Tensor) -> torch.Tensor:
        rows = self.month_row[a:b]
        if (rows < 0).any():
            raise ValueError("no monthly characteristics before the first decision month")
        xm = self.rank_m[rows[:, None], cols[None, :]]
        xd = self.rank_d[a:b][:, cols]
        return torch.cat([xm, xd], -1)

    def common_stats(self, a: int, b: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.common_raw[a:b]
        mean = torch.nanmean(x, 0)
        std = torch.sqrt(torch.nanmean((x - mean) ** 2, 0)).clamp(min=1e-8)
        return mean, std

    def x_common(self, a: int, b: int, stats) -> torch.Tensor:
        mean, std = stats
        return torch.nan_to_num((self.common_raw[a:b] - mean) / std)


def batches(lo: int, hi: int, size: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Cut the decision dates lo .. hi-1 into consecutive batches of about ``size`` dates.

    Choice: the first cut is at a random offset, every date is in exactly one batch per epoch,
    a piece shorter than half a batch is merged with its neighbor, and the order is random.
    """
    cuts = np.unique(np.r_[lo, np.arange(lo + int(rng.integers(0, size)), hi, size), hi])
    if len(cuts) > 2 and cuts[1] - cuts[0] < size // 2:
        cuts = np.delete(cuts, 1)
    if len(cuts) > 2 and cuts[-1] - cuts[-2] < size // 2:
        cuts = np.delete(cuts, -2)
    out = list(zip(cuts[:-1].tolist(), cuts[1:].tolist()))
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------- attention model

def att_forward(model: AttentionFactorModel, data: Data, j0: int, j1: int, stats, use_common: bool):
    """Model output for decision dates j0 .. j1-1 (the s dates before j0 feed the lookback)."""
    a = j0 - model.s
    cols = data.member[a:j1].any(0).nonzero().squeeze(1)
    xc = data.x_common(a, j1, stats) if use_common else None
    out = model(data.x_firm(a, j1, cols), xc, data.member[a:j1][:, cols], data.r_next[a:j1][:, cols],
                data.has_ret_next[a:j1][:, cols])
    return out, cols


def train_attention(data: Data, year: int, cfg: Config, log=None) -> tuple[AttentionFactorModel, tuple]:
    torch.manual_seed(cfg.seed * 100003 + year)
    rng = np.random.default_rng(cfg.seed * 100003 + year)
    lo, hi = data.train_range(year, cfg.train_years)
    lo = max(lo, int((data.month_row >= 0).nonzero()[0]) + cfg.s + 1)
    stats = data.common_stats(lo, hi)
    model = AttentionFactorModel(data.m_firm, data.m_common if cfg.use_common else 0, cfg.k, cfg.d, cfg.s, cfg.d_h,
                                 cfg.dropout, cfg.squash, cfg.ridge, cfg.lookback)
    opt = torch.optim.AdamW([
        {"params": model.factors.parameters(), "weight_decay": 0.0},
        {"params": model.policy.parameters(), "weight_decay": cfg.weight_decay},
    ], lr=cfg.lr)
    for epoch in range(cfg.epochs):
        model.train()
        srs, evs = [], []
        for j0, j1 in batches(lo, hi, cfg.batch, rng):
            out, _ = att_forward(model, data, j0, j1, stats, cfg.use_common)
            obj, parts = objective(out, data.rf_next[j0:j1], cfg.lambda_var, cfg.sr_scale, cfg.ev_floor)
            opt.zero_grad(set_to_none=True)
            (-obj).backward()
            if model.factors.m_common:
                # zero in exact arithmetic (the level inputs cancel in the softmax); keep rounding
                # noise from moving these rows of W
                model.factors.W.grad[model.factors.m_firm:] = 0.0
            opt.step()
            srs.append(float(parts["sr"]))
            evs.append(float(parts["ev"]))
        if log and (epoch % 10 == 9 or epoch == 0):
            log(f"  year {year} epoch {epoch + 1}: net SR per day {np.mean(srs) / cfg.sr_scale:.3f} "
                f"(annualized {np.mean(srs) / cfg.sr_scale * math.sqrt(252):.2f}), explained variance {np.mean(evs):.3f}")
    return model, stats


@torch.no_grad()
def apply_attention(model: AttentionFactorModel, data: Data, lo: int, hi: int, stats, cfg: Config) -> tuple[np.ndarray, dict]:
    """Asset weights [hi - lo, N] for decision dates lo .. hi-1, and diagnostics of the factor layer."""
    model.eval()
    w_all = np.zeros((hi - lo, data.member.shape[1]), dtype=np.float32)
    eff, top10, evs = [], [], []
    for j0 in range(lo, hi, 64):
        j1 = min(j0 + 64, hi)
        out, cols = att_forward(model, data, j0, j1, stats, cfg.use_common)
        w_all[j0 - lo:j1 - lo, cols.numpy()] = out["w"].numpy()
        w_f, beta = out["w_f"][model.s:], out["beta"][model.s:]
        # trace of w^F beta^T: K without a ridge penalty, fewer when the penalty shrinks weak directions
        eff.append(torch.einsum("bkn,bkn->b", w_f, beta))
        top10.append(w_f.topk(10, dim=-1).values.sum(-1).mean(1))
        evs.append(float(explained_variance(out["eps"], out["r"], out["valid"])))
    diag = {"effective_factors": float(torch.cat(eff).mean()), "top10_weight": float(torch.cat(top10).mean()),
            "explained_variance": float(np.mean(evs))}
    return w_all, diag


# ---------------------------------------------------------------- PCA residuals

class PcaData:
    """The PCA factor model of each date as a pair of [500, k] matrices per slot: factor weights A
    (w^F = A^T) and loadings B (beta = B^T). Residual: R - B A^T R. Asset weights: a - A B^T a.
    Covariance PCA (mode "cov"): A = B = V. Correlation PCA (mode "corr"): A = V / sigma, B = sigma V
    (features.build_pca). ``lookback`` selects how the residual history of the signal is built."""

    mode, lookback_mode, sd = "cov", "own", None      # the defaults of every reported run

    def __init__(self, data: Data, mode: str = "cov", lookback: str = "own"):
        self.data, self.mode, self.lookback_mode = data, mode, lookback
        v_name, sd_name = pca_files(mode)
        self.sources = [v_name, "pca_cols.npy", "daily_ret.npy"] + ([sd_name] if sd_name else [])
        for n in self.sources:
            if not (PANEL / n).exists():
                raise FileNotFoundError(f"{PANEL / n} is missing; build it with scripts/build.py "
                                        f"{'pca' if mode == 'cov' else 'pca_' + mode}")
        if (PANEL / v_name).stat().st_mtime < (PANEL / "daily_ret.npy").stat().st_mtime:
            raise RuntimeError(f"{v_name} predates the panel; rebuild it with scripts/build.py "
                               f"{'pca' if mode == 'cov' else 'pca_' + mode}")
        self.v = np.load(PANEL / v_name, mmap_mode="r")
        self.sd = np.load(PANEL / sd_name) if sd_name else None
        self.cols = np.load(PANEL / "pca_cols.npy")
        self._eps: dict[int, torch.Tensor] = {}

    def hedge(self, j0: int, j1: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A, B [j1 - j0, 500, k] of the decision dates j0 .. j1-1; zero rows in empty slots."""
        v = torch.from_numpy(np.asarray(self.v[j0:j1, :, :k]))
        if self.sd is None:
            return v, v
        sd = torch.from_numpy(self.sd[j0:j1]).clamp(min=1e-8)[..., None]   # empty slots: v is zero there
        return v / sd, v * sd

    def eps(self, k: int) -> torch.Tensor:
        """Residual of day t+1, dated t: R_{t+1} - B A^T R_{t+1} with the first k eigenvectors; NaN outside the PCA."""
        if k not in self._eps:
            f = PANEL / (f"pca_eps_K{k}.npy" if self.mode == "cov" else f"pca_eps_{self.mode}_K{k}.npy")
            newest_source = max((PANEL / n).stat().st_mtime for n in self.sources)
            if f.exists() and f.stat().st_mtime >= newest_source:
                e = np.load(f)
            else:
                r_next = self.data.r_next.numpy()
                e = np.full_like(r_next, np.nan)
                for t in range(r_next.shape[0]):
                    c = self.cols[t]
                    n = int((c >= 0).sum())
                    if n == 0:
                        continue
                    c = c[:n]
                    v = np.asarray(self.v[t, :n, :k], dtype=np.float64)
                    if self.sd is None:
                        a = b = v
                    else:
                        sd = np.maximum(self.sd[t, :n].astype(np.float64), 1e-8)[:, None]
                        a, b = v / sd, v * sd
                    r = r_next[t, c].astype(np.float64)
                    e[t, c] = (r - b @ (a.T @ r)).astype(np.float32)
                tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp.npy")     # another process may want the same file
                np.save(tmp, e)
                os.replace(tmp, f)
            self._eps[k] = torch.from_numpy(e)
        return self._eps[k]

    def batch_cols(self, j0: int, j1: int) -> torch.Tensor:
        """Columns of the securities that are in the PCA on any of the decision dates j0 .. j1-1."""
        c = self.cols[j0:j1]
        return torch.from_numpy(np.unique(c[c >= 0]).astype(np.int64))

    def slots(self, j0: int, j1: int, sub: torch.Tensor, n_all: int) -> tuple[torch.Tensor, torch.Tensor]:
        """For each PCA slot of each date: its position within ``sub`` and whether the slot is filled."""
        c = torch.from_numpy(self.cols[j0:j1].astype(np.int64))                # [B, 500]
        ok = c >= 0                                                            # empty slots carry index -1
        pos = torch.zeros(n_all, dtype=torch.int64)
        pos[sub] = torch.arange(len(sub))
        return pos[c.clamp(min=0)], ok

    def weights(self, alloc: torch.Tensor, j0: int, j1: int, k: int, cs: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
        """w = (w^eps)^T alloc = alloc - A B^T alloc on the securities of each day's PCA, scaled to an L1 norm of one.

        alloc and the result are [B, len(sub)], in the columns of ``batch_cols``."""
        wa, lb = self.hedge(j0, j1, k)                                         # [B, 500, k], zero rows where empty
        a = torch.gather(alloc, 1, cs) * ok
        ws = a - torch.einsum("bnk,bk->bn", wa, torch.einsum("bnk,bn->bk", lb, a))
        w = torch.zeros_like(alloc).scatter_add(1, cs, ws * ok)                # empty slots add zero
        return w / (w.abs().sum(1, keepdim=True) + 1e-12)

    def lookback(self, k: int, j0: int, j1: int, s: int, sub: torch.Tensor, cs: torch.Tensor, ok: torch.Tensor):
        """Residual windows [B, len(sub), s] for decision dates j0 .. j1-1 and the tradable mask.

        The tradable mask is that of the default in both modes (in the PCA on the decision date and
        on the s dates before it), so that the mode changes the signal and not the securities."""
        e = self.eps(k)[j0 - s:j1][:, sub]
        look = e.unfold(0, s, 1)[:-1]
        in_pca = torch.zeros(j1 - j0, len(sub), dtype=torch.int32).scatter_add_(1, cs, ok.to(torch.int32)) > 0
        tradable = in_pca & torch.isfinite(look).all(-1)
        if self.lookback_mode == "own":
            return torch.nan_to_num(look), tradable
        # "current": the residual portfolio of the decision date j applied to the returns of days
        # j-s+1 .. j (r_next dated j-s .. j-1). A security in the PCA on date j has a return on each
        # of the 252 days up to j, so the window has no missing values.
        rw = self.data.r_next[j0 - s:j1].unfold(0, s, 1)[:-1]                 # [B, N_all, s]
        c = torch.from_numpy(self.cols[j0:j1].astype(np.int64)).clamp(min=0)   # [B, 500]
        rs = torch.gather(rw, 1, c[..., None].expand(-1, -1, s)) * ok[..., None]
        wa, lb = self.hedge(j0, j1, k)
        es = rs - torch.einsum("bnk,bks->bns", lb, torch.einsum("bnk,bns->bks", wa, rs))
        cur = torch.zeros(j1 - j0, len(sub), s, dtype=es.dtype).scatter_add_(1, cs[..., None].expand(-1, -1, s),
                                                                             es * ok[..., None])
        return cur, tradable


def pca_forward(policy: LongConvPolicy, pd_: PcaData, k: int, j0: int, j1: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Asset weights [B, len(sub)] of the PCA benchmark and the security columns ``sub`` they refer to."""
    sub = pd_.batch_cols(j0, j1)
    cs, ok = pd_.slots(j0, j1, sub, pd_.data.member.shape[1])
    look, tradable = pd_.lookback(k, j0, j1, s, sub, cs, ok)
    alloc = policy(look) * tradable
    return pd_.weights(alloc, j0, j1, k, cs, ok), sub


def train_pca(data: Data, pd_: PcaData, year: int, cfg: Config, log=None) -> LongConvPolicy:
    torch.manual_seed(cfg.seed * 100003 + year)
    rng = np.random.default_rng(cfg.seed * 100003 + year)
    lo, hi = data.train_range(year, cfg.train_years)
    first = int(np.where((pd_.cols >= 0).any(1))[0][0]) + cfg.s + 1
    lo = max(lo, first)
    policy = LongConvPolicy(cfg.s, cfg.d_h, cfg.dropout, cfg.squash)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for epoch in range(cfg.epochs):
        policy.train()
        srs = []
        for j0, j1 in batches(lo, hi, cfg.batch, rng):
            w, sub = pca_forward(policy, pd_, cfg.k, j0, j1, cfg.s)
            gross = (w * data.r_next[j0:j1][:, sub]).sum(1)
            net = gross[1:] - trading_cost(w)
            sr = sharpe(net, data.rf_next[j0 + 1:j1])
            opt.zero_grad(set_to_none=True)
            (-sr).backward()
            opt.step()
            srs.append(float(sr))
        if log and (epoch % 10 == 9 or epoch == 0):
            log(f"  year {year} epoch {epoch + 1}: net SR per day {np.mean(srs):.3f} (annualized {np.mean(srs) * math.sqrt(252):.2f})")
    return policy


@torch.no_grad()
def apply_pca(policy: LongConvPolicy, pd_: PcaData, lo: int, hi: int, cfg: Config) -> np.ndarray:
    policy.eval()
    w_all = np.zeros((hi - lo, pd_.data.member.shape[1]), dtype=np.float32)
    for j0 in range(lo, hi, 64):
        j1 = min(j0 + 64, hi)
        w, sub = pca_forward(policy, pd_, cfg.k, j0, j1, cfg.s)
        w_all[j0 - lo:j1 - lo, sub.numpy()] = w.numpy()
    return w_all


# ---------------------------------------------------------------- OU threshold rule

@torch.no_grad()
def apply_ou(pd_: PcaData, lo: int, hi: int, cfg: Config) -> np.ndarray:
    """Avellaneda and Lee (2010) signal with the threshold rule in the form used by
    Guijarro-Ordonez, Pelger and Zanotti: an AR(1) model is fitted to the cumulated residuals of
    the lookback; the allocation is -1 above the threshold, +1 below minus the threshold, and 0
    otherwise or when the fit is poor (choice of constants: see Config)."""
    w_all = np.zeros((hi - lo, pd_.data.member.shape[1]), dtype=np.float32)
    for j0 in range(lo, hi, 64):
        j1 = min(j0 + 64, hi)
        sub = pd_.batch_cols(j0, j1)
        cs, ok_slot = pd_.slots(j0, j1, sub, pd_.data.member.shape[1])
        look, tradable = pd_.lookback(cfg.k, j0, j1, cfg.s, sub, cs, ok_slot)
        x = look.double().cumsum(-1)                                 # price-like path of the residual
        x0, x1 = x[..., :-1], x[..., 1:]
        m0, m1 = x0.mean(-1, keepdim=True), x1.mean(-1, keepdim=True)
        var0 = ((x0 - m0) ** 2).mean(-1)
        cov = ((x0 - m0) * (x1 - m1)).mean(-1)
        b = cov / var0.clamp(min=1e-18)
        a = m1.squeeze(-1) - b * m0.squeeze(-1)
        resid = x1 - (a[..., None] + b[..., None] * x0)
        var_e = (resid ** 2).mean(-1)
        var1 = ((x1 - m1) ** 2).mean(-1)
        r2 = 1 - var_e / var1.clamp(min=1e-18)
        ok = tradable & (b > 0) & (b < 1) & (r2 > cfg.ou_r2)
        mu = a / (1 - b).clamp(min=1e-9)
        sigma_eq = torch.sqrt(var_e / (1 - b ** 2).clamp(min=1e-9))
        score = (x[..., -1] - mu) / sigma_eq.clamp(min=1e-12)
        alloc = torch.zeros_like(score)
        alloc[ok & (score > cfg.ou_threshold)] = -1.0
        alloc[ok & (score < -cfg.ou_threshold)] = 1.0
        w_all[j0 - lo:j1 - lo, sub.numpy()] = pd_.weights(alloc.float(), j0, j1, cfg.k, cs, ok_slot).numpy()
    return w_all


# ---------------------------------------------------------------- evaluation

def performance(w: np.ndarray, data: Data, lo: int, hi: int) -> tuple[dict, dict]:
    """Daily series and annualized numbers for weights w [hi - lo, N] dated lo .. hi-1."""
    r = data.r_next[lo:hi].numpy()
    gross = (w * r).sum(1)
    prev = np.vstack([np.zeros((1, w.shape[1]), dtype=w.dtype), w[:-1]])
    turnover = np.abs(w - prev).sum(1)
    short = np.clip(-w, 0, None).sum(1)
    cost = COST_TRADE * turnover + COST_SHORT * short
    net = gross - cost
    mkt = data.mkt_next[lo:hi].numpy()

    def stats(x: np.ndarray) -> tuple[float, float, float]:
        mu, sd = x.mean() * 252, x.std(ddof=1) * math.sqrt(252)
        return (float(mu / sd) if sd > 0 else float("nan")), float(100 * mu), float(100 * sd)

    sr, mu, sd = stats(gross)
    srn, mun, sdn = stats(net)
    beta = float(np.cov(gross, mkt)[0, 1] / np.var(mkt, ddof=1))
    corr = float(np.corrcoef(gross, mkt)[0, 1])
    m = {"SR": sr, "mu": mu, "sigma": sd, "SR_net": srn, "mu_net": mun, "sigma_net": sdn, "beta": beta,
         "corr_market": corr, "turnover_per_day": float(turnover[1:].mean()), "short_exposure": float(short.mean()),
         "cost_drag": float(mu - mun), "days": int(len(gross))}
    series = {"date": data.dates[lo + 1:hi + 1].astype(str), "gross": gross, "net": net, "turnover": turnover,
              "short": short, "market": mkt}
    return m, series


def saved_config(cfg: Config) -> dict:
    """The configuration as written to config.json and metrics.json. The variant options are left
    out at their defaults, so that runs written before they existed keep the same configuration."""
    c = asdict(cfg)
    for opt, default in (("pca", "cov"), ("lookback", "own")):
        if c[opt] == default:
            del c[opt]
    return c


def run(cfg: Config, data: Data | None = None, pd_: PcaData | None = None) -> dict:
    """Train (if the class has parameters) and evaluate one configuration over all test years."""
    torch.set_num_threads(int(torch.get_num_threads()))
    data = data or Data(cfg.drop_groups)
    out_dir = RUNS / cfg.run_name()
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "train.log", "a")

    def log(msg: str) -> None:
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if cfg.kind in ("pca", "ou"):
        pd_ = pd_ or PcaData(data, cfg.pca, cfg.lookback)
        # residuals passed in by a caller (pca_regression.PcaRegressionData) have their own construction:
        # refuse a configuration whose variant options they would ignore without an error
        if (pd_.mode, pd_.lookback_mode) != (cfg.pca, cfg.lookback):
            raise ValueError(f"the PCA residuals passed in ({type(pd_).__name__}: {pd_.mode}, {pd_.lookback_mode}) "
                             f"do not match pca={cfg.pca!r}, lookback={cfg.lookback!r}")
    # a folder holds the results of one configuration on one panel; refuse to mix
    key = {k: v for k, v in saved_config(cfg).items() if k not in ("first_year", "last_year", "name")}
    key["panel"] = int((PANEL / "daily_ret.npy").stat().st_mtime)
    key["panel_dir"] = PANEL.name if PANEL.parent == ROOT / "data" else str(PANEL.relative_to(ROOT))
    # the class of the PCA residuals and, if it is not the default of that class, the set of files it reads
    key["pca"] = None if pd_ is None else type(pd_).__name__ + ("" if getattr(pd_, "pre", "pcareg") == "pcareg" else f":{pd_.pre}")
    if pd_ is not None and pd_.mode != "cov":
        key["pca"] = pd_.mode            # the correlation PCA of PcaData: "corr", as the folders of those runs hold it
    # a folder written before a setting existed holds the default of that setting
    older = {**asdict(Config()), "panel_dir": "panel", "pca": "PcaData" if cfg.kind in ("pca", "ou") else None}
    cfile = out_dir / "config.json"
    if cfile.exists():
        old = json.loads(cfile.read_text())
        diff = sorted(k for k in set(old) | set(key) if old.get(k, older.get(k)) != key.get(k))
        if diff:
            raise RuntimeError(f"{out_dir} holds results of another configuration or panel (differs in {diff}); "
                               "use --name or delete the folder")
    elif any(out_dir.glob("weights_*.npy")):
        raise RuntimeError(f"{out_dir} holds weights without a config.json; delete the folder")
    else:
        cfile.write_text(json.dumps(key, indent=1))
    log(f"run {cfg.run_name()}: {json.dumps(asdict(cfg))}")
    ws, lo0, hi1, diags = [], None, None, {}
    dfile = out_dir / "diagnostics.json"
    if dfile.exists():
        diags = json.loads(dfile.read_text())
    for y in range(cfg.first_year, cfg.last_year + 1):
        lo, hi = data.test_range(y)
        f = out_dir / f"weights_{y}.npy"
        if f.exists():
            w = np.load(f)
        else:
            t0 = time.time()
            if cfg.kind == "attention":
                model, stats = train_attention(data, y, cfg, log)
                w, diags[str(y)] = apply_attention(model, data, lo, hi, stats, cfg)
                # several processes may train different years of one run (--first-year/--last-year):
                # merge with what the others wrote, and replace the file in one step
                if dfile.exists():
                    diags = {**json.loads(dfile.read_text()), **diags}
                tmp = dfile.with_name(f"diagnostics.{os.getpid()}.tmp")
                tmp.write_text(json.dumps(diags, indent=1))
                os.replace(tmp, dfile)
                torch.save(model.state_dict(), out_dir / f"model_{y}.pt")
            elif cfg.kind == "pca":
                policy = train_pca(data, pd_, y, cfg, log)
                w = apply_pca(policy, pd_, lo, hi, cfg)
            elif cfg.kind == "ou":
                w = apply_ou(pd_, lo, hi, cfg)
            else:
                raise ValueError(cfg.kind)
            np.save(f, w)
            m, _ = performance(w, data, lo, hi)
            log(f"year {y}: SR {m['SR']:.2f}, net SR {m['SR_net']:.2f}, mu {m['mu']:.1f}, turnover {m['turnover_per_day']:.2f} "
                f"({time.time() - t0:.0f}s)")
        ws.append(w)
        lo0 = lo if lo0 is None else lo0
        hi1 = hi
    w_all = np.vstack(ws)
    m, series = performance(w_all, data, lo0, hi1)
    if diags:
        for name in ("effective_factors", "top10_weight", "explained_variance"):
            m[name] = float(np.mean([d[name] for d in diags.values()]))
    m["config"] = saved_config(cfg)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=1))
    import pandas as pd
    pd.DataFrame(series).to_csv(out_dir / "daily.csv", index=False)
    log(f"TOTAL {cfg.first_year}-{cfg.last_year}: SR {m['SR']:.2f} mu {m['mu']:.2f} sigma {m['sigma']:.2f} | "
        f"net SR {m['SR_net']:.2f} mu {m['mu_net']:.2f} | beta {m['beta']:.2f} | turnover/day {m['turnover_per_day']:.2f}")
    logf.close()
    return m
