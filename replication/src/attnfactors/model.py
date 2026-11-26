"""The Attention Factor model of Epstein, Wang, Choi and Pelger (2025), as its equations state it.

Shapes: B decision dates, N securities, K factors, M inputs, d attention dimension, s lookback.
A quantity "dated t" is known at the close of day t; ``r_next`` is the return of day t+1.

What the paper specifies is implemented as written. What it leaves open is a choice of this
replication; the README lists the choices ("Choices the paper leaves open"), and the ones made
in this file are marked "choice".
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

COST_TRADE = 0.0005   # 5 basis points per unit of L1 change of the weights (Section 3.3)
COST_SHORT = 0.0001   # 1 basis point per unit of short exposure held


class AttentionFactorLayer(nn.Module):
    """Factor weights  w^F = Softmax(Q X~^T / sqrt(d)),  X~ = X W  (Section 3.2).

    The softmax runs over the securities of one day. ``x_common`` holds the inputs that are
    equal for all securities (cross-sectional medians, risk-free rate); they enter through
    the same matrix W, as the paper's equation has them. Under that equation they cannot
    change the factor weights: a term that is equal for all securities cancels in a softmax
    over the securities (tests/test_model.py). The paper says they "keep the level
    information"; its equation does not show how.
    """

    def __init__(self, m_firm: int, m_common: int, k: int, d: int = 32):
        super().__init__()
        self.m_firm, self.m_common, self.k, self.d = m_firm, m_common, k, d
        # choice: Gaussian initialization scaled so that initial scores have a spread of about 0.3
        self.W = nn.Parameter(torch.randn(m_firm + m_common, d) / math.sqrt(m_firm))
        self.Q = nn.Parameter(torch.randn(k, d))

    def scores(self, x_firm: torch.Tensor, x_common: torch.Tensor | None) -> torch.Tensor:
        xt = x_firm @ self.W[: self.m_firm]                       # [B, N, d]
        if x_common is not None and self.m_common > 0:
            xt = xt + (x_common @ self.W[self.m_firm:])[:, None, :]
        return xt @ self.Q.T / math.sqrt(self.d)                  # [B, N, K]

    def forward(self, x_firm: torch.Tensor, x_common: torch.Tensor | None, member: torch.Tensor) -> torch.Tensor:
        sc = self.scores(x_firm, x_common).masked_fill(~member[..., None], float("-inf"))
        return torch.softmax(sc, dim=1).transpose(1, 2)           # [B, K, N], each row sums to one


def loadings(w_f: torch.Tensor, ridge: float) -> torch.Tensor:
    """beta [B, K, N] with  beta^T = (w^F)^T (w^F (w^F)^T + ridge I_K)^{-1}  (Section 3.2)."""
    wf = w_f.double()
    k = wf.shape[1]
    g = wf @ wf.transpose(1, 2) + ridge * torch.eye(k, dtype=wf.dtype, device=wf.device)
    return torch.linalg.solve(g, wf).to(w_f.dtype)                # G is symmetric: G^{-1} w^F = beta


def residuals(r_next: torch.Tensor, w_f: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """eps_{t+1} = R_{t+1} - beta_t^T F_{t+1},  F_{t+1} = w^F_t R_{t+1}  (Section 3.1)."""
    f = torch.einsum("bkn,bn->bk", w_f, r_next)
    return r_next - torch.einsum("bkn,bk->bn", beta, f)


def asset_weights(alloc: torch.Tensor, w_f: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """w = (w^eps)^T w^port = w^port - (w^F)^T (beta w^port), scaled to an L1 norm of one."""
    h = torch.einsum("bkn,bn->bk", beta, alloc)
    w = alloc - torch.einsum("bkn,bk->bn", w_f, h)
    return w / (w.abs().sum(1, keepdim=True) + 1e-12)


class LongConvPolicy(nn.Module):
    """Allocation of one residual portfolio from its last s residuals (Section 3.3, Appendix A).

    Specified by the paper: one long-convolution layer with hidden dimension 32, a kernel as
    long as the sequence, a skip term, the Squash operator on the kernel, the geometric-decay
    initialization, a dropout rate of 0.1.

    Choices: (1) the scalar residual series enters the 32 channels through an affine lift,
    one gain and one offset per channel; (2) the convolution is causal and the allocation is
    read at the last position, so only the output for the decision date is computed, which
    makes the FFT unnecessary; (3) a GELU and a linear read-out turn the 32 channels into one
    number; (4) residuals are multiplied by ``in_scale`` (percent units) before the lift;
    (5) dropout acts on the 32 channels after the GELU (the paper gives the rate, not the place).
    """

    def __init__(self, s: int = 30, d_h: int = 32, dropout: float = 0.1, squash: float = 0.001, in_scale: float = 100.0):
        super().__init__()
        self.s, self.d_h, self.squash, self.in_scale = s, d_h, squash, in_scale
        t = torch.arange(1, s + 1, dtype=torch.float32) / s
        h = torch.arange(1, d_h + 1, dtype=torch.float32)
        rate = (d_h / 2.0) ** (h / d_h)
        self.kernel = nn.Parameter(torch.randn(d_h, s) * torch.exp(-t[None, :] * rate[:, None]))  # Appendix A
        self.skip = nn.Parameter(torch.randn(d_h))
        self.gain = nn.Parameter(torch.randn(d_h) / math.sqrt(d_h) + 1.0)
        self.offset = nn.Parameter(torch.zeros(d_h))
        self.out = nn.Linear(d_h, 1)
        self.drop = nn.Dropout(dropout)

    def squashed_kernel(self) -> torch.Tensor:
        return torch.sign(self.kernel) * F.relu(self.kernel.abs() - self.squash)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u [..., s], most recent residual last. Returns the raw allocation [...]."""
        u = u * self.in_scale
        k = self.squashed_kernel()                                # [d_h, s], index = lag
        z = torch.einsum("...s,cs->...c", u.flip(-1), k)          # sum over lags of k[lag] * u[t - lag]
        last = u[..., -1:]
        y = self.gain * (z + self.skip * last) + self.offset * (k.sum(1) + self.skip)
        return self.out(self.drop(F.gelu(y))).squeeze(-1)


def trading_cost(w: torch.Tensor) -> torch.Tensor:
    """Cost of holding w[b], reached from w[b-1], for b = 1 .. B-1  (Section 3.3).

    Choice of timing: the return that a weight earns is charged with the trade that established
    the weight and with the short cost of that weight. The paper's printed formula,
    R_t^T w_{t-1} - cost(w_t, w_{t-1}), pairs the return with the next trade; the two differ by a
    shift of one day in the cost series.
    """
    turnover = (w[1:] - w[:-1]).abs().sum(1)
    short = torch.clamp(-w[1:], min=0).sum(1)
    return COST_TRADE * turnover + COST_SHORT * short


def explained_variance(eps: torch.Tensor, r: torch.Tensor, valid: torch.Tensor, floor: float | None = -1.0) -> torch.Tensor:
    """(1/N) sum_i (1 - Var(eps_i) / Var(R_i)), variances over the days of the batch.

    ``valid`` [B, N] marks the days on which security i is in the universe and has a return.
    Choices, both to keep a few securities from dominating the term: a security counts if it is
    valid on at least half of the days of the batch and its return varies; and the term of one
    security is bounded below at -1. Without the bound a stock whose price hardly moves in the
    batch (a takeover target) has a ratio in the hundreds and supplies most of the gradient.
    ``floor=None`` leaves the bound out, which is the formula as the paper prints it.
    """
    v = valid.to(eps.dtype)
    n = v.sum(0)
    keep = n >= 0.5 * valid.shape[0]
    n = n.clamp(min=1)

    def var(x: torch.Tensor) -> torch.Tensor:
        mean = (x * v).sum(0) / n
        return (((x - mean) ** 2) * v).sum(0) / n

    var_r = var(r)
    keep = keep & (var_r > 1e-10)             # a security whose price did not move has no variance to explain
    terms = 1.0 - var(eps)[keep] / var_r[keep]
    return (terms if floor is None else terms.clamp(min=floor)).mean()


def sharpe(net: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    """Sharpe ratio per period of the batch: (mean net return - R_f) / standard deviation."""
    return (net.mean() - rf.mean()) / (net.std(unbiased=False) + 1e-8)


class AttentionFactorModel(nn.Module):
    """Factor layer, closed-form loadings and residuals, and the trading policy, trained as one."""

    def __init__(self, m_firm: int, m_common: int, k: int, d: int = 32, s: int = 30, d_h: int = 32,
                 dropout: float = 0.1, squash: float = 0.001, ridge: float = 1e-5, lookback: str = "own"):
        super().__init__()
        if lookback not in ("own", "current"):
            raise ValueError(lookback)
        self.factors = AttentionFactorLayer(m_firm, m_common, k, d)
        self.policy = LongConvPolicy(s, d_h, dropout, squash)
        self.s, self.ridge, self.lookback = s, ridge, lookback

    def forward(self, x_firm, x_common, member, r_next, has_ret=None):
        """All arguments cover B + s consecutive decision dates; the first s only feed the lookback.

        Returns a dict with, for the last B dates: w [B, N] asset weights, eps [B, N] residuals of
        the next day, tradable [B, N], and w_f, beta for all B + s dates.
        """
        s = self.s
        w_f = self.factors(x_firm, x_common, member)              # [B+s, K, N]
        beta = loadings(w_f, self.ridge)
        r = torch.where(member, r_next, torch.zeros_like(r_next))
        eps = torch.where(member, residuals(r, w_f, beta), torch.zeros_like(r))  # residual of day t+1, dated t
        if self.lookback == "own":
            # lookback of the allocation dated j: residuals of days j-s+1 .. j, that is eps dated j-s .. j-1,
            # each computed with the factor model of its own date (the default; every reported run)
            look = eps.unfold(0, s, 1)[:-1]                       # [B, N, s]
        else:
            # choice "current": the residual portfolio of the decision date j applied to the returns of
            # days j-s+1 .. j, so the lookback is the recent history of the portfolio that is traded.
            # The returns are those of the securities whether or not they were members on the earlier
            # days: the factor portfolio of date j holds every member of date j.
            rw = r_next.unfold(0, s, 1)[:-1]                      # [B, N, s]: r_next dated j-s .. j-1
            f = torch.einsum("bkn,bns->bks", w_f[s:], rw)
            look = rw - torch.einsum("bkn,bks->bns", beta[s:], f)
        # choice: a security gets an allocation only if it was in the universe on the decision date
        # and on the s dates before it, and has a return on each day of the lookback (the same rule
        # as in the PCA benchmarks). A security that enters the universe waits s days.
        hist = member.unfold(0, s, 1)[:-1].all(-1)
        tradable = member[s:] & hist
        if has_ret is not None:
            tradable = tradable & has_ret.unfold(0, s, 1)[:-1].all(-1)
        alloc = self.policy(look) * tradable
        w = asset_weights(alloc, w_f[s:], beta[s:])
        valid = member if has_ret is None else member & has_ret
        return {"w": w, "eps": eps[s:], "r": r[s:], "member": member[s:], "valid": valid[s:], "tradable": tradable,
                "alloc": alloc, "w_f": w_f, "beta": beta}


def objective(out: dict, rf_next: torch.Tensor, lambda_var: float, sr_scale: float = 1.0, ev_floor: float | None = -1.0):
    """The paper's objective for one batch: net Sharpe ratio + lambda_Var * explained variance.

    Returns (objective, parts). The first decision date of the batch only provides the weights
    that the second date trades away from, so B - 1 net returns enter the Sharpe ratio.
    """
    w, r = out["w"], out["r"]
    gross = (w * r).sum(1)
    net = gross[1:] - trading_cost(w)
    sr = sharpe(net, rf_next[1:]) * sr_scale
    ev = explained_variance(out["eps"], r, out["valid"], ev_floor)
    return sr + lambda_var * ev, {"sr": sr.detach(), "ev": ev.detach(), "net": net.detach(), "gross": gross[1:].detach()}
