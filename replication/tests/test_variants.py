"""The variant options of open choices (--pca corr, --lookback current). No data needed.

Run: python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attnfactors.features import pca_window  # noqa: E402
from attnfactors.model import AttentionFactorModel  # noqa: E402
from attnfactors.train import Config, PcaData  # noqa: E402

rng = np.random.default_rng(0)


def returns(t: int, n: int) -> np.ndarray:
    """Returns with a common factor and different volatilities, so that covariance and correlation differ."""
    f = rng.normal(0, 0.01, (t, 2))
    load = rng.normal(1, 0.5, (2, n))
    return f @ load + rng.normal(0, 1, (t, n)) * rng.uniform(0.005, 0.04, n)


# ---------------------------------------------------------------- correlation PCA

def test_correlation_pca_loadings_are_the_ols_loadings():
    """With factor weights V / sigma, the OLS loadings on the same window are sigma * V (the
    identity that lets the correlation PCA skip the regression)."""
    x = returns(252, 60)
    for k in (1, 5, 20):
        v, sd = pca_window(x, True, k)
        a, b = v / sd[:, None], v * sd[:, None]
        f = (x - x.mean(0)) @ a                                       # factor returns on the window
        ols, *_ = np.linalg.lstsq(np.c_[np.ones(len(f)), f], x, rcond=None)
        assert np.allclose(ols[1:].T, b, atol=1e-8)


def test_correlation_pca_differs_from_covariance_pca():
    x = returns(252, 60)
    v_cov, sd = pca_window(x, False, 3)
    v_cor, sd2 = pca_window(x, True, 3)
    assert sd is None and sd2 is not None
    assert np.allclose(v_cov.T @ v_cov, np.eye(3), atol=1e-8) and np.allclose(v_cor.T @ v_cor, np.eye(3), atol=1e-8)
    assert abs(v_cov[:, 0] @ v_cor[:, 0]) < 0.999                    # not the same first direction


def fake_pca(t: int, n_all: int, n_pca: int, k: int, corr: bool, lookback: str = "own") -> PcaData:
    """A PcaData on synthetic returns: securities 0 .. n_pca-1 sit in the slots 0 .. n_pca-1 every day."""
    r_next = torch.from_numpy(returns(t, n_all).astype(np.float32))
    pd_ = object.__new__(PcaData)
    pd_.data = SimpleNamespace(r_next=r_next)
    pd_.mode, pd_.lookback_mode = ("corr" if corr else "cov"), lookback
    v = np.zeros((t, 500, k), dtype=np.float32)
    sd = np.zeros((t, 500), dtype=np.float32)
    cols = np.full((t, 500), -1, dtype=np.int32)
    for j in range(t):
        lo = max(0, j - 59)
        vj, sdj = pca_window(r_next[lo:j + 1, :n_pca].numpy() if j >= 10 else r_next[:11, :n_pca].numpy(), corr, k)
        v[j, :n_pca] = vj
        if corr:
            sd[j, :n_pca] = sdj
        cols[j, :n_pca] = np.arange(n_pca)
    pd_.v, pd_.sd, pd_.cols = v, (sd if corr else None), cols
    # residuals of day t+1, dated t, as PcaData.eps computes them (no cache file)
    e = np.full((t, n_all), np.nan, dtype=np.float32)
    for j in range(t):
        a, b = (x[0].double().numpy() for x in pd_.hedge(j, j + 1, k))
        r = r_next[j, :n_pca].double().numpy()
        e[j, :n_pca] = r - b[:n_pca] @ (a[:n_pca].T @ r)
    pd_._eps = {k: torch.from_numpy(e)}
    return pd_


@pytest.mark.parametrize("corr", [False, True])
def test_pca_portfolio_has_no_factor_exposure(corr):
    """The traded asset weights a - A B^T a have zero loading on every factor (beta w = B^T w = 0),
    and earn exactly the residual return of the allocation."""
    t, n_all, n_pca, k, s = 40, 30, 25, 3, 5
    pd_ = fake_pca(t, n_all, n_pca, k, corr)
    j0, j1 = s + 1, t - 1
    sub = pd_.batch_cols(j0, j1)
    cs, ok = pd_.slots(j0, j1, sub, n_all)
    alloc = torch.randn(j1 - j0, len(sub))
    w = pd_.weights(alloc, j0, j1, k, cs, ok)
    _, lb = pd_.hedge(j0, j1, k)
    w_slots = torch.gather(w, 1, cs) * ok
    assert torch.einsum("bnk,bn->bk", lb, w_slots).abs().max() < 1e-5
    a, _ = pd_.hedge(j0, j1, k)
    w_raw = alloc - torch.einsum("bnk,bk->bn", a[:, :len(sub)], torch.einsum("bnk,bn->bk", lb[:, :len(sub)], alloc))
    eps = pd_._eps[k][j0:j1][:, sub]
    r = pd_.data.r_next[j0:j1][:, sub]
    assert torch.allclose((eps * alloc).sum(1), (r * w_raw).sum(1), atol=1e-6)


# ---------------------------------------------------------------- lookback "current"

@pytest.mark.parametrize("corr", [False, True])
def test_pca_current_lookback_applies_the_decision_dates_hedge(corr):
    t, n_all, n_pca, k, s = 40, 30, 25, 3, 5
    own = fake_pca(t, n_all, n_pca, k, corr, "own")
    cur = fake_pca(t, n_all, n_pca, k, corr, "current")
    cur.v, cur.sd, cur._eps, cur.data = own.v, own.sd, own._eps, own.data      # same data and hedges
    j0, j1 = s + 1, t - 1
    sub = own.batch_cols(j0, j1)
    cs, ok = own.slots(j0, j1, sub, n_all)
    look_own, trad_own = own.lookback(k, j0, j1, s, sub, cs, ok)
    look_cur, trad_cur = cur.lookback(k, j0, j1, s, sub, cs, ok)
    assert torch.equal(trad_own, trad_cur)                           # same securities, only the signal differs
    # by hand, for decision date j and security i: R of days j-s+1 .. j minus B_j A_j^T R
    j, i = j0 + 7, 4
    a, b = (x[0, :n_pca].double().numpy() for x in own.hedge(j, j + 1, k))
    rw = own.data.r_next[j - s:j, :n_pca].double().numpy()           # [s, n_pca]
    ref = rw[:, i] - (rw @ a @ b.T)[:, i]
    assert np.allclose(look_cur[j - j0, i].numpy(), ref, atol=1e-6)
    # the default builds day d's residual with the hedge of date d-1 (known when the trade on d-1 was
    # made); "current" uses the hedge of the decision date for every day, so the two differ
    assert not torch.allclose(look_own, look_cur, atol=1e-4)


def toy(b: int, n: int = 40, mf: int = 6, mc: int = 4):
    torch.manual_seed(1)
    x = torch.rand(b, n, mf) - 0.5
    c = torch.randn(b, mc)
    member = torch.ones(b, n, dtype=torch.bool)
    member[:, -3:] = False
    r = 0.02 * torch.randn(b, n)
    return x, c, member, r


def test_attention_current_lookback_equals_own_when_the_factor_model_is_constant():
    """If the characteristics do not change, every date has the same factor model, and the two
    ways of building the residual history agree."""
    B, S = 12, 5
    x, c, member, r = toy(B + S)
    x[:] = x[0]
    model = AttentionFactorModel(6, 4, 3, d=8, s=S, d_h=4).eval()
    w_own = model(x, c, member, r)["w"]
    model.lookback = "current"
    w_cur = model(x, c, member, r)["w"]
    assert torch.allclose(w_own, w_cur, atol=1e-6)
    x2 = x.clone()
    x2[: S] = torch.rand_like(x2[: S]) - 0.5                         # older factor models differ now
    model.lookback = "own"
    w_own2 = model(x2, c, member, r)["w"]
    model.lookback = "current"
    assert not torch.allclose(w_own2, model(x2, c, member, r)["w"], atol=1e-6)


def test_attention_current_lookback_has_no_look_ahead():
    B, S = 12, 5
    x, c, member, r = toy(B + S)
    model = AttentionFactorModel(6, 4, 3, d=8, s=S, d_h=4, lookback="current").eval()
    w = model(x, c, member, r)["w"]
    cut = S + 6
    x2, c2, r2 = x.clone(), c.clone(), r.clone()
    x2[cut + 1:] = torch.rand_like(x2[cut + 1:]) - 0.5
    c2[cut + 1:] = torch.randn_like(c2[cut + 1:])
    r2[cut:] = 0.05 * torch.randn_like(r2[cut:])                     # r[j] is the return of day j+1
    w2 = model(x2, c2, member, r2)["w"]
    assert torch.allclose(w[: cut - S + 1], w2[: cut - S + 1], atol=1e-7)
    assert not torch.allclose(w[cut - S + 1:], w2[cut - S + 1:], atol=1e-7)


# ---------------------------------------------------------------- configuration

def test_variant_names_and_validation():
    assert Config(kind="pca", k=30).run_name() == "pca_K30_seed0"     # the reported runs keep their names
    assert Config(kind="pca", k=30, pca="corr").run_name() == "pca_K30_corr_seed0"
    assert Config(kind="pca", k=30, lookback="current", seed=1).run_name() == "pca_K30_look-current_seed1"
    assert Config(kind="ou", k=8, pca="corr").run_name() == "ou_K8_corr"
    assert Config(kind="attention", k=30, lookback="current").run_name() == "attention_K30_look-current_seed0"
    with pytest.raises(ValueError):
        Config(kind="attention", pca="corr")
    with pytest.raises(ValueError):
        Config(kind="pca", lookback="today")


def test_saved_config_keeps_the_old_keys_at_the_defaults():
    """Runs at the defaults save the same configuration as before the variant options existed
    (scripts/daily_sharpe_comparison.py compares saved configurations key by key)."""
    from dataclasses import asdict
    from attnfactors.train import saved_config
    c = Config(kind="attention", k=30)
    assert saved_config(c) == {k: v for k, v in asdict(c).items() if k not in ("pca", "lookback")}
    assert saved_config(Config(kind="pca", k=30, pca="corr"))["pca"] == "corr"
    assert saved_config(Config(kind="attention", k=30, lookback="current"))["lookback"] == "current"
