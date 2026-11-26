"""Properties of the model that do not depend on data. Run: python -m pytest tests -q"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attnfactors.model import (AttentionFactorLayer, AttentionFactorModel, LongConvPolicy, asset_weights,  # noqa: E402
                               explained_variance, loadings, objective, residuals, trading_cost)

torch.manual_seed(0)
B, N, K, MF, MC, S = 12, 40, 3, 6, 4, 5


def toy(b=B + S):
    x = torch.rand(b, N, MF) - 0.5
    c = torch.randn(b, MC)
    member = torch.ones(b, N, dtype=torch.bool)
    member[:, -3:] = False                       # three securities outside the universe
    r = 0.02 * torch.randn(b, N)
    return x, c, member, r


def test_factor_weights_are_long_only_and_sum_to_one():
    x, c, member, _ = toy()
    w_f = AttentionFactorLayer(MF, MC, K, d=8)(x, c, member)
    assert (w_f >= 0).all() and torch.allclose(w_f.sum(-1), torch.ones(B + S, K), atol=1e-5)
    assert (w_f[:, :, -3:] == 0).all()           # no weight outside the universe


def test_projection_properties_without_ridge():
    """With (almost) no ridge penalty: w^F beta^T = I_K and the factor portfolios of the residuals are zero."""
    x, c, member, r = toy()
    w_f = AttentionFactorLayer(MF, MC, K, d=8)(x, c, member).double()
    beta = loadings(w_f, 1e-14)
    eye = torch.eye(K, dtype=torch.float64).expand(B + S, K, K)
    assert torch.allclose(w_f @ beta.transpose(1, 2), eye, atol=1e-6)
    eps = residuals(r.double(), w_f, beta)
    assert torch.einsum("bkn,bn->bk", w_f, eps).abs().max() < 1e-8


def test_ridge_leaves_factor_exposure():
    x, c, member, r = toy()
    w_f = AttentionFactorLayer(MF, MC, K, d=8)(x, c, member).double()
    eps = residuals(r.double(), w_f, loadings(w_f, 1e-2))
    assert torch.einsum("bkn,bn->bk", w_f, eps).abs().max() > 1e-5


def test_portfolio_return_identity():
    """eps^T w^port equals R^T (w^eps)^T w^port: the residual portfolio is a portfolio of the securities."""
    x, c, member, r = toy()
    w_f = AttentionFactorLayer(MF, MC, K, d=8)(x, c, member).double()
    beta = loadings(w_f, 1e-6)
    alloc = torch.randn(B + S, N, dtype=torch.float64)
    eps = residuals(r.double(), w_f, beta)
    h = torch.einsum("bkn,bn->bk", beta, alloc)
    w_raw = alloc - torch.einsum("bkn,bk->bn", w_f, h)
    assert torch.allclose((eps * alloc).sum(1), (r.double() * w_raw).sum(1), atol=1e-10)
    w = asset_weights(alloc, w_f, beta)
    assert torch.allclose(w.abs().sum(1), torch.ones(B + S, dtype=torch.float64), atol=1e-9)


def test_inputs_equal_for_all_securities_do_not_change_the_factor_weights():
    """The shift property of the softmax: the 40 'level' inputs cannot move the factor weights."""
    x, c, member, r = toy()
    layer = AttentionFactorLayer(MF, MC, K, d=8)
    w1 = layer(x, c, member)
    w2 = layer(x, 10.0 * torch.randn_like(c), member)
    assert torch.allclose(w1, w2, atol=1e-6)
    # ... and so their rows of W get no gradient from the objective
    model = AttentionFactorModel(MF, MC, K, d=8, s=S, d_h=4)
    out = model(x, c, member, r)
    obj, _ = objective(out, torch.zeros(B), 100.0)
    obj.backward()
    g = model.factors.W.grad
    assert g[:MF].abs().max() > 1e-6
    assert g[MF:].abs().max() < 1e-6 * g[:MF].abs().max()


def test_no_look_ahead():
    """A weight dated t does not change when returns after day t or inputs after t change."""
    x, c, member, r = toy()
    model = AttentionFactorModel(MF, MC, K, d=8, s=S, d_h=4).eval()
    w = model(x, c, member, r)["w"]
    cut = S + 6                                  # position of decision date t inside the block
    x2, c2, r2 = x.clone(), c.clone(), r.clone()
    x2[cut + 1:] = torch.rand_like(x2[cut + 1:]) - 0.5
    c2[cut + 1:] = torch.randn_like(c2[cut + 1:])
    r2[cut:] = 0.05 * torch.randn_like(r2[cut:])  # r[j] is the return of day j+1
    w2 = model(x2, c2, member, r2)["w"]
    assert torch.allclose(w[: cut - S + 1], w2[: cut - S + 1], atol=1e-7)
    assert not torch.allclose(w[cut - S + 1:], w2[cut - S + 1:], atol=1e-7)


def test_trading_cost_example():
    w = torch.tensor([[0.5, -0.5, 0.0], [0.2, -0.3, 0.5]])
    # turnover |−0.3| + |0.2| + |0.5| = 1.0; short exposure 0.3
    assert torch.allclose(trading_cost(w), torch.tensor([0.0005 * 1.0 + 0.0001 * 0.3]))


def test_squash_and_last_position_convolution():
    p = LongConvPolicy(s=S, d_h=4, squash=0.5).eval()
    k = p.squashed_kernel()
    assert ((p.kernel.abs() <= 0.5) == (k == 0)).all()
    # the layer equals an explicit causal convolution read at the last position
    u = torch.randn(7, S)
    z = torch.stack([sum(k[:, lag] * u[i, S - 1 - lag] * p.in_scale for lag in range(S)) for i in range(7)])
    last = u[:, -1:] * p.in_scale
    y = p.gain * (z + p.skip * last) + p.offset * (k.sum(1) + p.skip)
    ref = p.out(torch.nn.functional.gelu(y)).squeeze(-1)
    assert torch.allclose(p(u), ref, atol=1e-5)


def test_explained_variance_is_bounded_per_security():
    """A stock whose price hardly moves must not dominate the term."""
    b, n = 40, 6
    r = 0.02 * torch.randn(b, n)
    r[:, 0] = 1e-4 * torch.randn(b)              # a takeover target: almost no variance
    eps = 0.01 * torch.randn(b, n)               # its residual is large relative to its return
    valid = torch.ones(b, n, dtype=torch.bool)
    ev = explained_variance(eps, r, valid)
    assert ev > -1.0                             # without the bound this is about -1500
    valid[: b // 2 + 2, 1] = False               # fewer than half of the days: left out
    assert torch.isfinite(explained_variance(eps, r, valid))


def test_explained_variance_without_the_bound_is_the_printed_formula():
    """floor=None is (1/N) sum_i (1 - Var(eps_i) / Var(R_i)); the bound changes the value and removes a gradient."""
    b, n = 40, 6
    r = 0.02 * torch.randn(b, n)
    r[:, 0] = 1e-4 * torch.randn(b)
    eps = (0.01 * torch.randn(b, n)).requires_grad_()
    valid = torch.ones(b, n, dtype=torch.bool)
    printed = explained_variance(eps, r, valid, floor=None)
    by_hand = (1 - eps.var(0, unbiased=False) / r.var(0, unbiased=False)).mean()
    assert torch.allclose(printed, by_hand, rtol=1e-4)
    assert printed < -100 < explained_variance(eps, r, valid)
    g_printed, = torch.autograd.grad(printed, eps, retain_graph=True)
    g_bounded, = torch.autograd.grad(explained_variance(eps, r, valid), eps)
    assert g_printed[:, 0].abs().sum() > 0 and g_bounded[:, 0].abs().sum() == 0   # below the bound: no gradient


def test_objective_passes_the_bound_on():
    """objective(..., ev_floor=None) is the printed formula; the default keeps the bound of round 1."""
    b, n = 20, 6
    r = 0.02 * torch.randn(b, n)
    r[:, 0] = 1e-4 * torch.randn(b)
    out = {"w": torch.randn(b, n) / n, "r": r, "eps": 0.01 * torch.randn(b, n), "valid": torch.ones(b, n, dtype=torch.bool)}
    rf = torch.zeros(b)
    bounded, _ = objective(out, rf, 100.0, 1.0)
    printed, parts = objective(out, rf, 100.0, 1.0, ev_floor=None)
    assert printed < bounded - 100
    assert torch.isclose(parts["ev"], explained_variance(out["eps"], r, out["valid"], None))


def test_a_security_without_a_return_in_the_lookback_gets_no_allocation():
    x, c, member, r = toy()
    has = torch.ones_like(member)
    has[S + 2, 5] = False                        # security 5 has no return on one day
    model = AttentionFactorModel(MF, MC, K, d=8, s=S, d_h=4).eval()
    out = model(x, c, member, r, has)
    hit = [j for j in range(B) if (j <= S + 2 < j + S)]   # decision dates whose lookback holds that day
    assert hit and all(not out["tradable"][j, 5] for j in hit)
    assert out["tradable"][:, 4].all()
