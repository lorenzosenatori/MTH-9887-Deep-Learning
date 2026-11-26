"""The whole-window gradient must include costs and variance across chunk boundaries."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from replicate_adjusted_loss import AttentionFactorModel, Config, full_loss, loss, variance_terms
from attnfactors.model import explained_variance, objective


def test_uncapped_term_and_historical_eligibility():
    ret = torch.tensor([[.01, .01, 0.], [.02, .02, 0.], [.03, .03, 0.], [.04, .04, 0.]])
    eps = (3 * ret).requires_grad_()
    valid = torch.ones_like(ret, dtype=torch.bool)
    valid[1:, 1] = False
    terms = variance_terms(eps, ret, valid)
    torch.testing.assert_close(terms, torch.tensor([-8.]))
    torch.testing.assert_close(terms.clamp(min=-1).mean(), explained_variance(eps, ret, valid))
    terms.mean().backward()
    assert eps.grad[:, 0].abs().sum() > 0
    assert torch.count_nonzero(eps.grad[:, 1:]) == 0


def test_full_window_value_and_gradient_match_one_forward():
    torch.manual_seed(123)
    s, n, days = 3, 9, 15
    model = AttentionFactorModel(4, 0, 3, d=5, s=s, d_h=6, dropout=0, ridge=1e-3).double()
    x = torch.randn(days + s, n, 4, dtype=torch.float64)
    member = torch.ones(days + s, n, dtype=torch.bool)
    member[:9, -1] = False  # a new stock enters between chunks
    ret = 0.02 * torch.randn(days + s, n, dtype=torch.float64)
    valid = member.clone()
    valid[10, 2] = False
    rf = torch.linspace(0, .0001, days, dtype=torch.float64)
    blocks = []
    for a, b in ((0, 6), (6, days)):
        t = slice(a, b + s)
        cols = member[t].any(0).nonzero().flatten()
        blocks.append((cols, (x[t][:, cols], None, member[t][:, cols], ret[t][:, cols], valid[t][:, cols])))
    cfg = Config(sr_scale=1)
    output = model(x, None, member, ret, valid)
    for capped in (True, False):
        direct = loss(output['w'], output['eps'], output['r'], output['valid'], rf, cfg, capped)
        if capped:
            original, _ = objective(output, rf, cfg.lambda_var, cfg.sr_scale)
            torch.testing.assert_close(direct, -original)
        chunked = full_loss(model, (blocks, ret[s:], valid[s:]), rf, cfg, capped)
        torch.testing.assert_close(direct, chunked)
        expected = torch.autograd.grad(direct, tuple(model.parameters()), retain_graph=True)
        actual = torch.autograd.grad(chunked, tuple(model.parameters()))
        for g1, g2 in zip(expected, actual):
            torch.testing.assert_close(g1, g2)
