"""Properties of the data code that need no data. Run: python -m pytest tests -q"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attnfactors.features import rank_row  # noqa: E402
from attnfactors.train import PcaData, batches  # noqa: E402
from attnfactors.wrds_api import _matches  # noqa: E402


def test_ranks_lie_in_the_open_interval_and_ties_share_a_rank():
    r = rank_row(np.array([3.0, 1.0, 2.0, 2.0, 5.0]))
    assert np.all(r > -0.5) and np.all(r < 0.5)
    assert r[2] == r[3]                                    # tied values
    assert np.allclose(r, np.array([4, 1, 2.5, 2.5, 5]) / 6 - 0.5)
    assert np.allclose(rank_row(np.array([0.3, 0.1, 0.2])), np.array([3, 1, 2]) / 4 - 0.5)


def test_batches_cover_every_date_once():
    rng = np.random.default_rng(0)
    for lo, hi, size in [(100, 2116, 64), (0, 505, 128), (7, 70, 64)]:
        b = batches(lo, hi, size, rng)
        days = sorted(d for j0, j1 in b for d in range(j0, j1))
        assert days == list(range(lo, hi))
        assert all(j1 - j0 >= min(size // 2, hi - lo) for j0, j1 in b)


def test_pca_lookback_keeps_the_first_security_tradable():
    """Empty PCA slots (index -1) must not switch off security 0."""
    t, n, s, k = 12, 6, 3, 2
    pd_ = object.__new__(PcaData)
    cols = np.full((t, 500), -1, dtype=np.int32)
    cols[:, :4] = np.array([0, 2, 3, 5])                   # securities 1 and 4 are outside the PCA
    pd_.cols = cols
    eps = torch.randn(t, n)
    eps[:, [1, 4]] = float("nan")
    pd_._eps = {k: eps}
    sub = pd_.batch_cols(s + 1, t)
    assert sub.tolist() == [0, 2, 3, 5]                    # securities 1 and 4 never enter
    cs, ok = pd_.slots(s + 1, t, sub, n)
    look, tradable = pd_.lookback(k, s + 1, t, s, sub, cs, ok)
    assert tradable.all() and look.shape == (t - s - 1, 4, s)


def test_filters_are_checked_locally():
    row = {"indfmt": "FS", "datafmt": "STD", "datadate": "2020-06-30", "gvkey": "012141", "permno": 10107}
    assert not _matches(row, "indfmt", "INDL")
    assert _matches(row, "datafmt", "STD")
    assert _matches(row, "datadate__gte", "1983-01-01") and not _matches(row, "datadate__lte", "2019-12-31")
    assert _matches(row, "gvkey__in", "001690,012141") and not _matches(row, "gvkey__in", "001690")
    assert _matches(row, "permno", "10107") and _matches(row, "missing_column", "x")


def test_regression_pca_residual_is_the_portfolio_the_weights_describe():
    """The residual of pca_regression equals the return of the asset portfolio alloc - a (b^T alloc),
    its loadings are the 60-day regression without intercept, and the PCA is of the correlation matrix."""
    from attnfactors.pca_regression import LOADING_WINDOW, residual_parts
    rng = np.random.default_rng(3)
    n, k = 40, 5
    scale = rng.uniform(0.5, 4.0, n)                        # very different volatilities
    x = (rng.standard_normal((252, 3)) @ rng.standard_normal((3, n)) + rng.standard_normal((252, n))) * 0.01 * scale
    a, betas = residual_parts(x, (k,))
    b = betas[k]
    vol = x.std(0)
    # correlation, not covariance: the result does not change when one stock's returns are rescaled
    x2 = x.copy()
    x2[:, 0] *= 7.0
    a2, betas2 = residual_parts(x2, (k,))
    assert np.allclose(np.abs(a2[1:, 0]), np.abs(a[1:, 0]), rtol=1e-6)
    assert np.allclose(np.abs(a2[0, 0]) * 7.0, np.abs(a[0, 0]), rtol=1e-6)
    # a = eigenvectors of the correlation matrix over the volatilities
    lam, vec = np.linalg.eigh(np.corrcoef(x.T))
    assert np.allclose(np.abs(a[:, 0] * vol), np.abs(vec[:, -1]), atol=1e-8)
    # loadings: least squares of the last 60 days on the factor returns, no intercept
    f = x[-LOADING_WINDOW:] @ a
    assert np.allclose(b, np.linalg.lstsq(f, x[-LOADING_WINDOW:], rcond=None)[0].T, atol=1e-8)
    # residual of a new day and the portfolio identity
    xn, alloc = rng.standard_normal(n) * 0.01, rng.standard_normal(n)
    eps = xn - b @ (a.T @ xn)
    assert np.isclose(alloc @ eps, (alloc - a @ (b.T @ alloc)) @ xn)
    # not an orthogonal projection: the residuals of the fitted 60 days are orthogonal to the factors, in sample only
    e60 = x[-LOADING_WINDOW:] - f @ b.T
    assert np.abs(f.T @ e60).max() < 1e-10


def test_cited_beta_is_correlation_of_three_day_returns_times_the_volatility_ratio():
    from attnfactors.chars import _fp_beta
    rng = np.random.default_rng(5)
    t, e = 1400, 1399
    lm = rng.standard_normal(t) * 0.01
    lr = np.stack([1.5 * lm + rng.standard_normal(t) * 0.01, 0.5 * lm + rng.standard_normal(t) * 0.02,
                   lm.copy(), rng.standard_normal(t) * 0.01], 1)
    lr[:900, 3] = np.nan                                      # listed for 500 days: too short for the correlation
    lr[700, 0] = np.nan                                       # one missing day
    r3, m3 = np.full_like(lr, np.nan), np.full_like(lm, np.nan)
    r3[2:] = lr[2:] + lr[1:-1] + lr[:-2]
    m3[2:] = lm[2:] + lm[1:-1] + lm[:-2]
    b = _fp_beta(lr, lm, r3, m3, e)
    assert np.isclose(b[2], 1.0)                              # the market itself
    assert np.isnan(b[3])                                     # fewer than 750 three-day returns
    w = slice(e - 1259, e + 1)
    ok = np.isfinite(r3[w, 0])
    corr = np.corrcoef(r3[w, 0][ok], m3[w][ok])[0, 1]
    vol = np.nanstd(lr[e - 251:e + 1, 0], ddof=1) / lm[e - 251:e + 1].std(ddof=1)
    assert np.isclose(b[0], corr * vol)
    # no later data: the value for day e does not change when the days after e change
    lr2, lm2 = lr.copy(), lm.copy()
    b_early = _fp_beta(lr, lm, r3, m3, 1300)
    lr2[1301:], lm2[1301:] = 0.0, 0.0
    r32, m32 = np.full_like(lr2, np.nan), np.full_like(lm2, np.nan)
    r32[2:] = lr2[2:] + lr2[1:-1] + lr2[:-2]
    m32[2:] = lm2[2:] + lm2[1:-1] + lm2[:-2]
    assert np.allclose(b_early, _fp_beta(lr2, lm2, r32, m32, 1300), equal_nan=True)


def test_regression_pca_weights_are_the_portfolio_of_the_residuals():
    """PcaRegressionData.weights maps an allocation over residuals to asset weights whose return on the
    excess returns is the allocation times the residuals (before the scaling to an L1 norm of one)."""
    from attnfactors.pca_regression import PcaRegressionData, residual_parts
    rng = np.random.default_rng(11)
    n, k, days = 12, 3, 2
    pd_ = object.__new__(PcaRegressionData)
    a = np.zeros((days, 500, k), dtype=np.float32)
    b = np.zeros((days, 500, k), dtype=np.float32)
    xs = []
    for t in range(days):
        x = rng.standard_normal((252, n)) * 0.01 * rng.uniform(0.5, 3.0, n)
        at, bt = residual_parts(x, (k,))
        a[t, :n], b[t, :n] = at, bt[k]
        xs.append(rng.standard_normal(n) * 0.01)
    pd_.a, pd_._b = a, {k: b}
    cs = torch.arange(500).clamp(max=n - 1)[None, :].repeat(days, 1)      # slot i holds security i; the rest is empty
    ok = torch.zeros(days, 500, dtype=torch.bool)
    ok[:, :n] = True
    alloc = torch.randn(days, n)
    w = pd_.weights(alloc, 0, days, k, cs, ok)
    for t in range(days):
        raw = alloc[t].numpy() - a[t, :n] @ (b[t, :n].T @ alloc[t].numpy())
        eps = xs[t] - b[t, :n] @ (a[t, :n].T @ xs[t])
        assert np.allclose(w[t].numpy(), raw / np.abs(raw).sum(), atol=1e-5)          # the day's own a and b
        assert np.isclose(raw @ xs[t], alloc[t].numpy() @ eps, atol=1e-7)
    assert not np.allclose(w[0].numpy(), (alloc[0].numpy() - b[0, :n] @ (a[0, :n].T @ alloc[0].numpy())) /
                           np.abs(alloc[0].numpy() - b[0, :n] @ (a[0, :n].T @ alloc[0].numpy())).sum(), atol=1e-3)   # a and b swapped


def test_history_rule_counts_months_without_a_gap():
    """The universe of round 4 asks for 120 months with a return: the count starts again after a missing
    month, after a month without a return, and for every security."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from replicate_company_universe import consecutive_returns, to_universe
    permno = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])
    month = np.array([0, 1, 2, 4, 5, 3, 4, 5, 6])            # security 1 has no row for month 3
    has = np.array([False, True, True, True, True, True, False, True, True])
    assert consecutive_returns(permno, month, has).tolist() == [0, 1, 2, 1, 2, 1, 0, 1, 2]
    # a ranking made at the end of a month is the universe of the next month
    import pandas as pd
    x = pd.DataFrame({"permno": [7, 8], "permco": [1, 2], "month": ["2020-12", "2020-12"], "value": [5.0, 9.0], "rank": [2.0, 1.0]})
    u = to_universe(x)
    assert u["month"].tolist() == ["2021-01", "2021-01"] and u["permno"].tolist() == [8, 7]


def test_a_day_without_a_trade_has_no_return_and_neither_has_the_next(tmp_path):
    """Round 4: a volume of zero marks a day without a trade. Its return and the return of the next day
    go; a delisting return, booked on a day without a price, stays."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from replicate_company_universe import no_trade_days
    nan = np.nan
    prc = np.array([[10.0, 5.0], [10.5, 5.1], [10.2, 5.2], [10.3, nan], [10.4, nan]], dtype=np.float32)
    vol = np.array([[100, 50], [0, 60], [90, 0], [80, nan], [70, nan]], dtype=np.float32)
    ret = np.array([[0.01, 0.01], [0.05, 0.02], [-0.03, 0.02], [0.01, -0.30], [0.01, nan]], dtype=np.float32)
    for name, a in (("daily_prc.npy", prc), ("daily_vol.npy", vol), ("daily_ret.npy", ret), ("member.npy", np.ones((5, 2), bool))):
        np.save(tmp_path / name, a)
    no_trade_days(tmp_path)
    out = np.load(tmp_path / "daily_ret.npy")
    # security 0: no trade on day 1, so days 1 and 2 go. security 1: no trade on day 2;
    # day 3 carries its delisting return on a day without a price and stays
    assert np.array_equal(np.isnan(out), np.array([[False, False], [True, False], [True, True], [False, False], [False, True]]))
    assert np.isclose(out[3, 1], -0.30)


def test_company_value_follows_splits_and_ignores_a_share_count_that_shows_one_early():
    """Round 4: price times the company's quarterly shares, carried between quarter ends in split-adjusted
    units; the median of three quarters removes a share count that shows a split before the price does."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import pandas as pd
    from replicate_company_universe import company_value_table, ranking_value
    months = [f"2000-{m:02d}" for m in range(1, 13)]

    def table(cshoq, ajexm, prccm):
        return pd.DataFrame({"gvkey": "1", "iid": "01", "month": months, "prccm": prccm, "cshoq": cshoq, "ajexm": ajexm})

    nan = np.nan
    # 100 million shares at 50; a two-for-one split takes effect in August, between two quarter ends
    cs = [nan, nan, 100, nan, nan, 100, nan, nan, 200, nan, nan, 200]
    aj = [2.0] * 7 + [1.0] * 5
    pr = [50.0] * 7 + [25.0] * 5
    v = company_value_table(table(cs, aj, pr)).set_index("month")["company"]
    assert np.allclose(v.loc["2000-03":].to_numpy(), 5_000_000.0)            # 5 billion USD in every month, in thousand USD
    assert "2000-02" not in v.index                                           # no share count yet
    # the September count is reported after the split although price and factor change in October only
    aj2, pr2 = [2.0] * 9 + [1.0] * 3, [50.0] * 9 + [25.0] * 3
    cs2 = [nan, nan, 100, nan, nan, 100, nan, nan, 200, nan, nan, 200]
    v2 = company_value_table(table(cs2, aj2, pr2)).set_index("month")["company"]
    assert np.allclose(v2.loc["2000-03":].to_numpy(), 5_000_000.0)
    # nothing after a month changes its value
    early = company_value_table(table(cs, aj, pr).iloc[:8]).set_index("month")["company"]
    assert np.allclose(early.to_numpy(), v.loc[:"2000-08"].to_numpy())
    # the firm's value stands on its largest security; a second class ranks with its own value
    me = pd.Series([90.0, 10.0, 40.0]); firm = pd.Series([500.0, 500.0, 40.0])
    assert ranking_value(me, firm, pd.Series([1, 1, 2]), pd.Series(["m", "m", "m"])).tolist() == [500.0, 10.0, 40.0]
