"""The 39 firm characteristics of Table 1 of the paper, raw (before rank normalization).

The paper points to the Internet Appendix of Chen, Pelger and Zhu (2024) for the
construction and gives no formulas. The definitions below are this replication's
reconstruction from the standard definitions in that literature (Freyberger, Neuhierl and
Weber 2020; Fama and French 1992). They are an approximation of the paper's inputs, not a
copy. The README lists the choices ("Choices the paper leaves open"); the ones made in this
file are marked "choice" where they are made.

Timing. A monthly characteristic for decision month m uses market data up to the last
trading day of month m-1. Accounting data follow the Fama-French convention: decision months
July of year y to June of year y+1 use the fiscal year that ended in calendar year y-1, and
ratios to market equity use the firm's market equity of December of year y-1. The four daily
characteristics for decision date t use data up to the close of day t.

Outputs under data/panel/: chars_monthly.npy [Mc, N, 35], chars_daily.npy [T, N, 4],
chars_meta.npz (decision months, names).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import PANEL, RAW, Panel

GROUPS: dict[str, list[str]] = {
    "past_returns": ["r2_1", "r12_2", "r12_7", "r36_13", "ST_Rev", "Ret_D1", "Ret_W1", "STD_W1"],
    "value": ["A2ME", "BEME", "C", "CF", "CF2P", "Q", "Lev", "E2P"],
    "investment": ["Investment", "NOA", "DPI2A"],
    "trading_frictions": ["AT", "LME", "LTurnover", "Rel2High", "Resid_Var", "Spread", "SUV", "Variance", "Vol", "Beta"],
    "profitability": ["PROF", "CTO", "FC2Y", "OP", "PM", "RNA", "D2A"],
    "intangibles": ["OA", "OL", "PCM"],
}
DAILY = ["Ret_D1", "Ret_W1", "STD_W1", "Vol"]
ACCOUNTING = ["A2ME", "BEME", "C", "CF", "CF2P", "Q", "Lev", "E2P", "Investment", "NOA", "DPI2A", "AT",
              "PROF", "CTO", "FC2Y", "OP", "PM", "RNA", "D2A", "OA", "OL", "PCM"]
MARKET_MONTHLY = ["r2_1", "r12_2", "r12_7", "r36_13", "ST_Rev", "LME", "LTurnover", "Rel2High", "Resid_Var",
                  "Spread", "SUV", "Variance", "Beta"]
MONTHLY = MARKET_MONTHLY + ACCOUNTING
ALL = [c for g in GROUPS.values() for c in g]
assert sorted(ALL) == sorted(MONTHLY + DAILY) and len(ALL) == 39

FIRST_DECISION_MONTH = "1989-01"


# ---------------------------------------------------------------- helpers

def _cumret(mret: np.ndarray, k: int, lo: int, hi: int, min_obs: int) -> np.ndarray:
    """Cumulative return over months k-hi .. k-lo (inclusive)."""
    w = np.log1p(mret[k - hi:k - lo + 1])
    cnt = np.isfinite(w).sum(0)
    out = np.expm1(np.nansum(w, 0))
    out[cnt < min_obs] = np.nan
    return out


def _masked_ols(x: np.ndarray, y: np.ndarray, min_obs: int):
    """Per-security OLS with missing values.

    x: [L, N, P] or [L, P] regressors; y: [L, N]. Returns (coef [N, P], resid_var [N], n [N]).
    """
    if x.ndim == 2:
        x = np.broadcast_to(x[:, None, :], (x.shape[0], y.shape[1], x.shape[1]))
    ok = np.isfinite(y) & np.isfinite(x).all(-1)
    xz = np.where(ok[..., None], x, 0.0).astype(np.float64)
    yz = np.where(ok, y, 0.0).astype(np.float64)
    xtx = np.einsum("lni,lnj->nij", xz, xz)
    xty = np.einsum("lni,ln->ni", xz, yz)
    n = ok.sum(0)
    p = x.shape[-1]
    good = n >= max(min_obs, p + 2)
    coef = np.full((y.shape[1], p), np.nan)
    rv = np.full(y.shape[1], np.nan)
    if good.any():
        a = xtx[good] + 1e-12 * np.eye(p)
        try:
            b = np.linalg.solve(a, xty[good][..., None])[..., 0]
        except np.linalg.LinAlgError:
            b = np.einsum("nij,nj->ni", np.linalg.pinv(a), xty[good])
        sse = (yz[:, good] ** 2).sum(0) - np.einsum("ni,ni->n", b, xty[good])
        coef[good] = b
        rv[good] = np.maximum(sse, 0.0) / (n[good] - p)
    return coef, rv, n


# ---------------------------------------------------------------- accounting

def _accounting(panel: Panel) -> np.ndarray:
    """acc[y, N, len(ACCOUNTING)] for fiscal years ending in calendar year 1983+y."""
    fa = pd.concat([pd.read_parquet(p) for p in sorted((RAW / "funda").glob("*.parquet"))], ignore_index=True)
    # the standard industrial record only; a stale download with other formats must not pass silently
    std = (fa["indfmt"] == "INDL") & (fa["datafmt"] == "STD") & (fa["consol"] == "C") & (fa["popsrc"] == "D")
    if not std.all():
        raise RuntimeError("data/raw/funda holds non-standard Compustat records: delete it and rerun the funda stage")
    fa = fa[fa["datadate"].between("1983-01-01", "2021-12-31")]
    if fa.duplicated(["gvkey", "datadate"]).any():
        raise RuntimeError("data/raw/funda holds duplicate (gvkey, datadate) rows")
    fa = fa.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    z = lambda c: fa[c].fillna(0.0)  # noqa: E731  (items that are zero when not reported)

    ps = fa["pstkrv"].fillna(fa["pstkl"]).fillna(fa["pstk"]).fillna(0.0)
    seq = fa["seq"].fillna(fa["ceq"] + fa["pstk"].fillna(0.0)).fillna(fa["at"] - fa["lt"])
    fa["be"] = seq + z("txditc") - ps
    fa.loc[fa["be"] <= 0, "be"] = np.nan
    fa["noa_level"] = (fa["at"] - fa["che"] - z("ivao")) - (fa["at"] - z("dlc") - z("dltt") - z("mib") - z("pstk") - fa["ceq"])

    g = fa.groupby("gvkey", sort=False)
    gap = (fa["datadate"] - g["datadate"].shift(1)).dt.days
    consecutive = gap.between(300, 430)

    def lag(c: str) -> pd.Series:
        s = g[c].shift(1)
        return s.where(consecutive)

    at1 = lag("at")
    at1 = at1.where(at1 > 0)
    sale = fa["sale"].where(fa["sale"] > 0)
    at = fa["at"].where(fa["at"] > 0)

    out = pd.DataFrame({"gvkey": fa["gvkey"], "datadate": fa["datadate"]})
    out["C"] = fa["che"] / at
    # choice: change of working capital from the balance sheet (wcap), not the cash-flow item
    out["CF_num"] = fa["ni"] + fa["dp"] - (fa["wcap"] - lag("wcap")) - z("capx")
    out["CF"] = out["CF_num"] / fa["be"]
    out["CF2P_num"] = fa["ib"] + fa["dp"] + z("txdb")
    out["Q_num"] = fa["at"] - fa["ceq"] - z("txdb")           # + market equity, / at
    debt = z("dltt") + z("dlc")
    out["Lev"] = debt / (debt + fa["seq"]).where((debt + fa["seq"]) > 0)
    out["E2P_num"] = fa["ib"]
    out["Investment"] = fa["at"] / at1 - 1.0
    out["NOA"] = fa["noa_level"] / at1
    out["DPI2A"] = ((fa["ppegt"] - lag("ppegt")) + (fa["invt"] - lag("invt"))) / at1
    out["AT"] = fa["at"]
    out["PROF"] = (fa["revt"] - fa["cogs"]) / fa["be"]
    out["CTO"] = fa["sale"] / at1
    out["FC2Y"] = (fa["xsga"] + z("xrd") + z("xad")) / sale
    out["OP"] = (fa["revt"] - fa["cogs"] - z("xsga") - z("xint")) / fa["be"]
    out["PM"] = fa["oiadp"] / sale
    noa1 = lag("noa_level")
    out["RNA"] = fa["oiadp"] / noa1.where(noa1 > 0)
    out["D2A"] = fa["dp"] / at
    out["OA"] = ((fa["act"] - lag("act")) - (fa["che"] - lag("che"))
                 - ((fa["lct"] - lag("lct")) - (z("dlc") - lag("dlc").fillna(0.0)) - (z("txp") - lag("txp").fillna(0.0)))
                 - fa["dp"]) / at1
    out["OL"] = (fa["cogs"] + fa["xsga"]) / at
    out["PCM"] = (fa["sale"] - fa["cogs"]) / sale
    out["be"] = fa["be"]
    out["at"] = fa["at"]

    # ---- link to CRSP securities: primary links first; a security without one (a second share
    # class) takes its own secondary link, which carries the same firm
    ccm = pd.read_parquet(RAW / "ccm.parquet")
    ccm = ccm[ccm["linktype"].isin(["LU", "LC"])].copy()
    ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    prim = ccm[ccm["linkprim"].isin(["P", "C"])]
    sec = ccm[ccm["linkprim"].isin(["J", "N"]) & ~ccm["lpermno"].isin(prim["lpermno"])]
    links = pd.concat([prim, sec], ignore_index=True)
    m = out.merge(links[["gvkey", "lpermno", "linkdt", "linkenddt"]], on="gvkey", how="inner")
    # the link must hold when the data are first used (end of June after the fiscal year), so that
    # a firm listed after its fiscal-year end keeps its first report
    use = pd.to_datetime((m["datadate"].dt.year + 1).astype(str) + "-06-30")
    m = m[(use >= m["linkdt"]) & (use <= m["linkenddt"])]
    pidx = {int(p): i for i, p in enumerate(panel.permnos)}
    m = m[m["lpermno"].isin(pidx)].copy()
    m["col"] = m["lpermno"].map(pidx).astype(int)
    m["year"] = m["datadate"].dt.year
    m = m.sort_values(["col", "datadate"]).drop_duplicates(["col", "year"], keep="last")
    m = m[(m["year"] >= 1983) & (m["year"] <= 2021)]

    # ---- market equity of the firm (all share classes) in December of the fiscal-year-end year, million USD
    midx = {mm: i for i, mm in enumerate(panel.months)}
    me = panel.monthly("me_firm")
    rows = np.array([midx.get(f"{y}-12", -1) for y in m["year"]])
    me_dec = np.where(rows >= 0, me[np.maximum(rows, 0), m["col"].values], np.nan) / 1000.0
    me_dec = np.where(me_dec > 0, me_dec, np.nan)
    m["A2ME"] = m["at"] / me_dec
    m["BEME"] = m["be"] / me_dec
    m["CF2P"] = m["CF2P_num"] / me_dec
    m["Q"] = (m["Q_num"] + me_dec) / m["at"].where(m["at"] > 0)
    m["E2P"] = m["E2P_num"] / me_dec

    acc = np.full((2021 - 1983 + 1, len(panel.permnos), len(ACCOUNTING)), np.nan, dtype=np.float32)
    vals = m[ACCOUNTING].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    acc[m["year"].values - 1983, m["col"].values] = vals
    return acc


# ---------------------------------------------------------------- build

DEFINITIONS = ("reconstructed", "cited")


def _fp_beta(lr: np.ndarray, lm: np.ndarray, r3: np.ndarray, m3: np.ndarray, e: int) -> np.ndarray:
    """Market beta as the source of the characteristics defines it (Frazzini and Pedersen 2014):
    correlation of overlapping three-day log returns over five years, times the ratio of the
    volatilities of daily log returns over one year.

    lr [T, N], lm [T]: daily log returns of the securities and of the market; r3, m3: their
    three-day sums. e: last day of the window. Minimum numbers of observations of the source: 120
    days for the volatility, 750 for the correlation. Data limit: the daily panel starts in 1988,
    so until the end of 1992 the correlation window is shorter than five years; while it is, 250
    observations are required instead of 750.
    """
    a = max(e - 251, 0)
    v = lr[a:e + 1]
    n_v = np.isfinite(v).sum(0)
    with np.errstate(all="ignore"):
        sd_i = np.nanstd(v, 0, ddof=1)
    sd_m = lm[a:e + 1].std(ddof=1)
    c0 = max(e - 1259, 2)
    x, y = r3[c0:e + 1], m3[c0:e + 1][:, None]
    ok = np.isfinite(x)
    n_c = ok.sum(0)
    need = 750 if e - 1259 >= 2 else 250
    cnt = np.maximum(n_c, 1)
    xm = np.where(ok, x, 0.0).sum(0) / cnt
    ym = (ok * y).sum(0) / cnt
    dx, dy = np.where(ok, x - xm, 0.0), np.where(ok, y - ym, 0.0)
    with np.errstate(all="ignore"):
        corr = (dx * dy).sum(0) / np.sqrt((dx ** 2).sum(0) * (dy ** 2).sum(0))
        beta = corr * sd_i / sd_m
    beta[(n_v < 120) | (n_c < need)] = np.nan
    return beta


def build(definitions: str = "reconstructed") -> None:
    """``definitions``: "reconstructed" is the first reading of this replication (the numbers on the
    slides). "cited" computes Beta, Variance and Resid_Var as the source that the paper cites for
    its characteristics defines them (scripts/replicate_as_cited.py); everything else is equal."""
    if definitions not in DEFINITIONS:
        raise ValueError(f"definitions must be one of {DEFINITIONS}")
    cited = definitions == "cited"
    panel = Panel()
    ret = panel.daily("ret")
    t_n, n = ret.shape
    months = list(panel.months)
    months_ext = months + [str(pd.Period(months[-1], "M") + 1)]   # one decision month after the last data month
    mret, me, mvol, mshr = (panel.monthly(f) for f in ["ret", "me", "vol", "shrout"])
    ff = panel.ff.astype(np.float64)
    rf = np.nan_to_num(ff[:, 3])

    # ---- daily characteristics (known at the close of day t)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log1p(ret.astype(np.float64))
    xd = np.full((t_n, n, len(DAILY)), np.nan, dtype=np.float32)
    xd[:, :, 0] = ret
    win = np.lib.stride_tricks.sliding_window_view(lr, 5, axis=0)            # [T-4, N, 5]
    xd[4:, :, 1] = np.expm1(win.sum(-1))
    winr = np.lib.stride_tricks.sliding_window_view(ret, 5, axis=0)
    xd[4:, :, 2] = winr.std(-1, ddof=1)
    # choice: "weekly trading volume" as dollar volume of the last five days. Nasdaq volume counts
    # dealer trades on both sides until about 2001-2003 (Gao and Ritter 2010); not adjusted.
    dvol = (panel.daily("prc") * panel.daily("vol")).astype(np.float64)
    xd[4:, :, 3] = np.lib.stride_tricks.sliding_window_view(dvol, 5, axis=0).sum(-1)
    np.save(PANEL / "chars_daily.npy", xd)
    del xd, win, winr, dvol

    # ---- monthly characteristics
    month_of_day = panel.month_of_day
    last_day = {m: int(np.where(month_of_day == m)[0][-1]) for m in np.unique(month_of_day)}
    first_day = {m: int(np.where(month_of_day == m)[0][0]) for m in np.unique(month_of_day)}
    acc = _accounting(panel)
    dec_months = [m for m in months_ext if m >= FIRST_DECISION_MONTH]
    xm = np.full((len(dec_months), n, len(MONTHLY)), np.nan, dtype=np.float32)
    col = {c: i for i, c in enumerate(MONTHLY)}
    adj = (panel.daily("prc") / panel.daily("cfacpr")).astype(np.float64)
    bid, ask = panel.daily("bid").astype(np.float64), panel.daily("ask").astype(np.float64)
    # data limit: CRSP has no closing bid and ask for NYSE and AMEX stocks before the end of 1992
    spread_d = np.where((bid > 0) & (ask >= bid), 2 * (ask - bid) / (ask + bid), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        turn_d = (panel.daily("vol") / panel.daily("shrout")).astype(np.float64)   # split-safe daily turnover
    exret = ret.astype(np.float64) - rf[:, None]
    x3 = np.column_stack([np.ones(t_n), ff[:, 0], ff[:, 1], ff[:, 2]])
    x1 = np.column_stack([np.ones(t_n), ff[:, 0]])
    rpos, rneg = np.maximum(ret, 0).astype(np.float64), np.maximum(-ret, 0).astype(np.float64)
    if cited:
        lm = np.log1p(ff[:, 0] + rf)                                   # market return: excess return plus risk-free rate
        lrb = np.where(np.isfinite(lr), lr, np.nan)                    # a return of -100 percent has no logarithm
        r3, m3 = np.full_like(lrb, np.nan), np.full_like(lm, np.nan)
        r3[2:] = lrb[2:] + lrb[1:-1] + lrb[:-2]                        # missing if one of the three days is missing
        m3[2:] = lm[2:] + lm[1:-1] + lm[:-2]

    for i, m in enumerate(dec_months):
        k = months_ext.index(m)
        prev = months_ext[k - 1]
        # -- from monthly returns
        xm[i, :, col["ST_Rev"]] = mret[k - 1]
        # Freyberger, Neuhierl and Weber: the return one month before prediction. This duplicates
        # ST_Rev, as it appears to in the paper's sources.
        xm[i, :, col["r2_1"]] = mret[k - 1]
        xm[i, :, col["r12_2"]] = _cumret(mret, k, 2, 12, 11)
        xm[i, :, col["r12_7"]] = _cumret(mret, k, 7, 12, 6)
        xm[i, :, col["r36_13"]] = _cumret(mret, k, 13, 36, 18)
        xm[i, :, col["LME"]] = me[k - 1]                               # of the security (data/company: of the company)
        with np.errstate(divide="ignore", invalid="ignore"):
            xm[i, :, col["LTurnover"]] = mvol[k - 1] / mshr[k - 1]     # Nasdaq volume not adjusted, see above
        # -- from daily data up to the last trading day of the previous month
        e = last_day.get(prev)
        if e is not None and e >= 251:
            with np.errstate(all="ignore"):
                xm[i, :, col["Rel2High"]] = adj[e] / np.nanmax(adj[e - 251:e + 1], 0)
                # reconstructed: 60 trading days. cited: the two calendar months before the decision month
                v0 = first_day.get(months_ext[k - 2], e - 59) if cited else e - 59
                w = ret[v0:e + 1].astype(np.float64)                   # at least 20 days
                v = np.nanvar(w, 0, ddof=1)
                v[np.isfinite(w).sum(0) < 20] = np.nan
                xm[i, :, col["Variance"]] = v
            # residual variance of a three-factor regression on the same days
            _, rv, _ = _masked_ols(x3[v0:e + 1], exret[v0:e + 1], 20)
            xm[i, :, col["Resid_Var"]] = rv
            if cited:
                xm[i, :, col["Beta"]] = _fp_beta(lrb, lm, r3, m3, e)
            else:
                # reconstructed: market beta as an OLS slope on 252 days (not the Frazzini-Pedersen estimator)
                b, _, _ = _masked_ols(x1[e - 251:e + 1], exret[e - 251:e + 1], 126)
                xm[i, :, col["Beta"]] = b[:, 1]
            s0 = first_day[prev]
            with np.errstate(all="ignore"):
                xm[i, :, col["Spread"]] = np.nanmean(spread_d[s0:e + 1], 0)
            # standardized unexplained volume: turnover of last month against a model fitted on the 60 days before it
            if s0 >= 60:
                xr = np.stack([np.ones((60, n)), rpos[s0 - 60:s0], rneg[s0 - 60:s0]], -1)
                c3, rv3, _ = _masked_ols(xr, turn_d[s0 - 60:s0], 40)
                xe = np.stack([np.ones((e + 1 - s0, n)), rpos[s0:e + 1], rneg[s0:e + 1]], -1)
                with np.errstate(all="ignore"):
                    resid = turn_d[s0:e + 1] - np.einsum("lnp,np->ln", xe, c3)
                    xm[i, :, col["SUV"]] = np.nanmean(resid, 0) / np.sqrt(rv3)
        # -- accounting, Fama-French timing
        y, mo = int(m[:4]), int(m[5:])
        fy = (y - 1 if mo >= 7 else y - 2) - 1983
        if 0 <= fy < acc.shape[0]:
            for c in ACCOUNTING:
                xm[i, :, col[c]] = acc[fy, :, ACCOUNTING.index(c)]

    xm[~np.isfinite(xm)] = np.nan
    np.save(PANEL / "chars_monthly.npy", xm)
    np.savez(PANEL / "chars_meta.npz", dec_months=np.array(dec_months), monthly=np.array(MONTHLY), daily=np.array(DAILY))
    print("chars:", definitions, "definitions; decision months", dec_months[0], "..", dec_months[-1])


if __name__ == "__main__":
    build()
