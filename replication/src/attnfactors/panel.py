"""Turn the raw WRDS downloads into dense arrays indexed by (trading day, security).

Output, under data/panel/ (all licensed data: never commit):
  meta.npz        dates [T], months [Mn] (as 'YYYY-MM'), permnos [N], month_of_day [T], month_of_ret [T]
  daily_*.npy     ret, prc, vol, shrout, bid, ask, cfacpr   float32 [T, N]
  member.npy      bool [T, N]: security is in the universe on decision date t
  monthly_*.npy   ret, me, me_firm, vol, shrout             float32 [Mn, N]
  ff.npy          float32 [T, 4]: mktrf, smb, hml, rf

Universe timing. The weight dated t earns the return of day t+1, so decision date t uses the
universe of the month of day t+1 (``month_of_ret``): the 500 largest stocks at the end of the
month before. On the last trading day of a month this is the ranking of that same day's close.

Delisting returns (choice). CRSP's delisting return runs from the last traded price to the
value received afterwards. It is stored as the return of the trading day after the last trade,
so that the weight held at the last close earns it and no input dated the last trading day
contains it.

Timing convention of the whole project: everything stored at day index t is known at the
close of day t. A portfolio weight "dated t" is chosen at the close of day t and earns the
return of day t+1.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# data/ unless ATTNFACTORS_DATA names another root with the same layout (raw/, universe.parquet, panel
# folders): data/company holds the universe ranked by company value (scripts/replicate_company_universe.py)
DATA = ROOT / os.environ.get("ATTNFACTORS_DATA", "data")
RAW = DATA / "raw"
# data/panel unless ATTNFACTORS_PANEL names another folder under data/ (a second set of inputs, for
# example data/panel_as_cited, built by scripts/replicate_as_cited.py)
PANEL = DATA / os.environ.get("ATTNFACTORS_PANEL", "panel")

DAILY_FIELDS = ["ret", "prc", "vol", "shrout", "bid", "ask", "cfacpr"]
FIRST_DAY = "1988-01-01"
LAST_DAY = "2021-12-31"


def build() -> None:
    PANEL.mkdir(parents=True, exist_ok=True)
    for f in PANEL.glob("pca_eps_*.npy"):        # residual caches of an older panel (both PCA modes)
        f.unlink()
    uni = pd.read_parquet(DATA / "universe.parquet")
    uni = uni[uni["month"] >= FIRST_DAY[:7]]
    permnos = np.array(sorted(uni["permno"].unique()), dtype=np.int64)
    pidx = {int(p): i for i, p in enumerate(permnos)}
    n = len(permnos)

    ff = pd.read_parquet(RAW / "ff_daily.parquet")
    dsi = pd.read_parquet(RAW / "dsi.parquet")
    cal = dsi[(dsi["date"] >= FIRST_DAY) & (dsi["date"] <= LAST_DAY)]["date"].sort_values().reset_index(drop=True)
    dates = cal.values.astype("datetime64[D]")
    t_n = len(dates)
    didx = {d: i for i, d in enumerate(dates)}

    # ---- daily fields
    arr = {f: np.full((t_n, n), np.nan, dtype=np.float32) for f in DAILY_FIELDS}
    for p in permnos:
        df = pd.read_parquet(RAW / "dsf" / f"{int(p)}.parquet")
        if df.empty:
            continue
        d = df["date"].values.astype("datetime64[D]")
        rows = np.array([didx.get(x, -1) for x in d])
        ok = rows >= 0
        j = pidx[int(p)]
        for f in DAILY_FIELDS:
            arr[f][rows[ok], j] = df[f].values[ok].astype(np.float32)
    arr["prc"] = np.abs(arr["prc"])  # CRSP marks a bid/ask average with a negative sign

    # ---- delisting returns: merged into the daily return on the delisting day
    dl = pd.read_parquet(RAW / "dsedelist.parquet")
    n_dl = 0
    for r in dl.itertuples(index=False):
        j = pidx.get(int(r.permno))
        if j is None or pd.isna(r.dlstdt):
            continue
        t = didx.get(np.datetime64(r.dlstdt, "D"))
        if t is None:
            continue
        dlret = r.dlret
        if pd.isna(dlret):
            # Shumway (1997): performance-related delistings without a delisting return
            dlret = -0.30 if (not pd.isna(r.dlstcd) and 500 <= int(r.dlstcd) < 600) else np.nan
        if pd.isna(dlret):
            continue
        t1 = t + 1                                   # realized after the last trade
        if t1 >= t_n:
            continue
        cur = arr["ret"][t1, j]
        arr["ret"][t1, j] = dlret if np.isnan(cur) else (1 + cur) * (1 + dlret) - 1
        n_dl += 1

    # ---- universe membership per decision date
    months_of_day = pd.DatetimeIndex(dates).to_period("M").astype(str).values.astype("U7")
    # month of the return that the weight dated t earns; the sample ends on the last trading day of a month
    after_last = str(pd.Period(months_of_day[-1], "M") + 1)
    month_of_ret = np.append(months_of_day[1:], after_last).astype("U7")
    member = np.zeros((t_n, n), dtype=bool)
    by_month = {m: g["permno"].map(pidx).values for m, g in uni.groupby("month")}
    for m in np.unique(month_of_ret):
        cols = by_month.get(m)
        if cols is None:
            continue
        rows = np.where(month_of_ret == m)[0]
        member[np.ix_(rows, cols)] = True

    # ---- monthly fields
    msf = pd.concat([pd.read_parquet(p) for p in sorted((RAW / "msf").glob("*.parquet"))], ignore_index=True)
    msf["month"] = msf["date"].dt.to_period("M").astype(str)
    msf["me"] = msf["altprc"].abs().fillna(msf["prc"].abs()) * msf["shrout"]
    # market equity of the firm: all share classes (permco), as Fama and French define it
    msf["me_firm"] = msf.groupby(["permco", "month"])["me"].transform("sum")
    msf = msf[msf["permno"].isin(permnos)].copy()
    months = np.array(sorted(msf["month"].unique()), dtype="U7")
    midx = {m: i for i, m in enumerate(months)}
    mon = {f: np.full((len(months), n), np.nan, dtype=np.float32) for f in ["ret", "me", "me_firm", "vol", "shrout"]}
    r = msf["month"].map(midx).values
    c = msf["permno"].map(pidx).values
    for f in mon:
        mon[f][r, c] = msf[f].values.astype(np.float32)

    # ---- factors
    ffc = ff.set_index("date").reindex(pd.DatetimeIndex(dates))[["mktrf", "smb", "hml", "rf"]]
    ffa = ffc.values.astype(np.float32)

    for f in DAILY_FIELDS:
        np.save(PANEL / f"daily_{f}.npy", arr[f])
    for f in mon:
        np.save(PANEL / f"monthly_{f}.npy", mon[f])
    np.save(PANEL / "member.npy", member)
    np.save(PANEL / "ff.npy", ffa)
    np.savez(PANEL / "meta.npz", dates=dates, months=months, permnos=permnos, month_of_day=months_of_day,
             month_of_ret=month_of_ret)
    print(f"panel: {t_n} days {dates[0]} .. {dates[-1]}, {n} securities, {len(months)} months; "
          f"{n_dl} delisting returns merged; members per day min {member[member.any(1)].sum(1).min()} "
          f"max {member.sum(1).max()}; ff missing days {int(np.isnan(ffa).any(1).sum())}")


class Panel:
    """Read-only access to the arrays written by build()."""

    def __init__(self) -> None:
        meta = np.load(PANEL / "meta.npz", allow_pickle=False)
        self.dates = meta["dates"]
        self.months = meta["months"]
        self.permnos = meta["permnos"]
        self.month_of_day = meta["month_of_day"]
        self.month_of_ret = meta["month_of_ret"]
        self.member = np.load(PANEL / "member.npy")
        self.ff = np.load(PANEL / "ff.npy")
        self._cache: dict[str, np.ndarray] = {}

    def daily(self, f: str) -> np.ndarray:
        if f not in self._cache:
            self._cache[f] = np.load(PANEL / f"daily_{f}.npy")
        return self._cache[f]

    def monthly(self, f: str) -> np.ndarray:
        k = "m_" + f
        if k not in self._cache:
            self._cache[k] = np.load(PANEL / f"monthly_{f}.npy")
        return self._cache[k]

    @property
    def rf(self) -> np.ndarray:
        return np.nan_to_num(self.ff[:, 3])


if __name__ == "__main__":
    build()
