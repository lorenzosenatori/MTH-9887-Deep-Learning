"""Download everything the replication needs from the WRDS data API.

Resumable: every piece is cached as a parquet file under data/raw/ and skipped
when present. Run from the replication/ folder:

    python scripts/download.py            # all stages
    python scripts/download.py msf names  # selected stages

Stages, in dependency order:
  ff      Fama-French daily factors and the risk-free rate (ff.factors_daily)
  dsi     CRSP daily market index returns (crsp.dsi)
  msf     CRSP monthly file, one request per month, all securities (crsp.msf)
  names   CRSP name history (share code, exchange code) for the candidates
  universe  the 500 largest common stocks per month; writes data/universe.parquet
  dsf     CRSP daily file, one request per security in the universe (crsp.dsf)
  delist  CRSP delisting returns for those securities (crsp.dsedelist)
  ccm     CRSP/Compustat link (crsp.ccmxpf_lnkhist)
  funda   Compustat annual fundamentals for the linked firms (comp.funda)

Licensed data: nothing under data/ may be committed or published.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attnfactors.wrds_api import Wrds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

FIRST_MONTH = "1985-01"   # monthly data: 36 months of history before 1990, plus slack
LAST_MONTH = "2021-12"
DSF_START = "1988-01-01"  # daily data: one year of history before the first lookback
DSF_END = "2021-12-31"
N_UNIVERSE = 500
N_CANDIDATES = 900        # top securities by market cap per month before the share-code filter
# choice: which CRSP securities count as "stocks". The paper names no filter ("securities on
# CRSP", "the 500 largest stocks"); its Figure 3 shows real-estate trusts and foreign firms among
# its 500 stocks. Admitted here: ordinary common shares (10, 11), common shares of firms
# incorporated outside the U.S. (12), real-estate investment trusts (18, 48) and American
# depositary receipts (30, 31). Left out: exchange-traded funds, closed-end funds, units.
SHARE_CODES = [10, 11, 12, 18, 48, 30, 31]
EXCHANGE_CODES = [1, 2, 3]
THREADS = 4

MSF_COLS = ["permno", "permco", "date", "ret", "retx", "prc", "altprc", "shrout", "vol", "hexcd", "hsiccd", "cfacpr", "cfacshr"]
DSF_COLS = ["permno", "date", "ret", "retx", "prc", "vol", "shrout", "bid", "ask", "bidlo", "askhi", "openprc", "numtrd", "cfacpr", "cfacshr"]
NAME_COLS = ["permno", "namedt", "nameendt", "shrcd", "exchcd", "siccd", "ticker", "comnam", "naics"]
DELIST_COLS = ["permno", "dlstdt", "dlstcd", "dlret", "dlretx", "dlprc"]
CCM_COLS = ["gvkey", "lpermno", "lpermco", "linkdt", "linkenddt", "linktype", "linkprim"]
FUNDA_COLS = [
    "gvkey", "datadate", "fyear", "fyr", "indfmt", "datafmt", "consol", "popsrc", "curcd",
    "at", "che", "ivao", "dlc", "dltt", "mib", "pstk", "pstkrv", "pstkl", "ceq", "seq", "txditc", "txdb",
    "itcb", "lt", "ni", "ib", "dp", "capx", "act", "lct", "txp", "revt", "sale", "cogs", "xsga", "xrd",
    "xad", "xint", "oiadp", "oibdp", "ppegt", "ppent", "invt", "wcap", "rect", "ap", "csho", "prcc_f",
]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def to_frame(rows: list[dict], cols: list[str], dates: list[str], ints: list[str] = (), strs: list[str] = ()) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=cols)
    for c in cols:
        if c in dates:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        elif c in strs:
            df[c] = df[c].astype("string")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if c in ints:
                df[c] = df[c].astype("Int64")
    return df


def months(first: str, last: str) -> list[pd.Period]:
    return list(pd.period_range(first, last, freq="M"))


# ---------------------------------------------------------------- stages

def stage_ff(w: Wrds) -> None:
    out = RAW / "ff_daily.parquet"
    if out.exists():
        return
    rows = w.rows("ff.factors_daily", {"date__gte": "1985-01-01", "date__lte": "2022-01-31"})
    df = to_frame(rows, ["date", "mktrf", "smb", "hml", "umd", "rf"], ["date"]).sort_values("date")
    df.to_parquet(out, index=False)
    log(f"ff: {len(df)} days {df.date.min().date()} .. {df.date.max().date()}")


def stage_dsi(w: Wrds) -> None:
    out = RAW / "dsi.parquet"
    if out.exists():
        return
    rows = w.rows("crsp.dsi", {"date__gte": "1985-01-01", "date__lte": "2022-01-31"})
    df = to_frame(rows, ["date", "vwretd", "vwretx", "ewretd", "ewretx", "sprtrn", "totcnt"], ["date"]).sort_values("date")
    df.to_parquet(out, index=False)
    log(f"dsi: {len(df)} days")


def stage_msf(w: Wrds) -> None:
    d = RAW / "msf"
    d.mkdir(parents=True, exist_ok=True)
    todo = [m for m in months(FIRST_MONTH, LAST_MONTH) if not (d / f"{m}.parquet").exists()]
    log(f"msf: {len(todo)} months to fetch")

    def one(m: pd.Period) -> int:
        rows = w.rows("crsp.msf", {"date__gte": str(m.start_time.date()), "date__lte": str(m.end_time.date())}, keep=MSF_COLS)
        df = to_frame(rows, MSF_COLS, ["date"], ints=["permno", "permco", "hexcd", "hsiccd"])
        df = df.drop_duplicates(["permno", "date"])
        df.to_parquet(d / f"{m}.parquet", index=False)
        return len(df)

    with ThreadPoolExecutor(THREADS) as ex:
        for i, n in enumerate(ex.map(one, todo)):
            if i % 24 == 0:
                log(f"msf: {todo[i]} -> {n} rows ({i + 1}/{len(todo)})")


def load_msf() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted((RAW / "msf").glob("*.parquet"))], ignore_index=True)


def candidates() -> pd.DataFrame:
    """Top N_CANDIDATES securities by month-end market capitalization, per month."""
    msf = load_msf()
    msf["me"] = msf["altprc"].abs().fillna(msf["prc"].abs()) * msf["shrout"]  # thousand USD
    msf["month"] = msf["date"].dt.to_period("M")
    msf = msf.dropna(subset=["me"])
    msf["rank"] = msf.groupby("month")["me"].rank(ascending=False, method="first")
    return msf.loc[msf["rank"] <= N_CANDIDATES, ["permno", "month", "date", "me", "rank"]]


def batched(xs: list, n: int) -> list[list]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def stage_names(w: Wrds) -> None:
    out = RAW / "msenames.parquet"
    if out.exists():
        return
    permnos = sorted(candidates()["permno"].unique().tolist())
    log(f"names: {len(permnos)} candidate securities")
    rows: list[dict] = []

    def one(batch: list[int]) -> list[dict]:
        return w.rows("crsp.msenames", {"permno__in": ",".join(map(str, batch))}, keep=NAME_COLS)

    with ThreadPoolExecutor(THREADS) as ex:
        for res in ex.map(one, batched(permnos, 150)):
            rows.extend(res)
    df = to_frame(rows, NAME_COLS, ["namedt", "nameendt"], ints=["permno", "shrcd", "exchcd", "siccd"], strs=["ticker", "comnam", "naics"])
    df = df.drop_duplicates(["permno", "namedt"]).sort_values(["permno", "namedt"])
    df.to_parquet(out, index=False)
    log(f"names: {len(df)} rows for {df.permno.nunique()} securities")


def stage_universe(w: Wrds) -> None:
    """Universe for month m: the N_UNIVERSE largest stocks (SHARE_CODES, EXCHANGE_CODES) by the
    CRSP market capitalization of the security at the end of month m-1."""
    out = ROOT / "data" / "universe.parquet"
    if out.exists():
        return
    cand = candidates()
    names = pd.read_parquet(RAW / "msenames.parquet")
    c = cand.merge(names[["permno", "namedt", "nameendt", "shrcd", "exchcd"]], on="permno", how="left")
    c = c[(c["date"] >= c["namedt"]) & (c["date"] <= c["nameendt"])]
    c = c[c["shrcd"].isin(SHARE_CODES) & c["exchcd"].isin(EXCHANGE_CODES)]
    # a security whose listing ends on the ranking date itself has no return in month m
    last_name = names.groupby("permno")["nameendt"].max()
    c = c[c["date"] < c["permno"].map(last_name)]
    c = c.drop_duplicates(["permno", "month"])
    c["rank"] = c.groupby("month")["me"].rank(ascending=False, method="first")
    u = c.loc[c["rank"] <= N_UNIVERSE, ["permno", "month", "me", "rank"]].copy()
    u["month"] = (u["month"] + 1).astype(str)  # ranked at end of m-1, used in month m
    u = u.rename(columns={"me": "me_prev"}).sort_values(["month", "rank"])
    per_month = u.groupby("month").size()
    log(f"universe: {u.permno.nunique()} securities over {len(per_month)} months; per month min {per_month.min()} max {per_month.max()}")
    u.to_parquet(out, index=False)


def universe_permnos() -> list[int]:
    """Securities that are in the universe in any decision month covered by the daily data."""
    u = pd.read_parquet(ROOT / "data" / "universe.parquet")
    return sorted(u.loc[u["month"] >= DSF_START[:7], "permno"].unique().tolist())


def stage_dsf(w: Wrds) -> None:
    d = RAW / "dsf"
    d.mkdir(parents=True, exist_ok=True)
    todo = [p for p in universe_permnos() if not (d / f"{p}.parquet").exists()]
    log(f"dsf: {len(todo)} securities to fetch")

    def one(permno: int) -> int:
        rows = w.rows("crsp.dsf", {"permno": permno, "date__gte": DSF_START, "date__lte": DSF_END}, keep=DSF_COLS)
        df = to_frame(rows, DSF_COLS, ["date"], ints=["permno"])
        df = df.drop_duplicates(["permno", "date"]).sort_values("date")
        df.to_parquet(d / f"{permno}.parquet", index=False)
        return len(df)

    t0 = time.time()
    total = 0
    with ThreadPoolExecutor(THREADS) as ex:
        for i, n in enumerate(ex.map(one, todo)):
            total += n
            if i % 100 == 0:
                log(f"dsf: {i + 1}/{len(todo)} securities, {total} rows, {time.time() - t0:.0f}s")
    log(f"dsf: done, {total} new rows")


def stage_delist(w: Wrds) -> None:
    out = RAW / "dsedelist.parquet"
    if out.exists():
        return
    rows: list[dict] = []
    for batch in batched(universe_permnos(), 150):
        rows.extend(w.rows("crsp.dsedelist", {"permno__in": ",".join(map(str, batch))}, keep=DELIST_COLS))
    df = to_frame(rows, DELIST_COLS, ["dlstdt"], ints=["permno", "dlstcd"]).drop_duplicates(["permno", "dlstdt"])
    df.to_parquet(out, index=False)
    log(f"delist: {len(df)} rows")


def stage_ccm(w: Wrds) -> None:
    out = RAW / "ccm.parquet"
    if out.exists():
        return
    rows: list[dict] = []
    for batch in batched(universe_permnos(), 150):
        rows.extend(w.rows("crsp.ccmxpf_lnkhist", {"lpermno__in": ",".join(map(str, batch))}, keep=CCM_COLS))
    df = to_frame(rows, CCM_COLS, ["linkdt", "linkenddt"], ints=["lpermno", "lpermco"], strs=["gvkey", "linktype", "linkprim"])
    df = df.drop_duplicates().sort_values(["lpermno", "linkdt"])
    df.to_parquet(out, index=False)
    log(f"ccm: {len(df)} links, {df.gvkey.nunique()} firms")


def stage_funda(w: Wrds) -> None:
    d = RAW / "funda"
    d.mkdir(parents=True, exist_ok=True)
    ccm = pd.read_parquet(RAW / "ccm.parquet")
    ccm = ccm[ccm["linktype"].isin(["LU", "LC"]) & ccm["linkprim"].isin(["P", "C"])]
    gvkeys = sorted(ccm["gvkey"].dropna().unique().tolist())
    batches = batched(gvkeys, 12)
    todo = [(i, b) for i, b in enumerate(batches) if not (d / f"batch_{i:04d}.parquet").exists()]
    log(f"funda: {len(gvkeys)} firms, {len(todo)} of {len(batches)} batches to fetch")

    def one(job: tuple[int, list[str]]) -> int:
        i, b = job
        rows = w.rows(
            "comp.funda",
            {"gvkey__in": ",".join(b), "indfmt": "INDL", "datafmt": "STD", "consol": "C", "popsrc": "D",
             "datadate__gte": "1983-01-01", "datadate__lte": "2021-12-31"},
            keep=FUNDA_COLS, limit=2000,
        )
        df = to_frame(rows, FUNDA_COLS, ["datadate"], ints=["fyear", "fyr"], strs=["gvkey", "indfmt", "datafmt", "consol", "popsrc", "curcd"])
        # the client has applied the filters locally (the server ignores some of them); make sure
        df = df[(df["indfmt"] == "INDL") & (df["datafmt"] == "STD") & (df["consol"] == "C") & (df["popsrc"] == "D")]
        if df.duplicated(["gvkey", "datadate"]).any():
            raise RuntimeError(f"funda batch {i}: more than one standard record for a firm and date")
        df = df.sort_values(["gvkey", "datadate"])
        df.to_parquet(d / f"batch_{i:04d}.parquet", index=False)
        return len(df)

    t0 = time.time()
    total = 0
    with ThreadPoolExecutor(3) as ex:
        for k, n in enumerate(ex.map(one, todo)):
            total += n
            if k % 20 == 0:
                log(f"funda: {k + 1}/{len(todo)} batches, {total} rows, {time.time() - t0:.0f}s")
    log(f"funda: done, {total} new rows; rows sent by the server that did not match the filters: {w.dropped}")


STAGES = {
    "ff": stage_ff, "dsi": stage_dsi, "msf": stage_msf, "names": stage_names, "universe": stage_universe,
    "dsf": stage_dsf, "delist": stage_delist, "ccm": stage_ccm, "funda": stage_funda,
}


def main(argv: list[str]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    w = Wrds()
    for name in (argv or list(STAGES)):
        log(f"== stage {name}")
        STAGES[name](w)
    log("all requested stages done")


if __name__ == "__main__":
    main(sys.argv[1:])
