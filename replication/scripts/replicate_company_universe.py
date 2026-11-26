"""Fourth round of runs: a universe built to match the firms that the paper's figures name.

The paper says "the 500 largest stocks based on the previous month market capitalization". Its
Figures 3 and 4 name 108 firms among its 500 on the first trading day of 2021, and the universe
of rounds 1 to 3 (the 500 largest securities by CRSP capitalization) holds only 68 of them. This
round changes the rule in three ways that those 108 names support (README_ISSUES.md, issue 5,
and DIFFERENCES.md):

  value     a firm counts with the value of the whole company: price times the company's shares
            in the Compustat monthly security file (company_values), or the sum of its
            securities in CRSP where that is larger or Compustat has none. A depositary receipt
            of a foreign firm then ranks with the firm, not with the receipts outstanding. The
            firm's value stands on its largest security; another share class of the same firm
            ranks with its own CRSP value (Alphabet's two classes are both members; a class
            that hardly trades is not).
  kinds     partnership units (share codes 71, 72) are admitted beside ordinary shares, trusts
            and depositary receipts.
  history   a firm needs a security with a return in each of the last 120 months. No firm that
            the paper names is younger than 10.4 years at the end of 2020.

One rule on the data is new as well: a day on which a security did not trade has no return,
and neither has the day after it (no_trade_days). The universe now holds receipts with a small
float, and their quotes on days without a trade are sometimes far from any price.

Everything else is round 3 ("as cited"): the loss, the inputs and the two benchmarks. Because
the paper leaves two things open, the runs cover both readings of each: the Sharpe ratio in the
objective per day and annualized, and PCA residuals with loadings from 60 days and by projection.

Run from replication/ (a WRDS token is needed for the first stage only):

    python scripts/replicate_company_universe.py download   # Compustat values, names, links, early monthly file
    python scripts/replicate_company_universe.py universe   # data/company/universe.parquet and its checks
    python scripts/replicate_company_universe.py data       # daily data, delistings, fundamentals of the new members
    python scripts/replicate_company_universe.py build      # panel, inputs, PCA residuals under data/company/
    python scripts/replicate_company_universe.py main       # the runs, 1998 to 2021
    python scripts/replicate_company_universe.py delisting  # the forced delistings of January 2021 ('main' runs it too)
    python scripts/replicate_company_universe.py report     # results/company/comparison.md from saved files

Full output goes to runs/company/ and data/company/ (not committed), the two small files of every
run to results/company/ (committed). Nothing of the earlier rounds is overwritten: the daily files
of new securities are added to data/raw/dsf, which the earlier panels do not read.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
DATA_ROOT = "data/company"                 # ATTNFACTORS_DATA of this round
C = ROOT / DATA_ROOT
RAW, OLD = C / "raw", ROOT / "data" / "raw"
PUBLIC = ROOT / "results" / "company"
PRIVATE = "company"                        # under runs/
ENV = {**os.environ, "ATTNFACTORS_DATA": DATA_ROOT}

SHARE_CODES = [10, 11, 12, 18, 48, 30, 31, 71, 72]
EXCHANGE_CODES = [1, 2, 3]
N_UNIVERSE = 500
HISTORY = 120                              # months with a return that a security of the firm must have
FIRST_RANK, LAST_RANK = "1987-12", "2021-11"   # month-ends at which the ranking is made
SECM = ["gvkey", "iid", "datadate", "tic", "cusip", "conm", "tpci", "prccm", "cshoq", "cshom", "adrrm", "ajexm",
        "primiss", "curcdm", "exchg", "fic", "secstat"]
SQRT252 = 15.874507866387544


def case(kind, k, seed=0, pca="regression", sr_scale=1.0):
    return dict(kind=kind, k=k, seed=seed, batch=125, ev_floor=None, sr_scale=sr_scale, pca=pca if kind != "attention" else None)


MAIN = {
    # two-step benchmarks first: they are short
    "pca_K30_seed0": case("pca", 30), "pca_K8_seed0": case("pca", 8),
    "pcaproj_K30_seed0": case("pca", 30, pca="projection"), "pcaproj_K8_seed0": case("pca", 8, pca="projection"),
    "ou_K30": case("ou", 30), "ou_K8": case("ou", 8),
    "ouproj_K30": case("ou", 30, pca="projection"), "ouproj_K8": case("ou", 8, pca="projection"),
    # the Attention Factor model: the objective as printed, and with the annualized Sharpe ratio
    "attention_K30_seed0": case("attention", 30), "attention_K8_seed0": case("attention", 8),
    "attention_annualized_K30_seed0": case("attention", 30, sr_scale=SQRT252),
    "attention_annualized_K8_seed0": case("attention", 8, sr_scale=SQRT252),
    "attention_K30_seed1": case("attention", 30, seed=1),
}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- download

def download() -> None:
    from attnfactors.wrds_api import Wrds
    from download import CCM_COLS, MSF_COLS, NAME_COLS, to_frame
    w = Wrds()
    (RAW / "secm").mkdir(parents=True, exist_ok=True)
    (RAW / "msf_early").mkdir(parents=True, exist_ok=True)
    if not (RAW / "msenames.parquet").exists():
        df = to_frame(w.rows("crsp.msenames", {}, keep=NAME_COLS), NAME_COLS, ["namedt", "nameendt"],
                      ints=["permno", "shrcd", "exchcd", "siccd"], strs=["ticker", "comnam", "naics"])
        df.drop_duplicates(["permno", "namedt"]).sort_values(["permno", "namedt"]).to_parquet(RAW / "msenames.parquet", index=False)
        log(f"names: {len(df)} rows")
    if not (RAW / "ccm.parquet").exists():
        df = to_frame(w.rows("crsp.ccmxpf_lnkhist", {}, keep=CCM_COLS), CCM_COLS, ["linkdt", "linkenddt"],
                      ints=["lpermno", "lpermco"], strs=["gvkey", "linktype", "linkprim"])
        df.drop_duplicates().sort_values(["lpermno", "linkdt"]).to_parquet(RAW / "ccm.parquet", index=False)
        log(f"ccm: {len(df)} links")

    def secm(m: pd.Period) -> None:
        rows = w.rows("comp.secm", {"datadate__gte": str(m.start_time.date()), "datadate__lte": str(m.end_time.date())}, keep=SECM)
        to_frame(rows, SECM, ["datadate"], ints=["exchg"],
                 strs=["gvkey", "iid", "tic", "cusip", "conm", "tpci", "primiss", "curcdm", "fic", "secstat"]
                 ).to_parquet(RAW / "secm" / f"{m}.parquet", index=False)

    def msf(m: pd.Period) -> None:
        rows = w.rows("crsp.msf", {"date__gte": str(m.start_time.date()), "date__lte": str(m.end_time.date())}, keep=MSF_COLS)
        to_frame(rows, MSF_COLS, ["date"], ints=["permno", "permco", "hexcd", "hsiccd"]
                 ).drop_duplicates(["permno", "date"]).to_parquet(RAW / "msf_early" / f"{m}.parquet", index=False)

    for name, fn, months in (("secm", secm, pd.period_range("1984-12", "2021-12", freq="M")),
                             ("msf_early", msf, pd.period_range("1975-01", "1984-12", freq="M"))):
        todo = [m for m in months if not (RAW / name / f"{m}.parquet").exists()]
        log(f"{name}: {len(todo)} months to fetch")
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(fn, todo))
    for name in ("ff_daily.parquet", "dsi.parquet", "msf", "dsf"):     # shared with the earlier rounds
        if not (RAW / name).exists():
            (RAW / name).symlink_to(os.path.relpath(OLD / name, RAW))
    log("download: done")


# ---------------------------------------------------------------- universe

def company_values() -> pd.DataFrame:
    """Value of the whole company, by gvkey and month, in thousand USD (the unit of CRSP's capitalization).

    Price of the month times the company's shares, from the Compustat monthly security file. The
    company's shares (``cshoq``) stand on the firm's primary issue and only in the months that end a
    fiscal quarter. They are put in split-adjusted units (times ``ajexm``), the median of the last three
    quarters is taken (a share count sometimes shows a split one quarter before the price does), and
    the result is carried forward for up to 12 months and put back in the units of the month's price.
    The largest value over the firm's issues quoted in USD is taken. Nothing after the month is used.

    A firm that never reports quarterly shares to Compustat (some foreign firms) has no value here and
    is ranked by its CRSP value. The file's monthly shares of an issue (``cshom``) are not used: they
    exist from 1998 only, and for a depositary receipt they need a ratio that the file lacks before 2002.
    """
    parts = []
    for f in sorted((RAW / "secm").glob("*.parquet")):
        s = pd.read_parquet(f, columns=["gvkey", "iid", "prccm", "cshoq", "ajexm", "curcdm"])
        s = s[(s["curcdm"] == "USD") & s["prccm"].notna() & (s["ajexm"] > 0)]
        parts.append(s[["gvkey", "iid", "prccm", "cshoq", "ajexm"]].assign(month=f.stem))
    return company_value_table(pd.concat(parts, ignore_index=True))


def company_value_table(s: pd.DataFrame) -> pd.DataFrame:
    """s: one row per issue and month with gvkey, iid, month, prccm, cshoq (missing outside the quarter
    ends) and ajexm. Returns gvkey, month, company (thousand USD). See company_values."""
    s = s.drop_duplicates(["gvkey", "iid", "month"]).sort_values(["gvkey", "iid", "month"]).reset_index(drop=True)
    s["adj"] = (s["cshoq"] * s["ajexm"]).where(s["cshoq"] > 0)
    q = s.dropna(subset=["adj"])
    s["shares"] = q.groupby(["gvkey", "iid"])["adj"].rolling(3, min_periods=1).median().reset_index(level=[0, 1], drop=True)
    s["shares"] = s.groupby(["gvkey", "iid"])["shares"].ffill(limit=12)
    s["company"] = s["prccm"].abs() * s["shares"] / s["ajexm"] * 1000.0
    return s.dropna(subset=["company"]).groupby(["gvkey", "month"], as_index=False)["company"].max()


def ranking_value(me: pd.Series, firm_value: pd.Series, permco: pd.Series, month: pd.Series) -> np.ndarray:
    """The firm's value for the firm's largest security of the month, its own CRSP value for any other."""
    largest = me == me.groupby([permco, month]).transform("max")
    return np.where(largest, firm_value, me)


def monthly_file() -> pd.DataFrame:
    cols = ["permno", "permco", "date", "ret", "prc", "altprc", "shrout"]
    files = sorted((RAW / "msf_early").glob("*.parquet")) + sorted((RAW / "msf").glob("*.parquet"))
    m = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    m["month"] = m["date"].dt.to_period("M").astype(str)
    m["me"] = m["altprc"].abs().fillna(m["prc"].abs()) * m["shrout"]
    return m


def consecutive_returns(permno: np.ndarray, month: np.ndarray, has_ret: np.ndarray) -> np.ndarray:
    """For rows sorted by security and month (month as a running number): the number of months, ending
    with the row, in which the security has a return without a gap. A month that is missing in the
    file, or a month without a return, starts the count again."""
    new = np.r_[True, (permno[1:] != permno[:-1]) | (month[1:] != month[:-1] + 1)]
    run = np.zeros(len(permno), dtype=np.int32)
    r = 0
    for i in range(len(permno)):
        r = 0 if new[i] else r
        r = r + 1 if has_ret[i] else 0
        run[i] = r
    return run


def rank_all(history: int = HISTORY, codes: list[int] = SHARE_CODES, company: bool = True) -> pd.DataFrame:
    """Every eligible security at every ranking month, with its value and its rank among those whose
    firm has the return history. Columns: permno, permco, month, me (CRSP, of the security), value, ok, rank."""
    m = monthly_file()
    months = sorted(m["month"].unique())
    # consecutive months with a return, per security
    m = m.sort_values(["permno", "month"])
    idx = m["month"].map({mm: i for i, mm in enumerate(months)}).to_numpy()
    m["run"] = consecutive_returns(m["permno"].to_numpy(), idx, m["ret"].notna().to_numpy())
    m = m[(m["month"] >= FIRST_RANK) & (m["month"] <= LAST_RANK) & m["me"].notna()]
    names = pd.read_parquet(RAW / "msenames.parquet")
    last_name = names.groupby("permno")["nameendt"].max()
    x = m.merge(names[["permno", "namedt", "nameendt", "shrcd", "exchcd"]], on="permno", how="left")
    x = x[(x["date"] >= x["namedt"]) & (x["date"] <= x["nameendt"])]
    x = x[x["shrcd"].isin(codes) & x["exchcd"].isin(EXCHANGE_CODES)]
    x = x[x["date"] < x["permno"].map(last_name)]        # a listing that ends on the ranking date has no next month
    x = x.drop_duplicates(["permno", "month"])
    x["me_firm"] = x.groupby(["permco", "month"])["me"].transform("sum")
    x["value"] = x["firm_value"] = x["me_firm"]
    if company:
        ccm = pd.read_parquet(RAW / "ccm.parquet")
        ccm = ccm[ccm["linktype"].isin(["LC", "LU"])].copy()        # the links that chars.py uses for the fundamentals
        ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
        ccm["prim"] = ccm["linkprim"].map({"P": 0, "C": 1, "J": 2, "N": 3}).fillna(4)
        y = x[["permno", "month", "date"]].merge(ccm[["lpermno", "gvkey", "linkdt", "linkenddt", "prim"]],
                                                 left_on="permno", right_on="lpermno", how="inner")
        y = y[(y["date"] >= y["linkdt"]) & (y["date"] <= y["linkenddt"])].sort_values("prim").drop_duplicates(["permno", "month"])
        x = x.merge(y[["permno", "month", "gvkey"]], on=["permno", "month"], how="left")
        x = x.merge(company_values(), on=["gvkey", "month"], how="left")
        x["firm_value"] = np.fmax(x["me_firm"], x["company"].astype(float))     # never below the firm's value in CRSP
        # the firm's value stands on its largest security; another class of the firm ranks with its own value
        x["value"] = ranking_value(x["me"], x["firm_value"], x["permco"], x["month"])
    x["ok"] = x.groupby(["permco", "month"])["run"].transform("max") >= history
    x["rank"] = x[x["ok"]].groupby("month")["value"].rank(ascending=False, method="first")
    return x[["permno", "permco", "month", "me", "me_firm", "value", "firm_value", "ok", "rank", "shrcd"]]


def named_firms() -> dict[str, set[int]]:
    """The 108 firms of Figures 3 and 4: for each, the securities of the firm that carries the name."""
    from check_issues import FIGURE_3, FIGURE_4, _norm
    names = pd.read_parquet(RAW / "msenames.parquet")
    cur = names[(names["nameendt"] >= "2021-01-01") & (names["namedt"] <= "2021-12-31") & names["shrcd"].isin(SHARE_CODES)]
    m = pd.read_parquet(OLD / "msf" / "2020-12.parquet", columns=["permno", "permco", "prc", "shrout"])
    m["me"] = m["prc"].abs() * m["shrout"]
    firm = m.groupby("permco")["me"].sum()
    out = {}
    for block in (FIGURE_3, FIGURE_4):
        for item in block.replace("\n", "").split("|"):
            keys = [_norm(k) for k in item.split(";")]
            hit = cur[cur["comnam"].fillna("").map(lambda c: any(_norm(c).startswith(k) for k in keys))]
            cand = m[m["permno"].isin(hit["permno"])]
            if cand.empty:
                out[item.split(";")[0]] = set()
                continue
            best = cand.assign(f=cand["permco"].map(firm)).sort_values("f")["permco"].iloc[-1]   # the largest firm of that name
            out[item.split(";")[0]] = set(m.loc[m["permco"] == best, "permno"])
    return out


_RET: list[pd.DataFrame] = []


def market_row(u: pd.DataFrame) -> tuple[float, float]:
    """Mean and volatility in percent a year of the equally weighted universe, 1998 to 2021, from monthly returns."""
    if not _RET:
        m = pd.concat([pd.read_parquet(f, columns=["permno", "date", "ret"]) for f in sorted((RAW / "msf").glob("*.parquet"))])
        m["month"] = m["date"].dt.to_period("M").astype(str)
        _RET.append(m[["permno", "month", "ret"]])
    r = u.merge(_RET[0], on=["permno", "month"], how="left")
    ew = r[(r["month"] >= "1998-01") & (r["month"] <= "2021-12")].groupby("month")["ret"].mean()
    return float(ew.mean() * 1200), float(ew.std(ddof=1) * np.sqrt(12) * 100)


def to_universe(x: pd.DataFrame) -> pd.DataFrame:
    u = x.loc[x["rank"] <= N_UNIVERSE, [c for c in ("permno", "permco", "month", "value", "rank") if c in x]].copy()
    u["month"] = (pd.PeriodIndex(u["month"], freq="M") + 1).astype(str)      # ranked at the end of m-1, used in month m
    return u.rename(columns={"value": "me_prev"}).sort_values(["month", "rank"])


def universe() -> None:
    """Write the universe of this round and the checks that compare rules."""
    firms = named_firms()
    rows = []
    rules = [("Rounds 1 to 3: CRSP value of the security, no units, no history", dict(history=0, codes=[10, 11, 12, 18, 48, 30, 31], company=False), "me"),
             ("Company value", dict(history=0, codes=[10, 11, 12, 18, 48, 30, 31], company=True), "value"),
             ("Company value, units admitted", dict(history=0, company=True), "value"),
             ("Company value, units, history of 96 months", dict(history=96, company=True), "value"),
             ("This round: company value, units, history of 120 months", dict(history=HISTORY, company=True), "value"),
             ("Company value, units, history of 144 months", dict(history=144, company=True), "value")]
    final = None
    for label, kw, col in rules:
        x = rank_all(**kw)
        if col == "me":                         # the old rule ranks securities by their own CRSP value
            x["value"] = x["me"]
            x["rank"] = x.groupby("month")["value"].rank(ascending=False, method="first")
        u = to_universe(x)
        if col == "me":                         # this builder must give back the universe of rounds 1 to 3
            old = pd.read_parquet(ROOT / "data" / "universe.parquet")
            old = old[old["month"].isin(u["month"].unique())]
            same = len(old.merge(u, on=["permno", "month"])) / len(old)
            log(f"the old rule in this builder gives {100 * same:.2f} percent of the member-months of data/universe.parquet")
            assert same > 0.999
        jan = set(u.loc[u["month"] == "2021-01", "permno"])
        inside = sum(bool(v & jan) for v in firms.values())
        mean, vol = market_row(u)
        cut = u.loc[u["month"] == "2021-01", "me_prev"].min() / 1e6
        kinds = x[(x["month"] == "2020-12") & (x["rank"] <= N_UNIVERSE)]["shrcd"]
        rows.append((label, inside, cut, int(kinds.isin([30, 31]).sum()), int(kinds.isin([71, 72]).sum()), mean, vol,
                     u["permno"].nunique()))
        log(f"{label}: {inside} of {len(firms)} named firms, market row {mean:.2f} / {vol:.2f}")
        if kw.get("history") == HISTORY and kw.get("company"):
            final = (x, u, [k for k, v in firms.items() if not (v & jan)])
    x, u, missing = final
    per_month = u.groupby("month").size()
    assert per_month.min() == N_UNIVERSE == per_month.max(), "a month without 500 members"
    u.to_parquet(C / "universe.parquet", index=False)
    x.loc[x["month"] >= FIRST_RANK, ["permno", "month", "me", "value", "firm_value"]].to_parquet(C / "company_value.parquet", index=False)
    old = pd.read_parquet(ROOT / "data" / "universe.parquet")
    old = old[old["month"].isin(u["month"].unique())]
    both = old.merge(u, on=["permno", "month"])
    out = ["## Which rule holds the firms that the paper names?", "",
           f"Figures 3 and 4 of the paper name {len(firms)} firms among its 500 on the first trading day of 2021. A rule is",
           "scored by how many of them it puts among its 500 for January 2021, and by the mean and the volatility of its",
           "equally weighted universe over 1998 to 2021, computed here from monthly returns (the paper's market row,",
           "from daily returns: 8.61 and 20.37 percent a year).", "",
           "| Rule | Named firms among the 500 | Value of the 500th, billion USD | Depositary receipts | Partnership units | Mean % | Volatility % | Securities ever in the universe |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r[0]} | {r[1]} of {len(firms)} | {r[2]:.1f} | {r[3]} | {r[4]} | {r[5]:.2f} | {r[6]:.2f} | {r[7]:,} |")
    out += ["", f"Named firms that this round's universe still does not hold in January 2021 ({len(missing)}): " + ", ".join(sorted(missing)) + ".", "",
            f"Overlap with the universe of rounds 1 to 3: {100 * len(both) / len(u):.0f} percent of the member-months "
            f"({u['month'].min()} to {u['month'].max()}); in January 2021, "
            f"{len(set(u.loc[u['month'] == '2021-01', 'permno']) & set(old.loc[old['month'] == '2021-01', 'permno']))} of 500 securities are in both."]
    (PUBLIC / "checks").mkdir(parents=True, exist_ok=True)
    (PUBLIC / "checks" / "universe_rules.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


# ---------------------------------------------------------------- data of the new members

def data() -> None:
    from attnfactors.wrds_api import Wrds
    from download import CCM_COLS, DELIST_COLS, DSF_COLS, DSF_END, DSF_START, FUNDA_COLS, batched, to_frame
    w = Wrds()
    u = pd.read_parquet(C / "universe.parquet")
    permnos = sorted(u.loc[u["month"] >= DSF_START[:7], "permno"].unique().tolist())
    todo = [p for p in permnos if not (OLD / "dsf" / f"{p}.parquet").exists()]
    log(f"dsf: {len(permnos)} securities in the universe, {len(todo)} to fetch")

    def one(permno: int) -> int:
        rows = w.rows("crsp.dsf", {"permno": permno, "date__gte": DSF_START, "date__lte": DSF_END}, keep=DSF_COLS)
        df = to_frame(rows, DSF_COLS, ["date"], ints=["permno"]).drop_duplicates(["permno", "date"]).sort_values("date")
        df.to_parquet(OLD / "dsf" / f"{permno}.parquet", index=False)
        return len(df)

    with ThreadPoolExecutor(4) as ex:
        for i, _ in enumerate(ex.map(one, todo)):
            if i % 100 == 0:
                log(f"dsf: {i + 1}/{len(todo)}")
    # delisting returns: every security of the universe is asked once (the list of those asked is kept)
    asked_file = RAW / "dsedelist_asked.json"
    done = set(json.loads(asked_file.read_text())) if asked_file.exists() and (RAW / "dsedelist.parquet").exists() else set()
    need = [q for q in permnos if q not in done]
    if need:
        rows: list[dict] = []
        for b in batched(need, 150):
            rows.extend(w.rows("crsp.dsedelist", {"permno__in": ",".join(map(str, b))}, keep=DELIST_COLS))
        df = to_frame(rows, DELIST_COLS, ["dlstdt"], ints=["permno", "dlstcd"])
        if done:
            df = pd.concat([pd.read_parquet(RAW / "dsedelist.parquet"), df], ignore_index=True)
        df.drop_duplicates(["permno", "dlstdt"]).to_parquet(RAW / "dsedelist.parquet", index=False)
        asked_file.write_text(json.dumps(sorted(done | set(need))))
        log(f"delist: asked for {len(need)} securities, {len(df)} rows in the file")
    # fundamentals: the batches of the earlier rounds are linked; the firms they lack are fetched
    d = RAW / "funda"
    d.mkdir(exist_ok=True)
    for f in sorted((OLD / "funda").glob("*.parquet")):
        if not (d / f"old_{f.name}").exists():
            (d / f"old_{f.name}").symlink_to(os.path.relpath(f, d))
    have = set(pd.concat([pd.read_parquet(f, columns=["gvkey"]) for f in d.glob("*.parquet")])["gvkey"])
    asked = set(json.loads((d / "asked.json").read_text())) if (d / "asked.json").exists() else set()
    ccm = pd.read_parquet(RAW / "ccm.parquet")
    ccm = ccm[ccm["lpermno"].isin(permnos) & ccm["linktype"].isin(["LU", "LC"])]
    gvkeys = sorted(set(ccm["gvkey"].dropna()) - have - asked)
    log(f"funda: {len(gvkeys)} firms to fetch")

    def fa(job: tuple[int, list[str]]) -> int:
        i, b = job
        rows = w.rows("comp.funda", {"gvkey__in": ",".join(b), "indfmt": "INDL", "datafmt": "STD", "consol": "C", "popsrc": "D",
                                     "datadate__gte": "1983-01-01", "datadate__lte": "2021-12-31"}, keep=FUNDA_COLS, limit=2000)
        df = to_frame(rows, FUNDA_COLS, ["datadate"], ints=["fyear", "fyr"], strs=["gvkey", "indfmt", "datafmt", "consol", "popsrc", "curcd"])
        df = df[(df["indfmt"] == "INDL") & (df["datafmt"] == "STD") & (df["consol"] == "C") & (df["popsrc"] == "D")]
        df = df[df["datadate"].between("1983-01-01", "2021-12-31")]
        if df.duplicated(["gvkey", "datadate"]).any():
            raise RuntimeError(f"funda batch {i}: more than one standard record for a firm and date")
        df.sort_values(["gvkey", "datadate"]).to_parquet(d / f"new_{b[0]}.parquet", index=False)
        return len(df)

    with ThreadPoolExecutor(3) as ex:
        n = sum(ex.map(fa, list(enumerate(batched(gvkeys, 12)))))
    (d / "asked.json").write_text(json.dumps(sorted(asked | set(gvkeys))))
    log(f"funda: {n} new rows")


# ---------------------------------------------------------------- build

def no_trade_days(panel: Path) -> None:
    """A day without a trade has no return, and neither has the day after it.

    A day without a trade is a day with a volume of zero; its price in CRSP is the middle of the
    closing bid and ask. For the receipts with a small float that this universe holds, such a quote
    can be far from any price (one receipt shows +329 percent and -79 percent on two days without a
    trade). The return of the next day is measured from that quote and goes too. A delisting return,
    which the panel books on a day without a price, stays. A real move over days without a trade is
    lost with the rule: among members, 73 two-day moves beyond 10 percent, none beyond 30."""
    prc, vol = np.load(panel / "daily_prc.npy"), np.load(panel / "daily_vol.npy")
    ret = np.load(panel / "daily_ret.npy")
    quiet = vol == 0
    after = np.zeros_like(quiet)
    after[1:] = quiet[:-1] & np.isfinite(prc[1:])
    drop = (quiet | after) & np.isfinite(ret)
    member = np.load(panel / "member.npy")
    ret[drop] = np.nan
    np.save(panel / "daily_ret.npy", ret)
    log(f"panel: {int(drop.sum()):,} returns on or after a day without a trade set to missing "
        f"({int((drop & member).sum()):,} of them of members, {100 * (drop & member).sum() / member.sum():.2f} percent of the member-days)")



def build() -> None:
    """Panel with the characteristics as cited, the inputs, and both sets of PCA residuals."""
    def py(code: str) -> None:
        subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'src'); " + code], cwd=ROOT, check=True, env=ENV)

    py("from attnfactors import panel; panel.build()")
    # the firm's value in the panel is the company value that ranked it: LME and the ratios to market equity use it
    from attnfactors.panel import DATA
    assert DATA == ROOT / "data", "build() itself runs with the default data root"
    panel = C / "panel"
    meta = np.load(panel / "meta.npz", allow_pickle=False)
    cv = pd.read_parquet(C / "company_value.parquet")
    pidx = {int(p): i for i, p in enumerate(meta["permnos"])}
    midx = {str(m): i for i, m in enumerate(meta["months"])}
    cv = cv[cv["permno"].isin(pidx) & cv["month"].isin(midx)]
    r, c = cv["month"].map(midx).to_numpy(), cv["permno"].map(pidx).to_numpy()
    # monthly_me: the value that ranks the security (the size characteristic); monthly_me_firm: the value of
    # its firm (the denominator of the ratios to market equity)
    for name, col in (("monthly_me.npy", "value"), ("monthly_me_firm.npy", "firm_value")):
        a = np.load(panel / name)
        a[r, c] = cv[col].to_numpy(dtype=np.float32)
        np.save(panel / name, a)
    log(f"panel: company value written for {len(cv):,} security-months")
    no_trade_days(panel)
    py("from attnfactors import chars, features; chars.build('cited'); features.build_inputs(); features.build_pca()")
    py("from attnfactors import pca_regression; pca_regression.build()")
    subprocess.run([sys.executable, "scripts/build.py", "check"], cwd=ROOT, check=True, env=ENV)
    market_check()


def market_check() -> None:
    """The paper's "Market" row beside the portfolios of both universes and four indices, from daily returns."""
    def stats(x: np.ndarray) -> str:
        mu, sd = x.mean() * 25200, x.std(ddof=1) * np.sqrt(252) * 100
        return f"{mu:.2f} | {sd:.2f} | {mu / sd:.2f}"

    out = ["## The market row", "",
           "Out of sample, 1998 to 2021, from daily returns, percent a year. \"Excess\": minus the risk-free rate.",
           "The note of the paper's Table 2 calls its row \"the equally weighted market portfolio\".", "",
           "| Portfolio | Mean | Volatility | Mean over volatility |", "|---|---|---|---|", "| **Paper, Table 2, \"Market\"** | **8.61** | **20.37** | **0.42** |"]
    for label, root in (("Universe of this round", C), ("Universe of rounds 1 to 3", ROOT / "data")):
        meta = np.load(root / "panel" / "meta.npz", allow_pickle=False)
        ret = np.load(root / "panel" / "daily_ret.npy")
        member = np.load(root / "panel" / "member.npy")
        rf = np.nan_to_num(np.load(root / "panel" / "ff.npy")[:, 3])
        years = meta["dates"].astype("datetime64[Y]").astype(int) + 1970
        u = pd.read_parquet(root / "universe.parquet")
        pidx = {int(q): i for i, q in enumerate(meta["permnos"])}
        w = np.zeros(ret.shape, dtype=np.float64)
        u = u[u["permno"].isin(pidx)]
        cols = u["permno"].map(pidx).to_numpy()
        for m, g in u.assign(c=cols).groupby("month"):
            rows = np.where(meta["month_of_ret"] == m)[0]
            if len(rows):
                # the ranking value: the firm's value on its largest security, a second class on its own
                w[np.ix_(rows, g["c"].to_numpy())] = g["me_prev"].to_numpy()
        nxt, has = np.nan_to_num(ret[1:]), np.isfinite(ret[1:])
        mem = member[:-1] & has
        oos = (years[1:] >= 1998) & (years[1:] <= 2021)
        ew = ((nxt * mem).sum(1) / np.maximum(mem.sum(1), 1))[oos]
        ww = w[:-1] * mem
        vw = ((nxt * ww).sum(1) / np.maximum(ww.sum(1), 1e-12))[oos]
        r = rf[1:][oos]
        out += [f"| {label}, equally weighted | {stats(ew)} |", f"| ... excess | {stats(ew - r)} |",
                f"| {label}, weighted by the ranking value | {stats(vw)} |", f"| ... excess | {stats(vw - r)} |"]
    d = pd.read_parquet(OLD / "dsi.parquet")
    d = d[(d["date"] >= "1998-01-01") & (d["date"] <= "2021-12-31")]
    ff = pd.read_parquet(OLD / "ff_daily.parquet")
    ff = ff[(ff["date"] >= "1998-01-01") & (ff["date"] <= "2021-12-31")]
    out += [f"| S&P 500 index, without dividends | {stats(d['sprtrn'].to_numpy(dtype=float))} |",
            f"| CRSP value-weighted index, with dividends | {stats(d['vwretd'].to_numpy(dtype=float))} |",
            f"| CRSP equally weighted index, with dividends | {stats(d['ewretd'].to_numpy(dtype=float))} |",
            f"| Fama-French market, excess | {stats(ff['mktrf'].to_numpy(dtype=float))} |"]
    (PUBLIC / "checks").mkdir(parents=True, exist_ok=True)
    (PUBLIC / "checks" / "market_row.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


# receipts of Chinese firms that the NYSE delisted to comply with an executive order; last trade on 8 January 2021
FORCED_2021 = {88282: "China Unicom", 89581: "China Telecom", 85421: "China Mobile"}


def delisting_check() -> None:
    """What the delisting returns of the three receipts add to each run: the result with and without them."""
    from company_comparison import ROWS
    meta = np.load(C / "panel" / "meta.npz", allow_pickle=False)
    dates = meta["dates"].astype("datetime64[D]")
    day = np.datetime64("2021-01-11")                                    # the day the delisting returns are booked
    r_day = np.nan_to_num(np.load(C / "panel" / "daily_ret.npy", mmap_mode="r")[int(np.where(dates == day)[0][0])]).astype(np.float64)
    cols = [int(np.where(meta["permnos"] == q)[0][0]) for q in FORCED_2021]
    labels = {run.format(k=k): f"{label}, K = {k}" for k in (8, 30) for label, _, run, _ in ROWS}
    labels["attention_K30_seed1"] = "Attention Factor model, objective as printed, K = 30, second seed"

    def sr(x: np.ndarray) -> float:
        return float(x.mean() * 252 / (x.std(ddof=1) * np.sqrt(252)))

    out = ["## The forced delistings of January 2021", "",
           "Returns booked on 11 January 2021 for the receipts of " + ", ".join(
               f"{n} ({100 * r_day[c]:.0f} percent)" for n, c in zip(FORCED_2021.values(), cols)) + ".",
           "\"Without\": the result of that day without the three returns. Sharpe ratios before / after costs.", "",
           "| Run | Weight on the three, % of the portfolio | Their part of the day's return, points | 1998 to 2021 | without | 2021 alone | without |",
           "|---|---|---|---|---|---|---|"]
    for name, label in labels.items():
        wf, df = ROOT / "runs" / PRIVATE / name / "weights_2021.npy", PUBLIC / name / "daily.csv"
        if not (wf.exists() and df.exists()):
            continue
        d = pd.read_csv(df, parse_dates=["date"])
        y = (d["date"].dt.year == 2021).to_numpy()
        i = int(np.where(d["date"].to_numpy() == day)[0][0])
        w = np.load(wf)[i - int(np.argmax(y))].astype(np.float64)       # the weights that earn the return of that day
        if abs(float(w @ r_day) - d["gross"][i]) > 1e-6:
            raise RuntimeError(f"{name}: the weights of 2021 do not give the saved return of {day}")
        part = float(w[cols] @ r_day[cols])
        cells = []
        for rows in (slice(None), y):
            with_, without = [], []
            for col in ("gross", "net"):
                x = d[col].to_numpy().copy()
                with_.append(sr(x[rows]))
                x[i] -= part
                without.append(sr(x[rows]))
            cells += [f"{with_[0]:.2f} / {with_[1]:.2f}", f"{without[0]:.2f} / {without[1]:.2f}"]
        out.append(f"| {label} | {100 * w[cols].sum():+.2f} | {100 * part:+.2f} | " + " | ".join(cells) + " |")
    (PUBLIC / "checks").mkdir(parents=True, exist_ok=True)
    (PUBLIC / "checks" / "delisting_2021.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


# ---------------------------------------------------------------- runs

def run_case(name: str, threads: int) -> None:
    if os.environ.get("ATTNFACTORS_DATA") != DATA_ROOT:
        raise RuntimeError(f"{name} needs ATTNFACTORS_DATA={DATA_ROOT}")
    import torch
    from attnfactors.pca_regression import PcaRegressionData
    from attnfactors.train import Config, Data, PcaData, run
    spec = MAIN[name]
    torch.set_num_threads(threads)
    # the panel must hold the company values of this round (panel.build alone writes CRSP values)
    meta = np.load(C / "panel" / "meta.npz", allow_pickle=False)
    cv = pd.read_parquet(C / "company_value.parquet").sample(3000, random_state=0)
    pi, mi = {int(q): i for i, q in enumerate(meta["permnos"])}, {str(m): i for i, m in enumerate(meta["months"])}
    cv = cv[cv["permno"].isin(pi) & cv["month"].isin(mi)]
    me = np.load(C / "panel" / "monthly_me.npy", mmap_mode="r")
    got = np.array([me[mi[m], pi[q]] for m, q in zip(cv["month"], cv["permno"])])
    if not np.allclose(got, cv["value"].to_numpy(dtype=np.float32), rtol=1e-5):
        raise RuntimeError("data/company/panel does not hold the company values: run the stage 'build' again")
    cfg = Config(kind=spec["kind"], k=spec["k"], seed=spec["seed"], batch=spec["batch"], ev_floor=spec["ev_floor"],
                 sr_scale=spec["sr_scale"], name=f"{PRIVATE}/{name}")
    data_ = Data()
    pd_ = None
    if spec["kind"] != "attention":
        pd_ = PcaRegressionData(data_) if spec["pca"] == "regression" else PcaData(data_)
    run(cfg, data_, pd_)
    out = PUBLIC / name
    out.mkdir(parents=True, exist_ok=True)
    src = ROOT / "runs" / PRIVATE / name
    metrics = json.loads((src / "metrics.json").read_text())
    metrics["round"] = {"universe": "company value, units, history of 120 months", "characteristics": "cited",
                        "pca_residuals": spec["pca"], "data_root": DATA_ROOT}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    shutil.copyfile(src / "daily.csv", out / "daily.csv")


def run_many(names: list[str], workers: int, threads: int) -> None:
    def one(name: str) -> tuple[str, int]:
        logf = ROOT / "runs" / PRIVATE / f"job-{name}.log"
        logf.parent.mkdir(parents=True, exist_ok=True)
        with logf.open("a") as f:
            cmd = [sys.executable, str(Path(__file__).resolve()), "case", "--name", name, "--threads", str(threads)]
            return name, subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode

    with ThreadPoolExecutor(workers) as pool:
        failed = [n for n, code in pool.map(one, names) if code != 0]
    if failed:
        raise SystemExit(f"failed: {failed} (see runs/{PRIVATE}/job-*.log)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["download", "universe", "data", "build", "market", "main", "case", "delisting", "report"])
    ap.add_argument("--name", help="with 'case': the run; with 'main': a comma-separated subset")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    if a.stage == "case":
        run_case(a.name, a.threads)
    elif a.stage == "main":
        run_many(a.name.split(",") if a.name else list(MAIN), a.workers, a.threads)
    else:
        {"download": download, "universe": universe, "data": data, "build": build, "market": market_check,
         "delisting": delisting_check, "report": lambda: None}[a.stage]()
    if a.stage == "main":
        delisting_check()
    if a.stage in ("main", "report"):
        from company_comparison import report
        report()


if __name__ == "__main__":
    main()
