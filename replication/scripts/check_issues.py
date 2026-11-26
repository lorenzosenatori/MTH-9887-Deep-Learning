"""Checks behind README_ISSUES.md: is each reported difference from the paper real, and how large?

    python scripts/check_issues.py floor      # needs data/ and runs/attention_K30_seed0 (the models of round 1)
    python scripts/check_issues.py steps      # needs data/; trains the validation model once (about 5 minutes)
    python scripts/check_issues.py universe   # needs data/raw
    python scripts/check_issues.py company    # needs a WRDS token; about 5 requests

Each check writes one file under results/as_cited/checks/. Only aggregate numbers and names
that the paper itself prints are written.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "as_cited" / "checks"

# Firms that Figures 3 and 4 of the paper name, as printed there (CRSP's own abbreviations). After a
# semicolon: another spelling of the same firm in CRSP's name history.
FIGURE_3 = """Ameriprise Financial|Intercontinentalexchange|C M E Group|Nasdaq|Broadridge Financial Solutns|Celanese|Blackstone|
American Water Works|Energy Transfer|Unitedhealth|First Solar|Leidos|Chipotle Mexican Grill|Philip Morris International|
Natwest;Royal Bank of Scotland|Anheuser Busch Inbev|Netflix|Johnson & Johnson|Ecopetrol|Banco Santander Brasil|Molson Coors|
Pfizer|Delta Air Lines|Cheniere Energy Partners|Masimo|T Mobile|Marketaxess|Liberty Global|Las Vegas Sands|Visa Inc|Alphabet|
Mizuho Financial|United Airlines;United Continental|Shell PLC;Royal Dutch Shell|Imperial Oil|H World;Huazhu|Mastercard|M S C I|
Wheaton Precious Metals|Microsoft|Apple Inc|Expedia|I P G Photonics|Baidu|Fortinet|Generac|Live Nation Entertainment|
Monolithic Pwr Sys|Tesla|Mercadolibre|Digital Realty Trust|Medical Properties Trust"""
FIGURE_4 = """K L A Corp|On Semiconductor|N X P Semiconductors|Align Technology|Qorvo|Lam Resh|Entegris|Conocophillips|
Totalenergies;Total S A;Total S E|Suncor Energy|Exxon Mobil|Canadian Natural Resources|Pioneer Natural Resources|Equinor;Statoil|
Schlumberger|B H P;BHP|Keycorp|Huntington Bancshares|Fifth Third Bancorp|Bank of America|M & T Bank|Regions Financial|Citigroup|
Wells Fargo|Truist Financial|U S Bancorp|Extra Space Storage|Essex Property Trust|U D R Inc|Avalonbay Communities|
Equity Residential|Public Storage|Camden Property Trust|Healthpeak Properties|Mid America Apt Communities|
Equity Lifestyle Properties|Unilever|Novartis|Sanofi|Toyota Motor|British American Tobacco|Glaxosmithkline;G S K|Merck & Co|
Procter & Gamble|Orange|Verizon Communications|Consolidated Edison|American Electric Power|Ameren|C M S Energy|
Dominion Energy|Duke Energy|X C E L Energy|Evergy|W E C Energy|Alliant Energy"""


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def floor() -> list[str]:
    """How often does the lower bound of -1 on one security's explained-variance term bind?

    The saved 30-factor model of round 1 (seed 0) is replayed over its own training window in
    batches of 64 days, without dropout and with its weights fixed."""
    import torch
    from attnfactors.model import AttentionFactorModel
    from attnfactors.panel import PANEL
    from attnfactors.train import Config, Data, att_forward
    assert PANEL.name == "panel", "the models of round 1 belong to data/panel: unset ATTNFACTORS_PANEL"
    torch.set_num_threads(2)
    data = Data()
    cfg = Config(**json.loads((ROOT / "results/runs/attention_K30_seed0/metrics.json").read_text())["config"])
    out = ["## The lower bound on the explained-variance term", "",
           "The saved model of round 1 (30 factors, seed 0) for a test year, replayed over its eight training years in",
           "batches of 64 days (fixed cuts; the training used random cuts). A term is one security in one batch.", "",
           "| Model for the test year | Batches in which the bound binds | Terms below -1 | Mean term, bounded | Mean term, as printed | Difference, points | Lowest term |",
           "|---|---|---|---|---|---|---|"]
    whole = ["", "The same models with one variance per security over the whole training window:", "",
             "| Model for the test year | Securities ever in the universe | ... with a return on half of the days | Terms below -1 among them | Mean term |",
             "|---|---|---|---|---|"]
    for year in (1998, 2010, 2021):
        lo, hi = data.train_range(year, cfg.train_years)
        lo = max(lo, int((data.month_row >= 0).nonzero()[0]) + cfg.s + 1)
        stats = data.common_stats(lo, hi)
        model = AttentionFactorModel(data.m_firm, data.m_common, cfg.k, cfg.d, cfg.s, cfg.d_h, cfg.dropout, cfg.squash, cfg.ridge)
        model.load_state_dict(torch.load(ROOT / f"runs/attention_K30_seed0/model_{year}.pt", weights_only=True))
        model.eval()
        union = data.member[lo - cfg.s:hi].any(0).nonzero().squeeze(1)
        pos = torch.full((data.member.shape[1],), -1, dtype=torch.long)
        pos[union] = torch.arange(len(union))

        def terms(eps, r, valid):
            v = valid.float()
            n = v.sum(0)
            keep = n >= 0.5 * valid.shape[0]
            n = n.clamp(min=1)
            var = lambda x: (((x - (x * v).sum(0) / n) ** 2) * v).sum(0) / n  # noqa: E731
            vr = var(r)
            keep = keep & (vr > 1e-10)
            return 1 - var(eps)[keep] / vr[keep]

        rows, parts = [], {"eps": [], "r": [], "valid": []}
        with torch.no_grad():
            for j0 in range(lo, hi, 64):
                j1 = min(j0 + 64, hi)
                o, cols = att_forward(model, data, j0, j1, stats, True)
                t = terms(o["eps"], o["r"], o["valid"])
                rows.append((int((t < -1).sum()), len(t), float(t.clamp(min=-1).mean()), float(t.mean()), float(t.min())))
                for name in parts:
                    z = torch.zeros(j1 - j0, len(union), dtype=o[name].dtype)
                    z[:, pos[cols]] = o[name]
                    parts[name].append(z)
            t = terms(torch.cat(parts["eps"]), torch.cat(parts["r"]), torch.cat(parts["valid"]))
        a = np.array(rows)
        out.append(f"| {year} | {(a[:, 0] > 0).sum()} of {len(a)} | {int(a[:, 0].sum())} of {int(a[:, 1].sum()):,} | {a[:, 2].mean():.4f} | "
                   f"{a[:, 3].mean():.4f} | {100 * (a[:, 2].mean() - a[:, 3].mean()):.2f} | {a[:, 4].min():.1f} |")
        whole.append(f"| {year} | {len(union)} | {len(t)} | {int((t < -1).sum())} | {float(t.mean()):.4f} |")
    return out + whole


def steps() -> list[str]:
    """Validation performance against the number of optimizer steps, for the per-day objective of round 2."""
    import torch
    from attnfactors.model import AttentionFactorModel, objective
    from attnfactors.panel import PANEL
    from attnfactors.train import Config, Data, apply_attention, att_forward, batches, performance
    assert PANEL.name == "panel", "this check repeats a run of round 2a on data/panel: unset ATTNFACTORS_PANEL"
    torch.set_num_threads(2)
    data = Data()
    cfg = Config(**json.loads((ROOT / "results/runs/val_att_K30_sr-per-day/metrics.json").read_text())["config"])
    marks = [0, 1, 2, 3, 5, 10, 30]
    w, at = {m: [] for m in marks}, {}
    for year in (1996, 1997):
        torch.manual_seed(cfg.seed * 100003 + year)
        rng = np.random.default_rng(cfg.seed * 100003 + year)
        lo, hi = data.train_range(year, cfg.train_years)
        lo = max(lo, int((data.month_row >= 0).nonzero()[0]) + cfg.s + 1)
        stats = data.common_stats(lo, hi)
        model = AttentionFactorModel(data.m_firm, data.m_common, cfg.k, cfg.d, cfg.s, cfg.d_h, cfg.dropout, cfg.squash, cfg.ridge)
        opt = torch.optim.AdamW([{"params": model.factors.parameters(), "weight_decay": 0.0},
                                 {"params": model.policy.parameters(), "weight_decay": cfg.weight_decay}], lr=cfg.lr)
        a, b = data.test_range(year)
        n = 0
        w[0].append(apply_attention(model, data, a, b, stats, cfg)[0])
        at[0] = 0
        for epoch in range(1, max(marks) + 1):
            model.train()
            for j0, j1 in batches(lo, hi, cfg.batch, rng):
                o, _ = att_forward(model, data, j0, j1, stats, True)
                obj, _ = objective(o, data.rf_next[j0:j1], cfg.lambda_var, cfg.sr_scale)
                opt.zero_grad(set_to_none=True)
                (-obj).backward()
                model.factors.W.grad[model.factors.m_firm:] = 0.0
                opt.step()
                n += 1
            if epoch in marks:
                w[epoch].append(apply_attention(model, data, a, b, stats, cfg)[0])
                at[epoch] = n
    lo, hi = data.test_range(1996)[0], data.test_range(1997)[1]
    out = ["## Optimizer steps", "",
           "Validation years 1996 and 1997, 30 factors, seed 0, the Sharpe ratio per day in the objective, batches of 64 days.",
           "The same training run, evaluated after a number of passes over the data.", "",
           "| Passes over the data | Optimizer steps a year | Sharpe ratio | after costs | Turnover a day |", "|---|---|---|---|---|"]
    for m in marks:
        p, _ = performance(np.vstack(w[m]), data, lo, hi)
        out.append(f"| {m} | {at[m]} | {p['SR']:.2f} | {p['SR_net']:.2f} | {p['turnover_per_day']:.2f} |")
    full = json.loads((ROOT / "results/adjusted_loss/validation/full_uncapped/metrics.json").read_text())
    out += ["", f"The whole-window run of round 2 on the same years takes {sorted(set(full['optimizer_steps'].values()))[0]} steps a year: "
                f"Sharpe ratio {full['SR']:.2f}, after costs {full['SR_net']:.2f}, turnover {full['turnover_per_day']:.2f}."]
    return out


def universe() -> list[str]:
    """Which of the firms that the paper's Figures 3 and 4 name are in this replication's universe in 2021?"""
    raw = ROOT / "data" / "raw"
    names = pd.read_parquet(raw / "msenames.parquet")
    for c in ("namedt", "nameendt"):
        names[c] = pd.to_datetime(names[c])
    cur = names[(names.nameendt >= "2021-01-01") & (names.namedt <= "2021-12-31")]
    uni = pd.read_parquet(ROOT / "data" / "universe.parquet")
    u21 = uni[uni["month"].astype(str).str.startswith("2021")]
    jan = set(uni.loc[uni["month"].astype(str) == "2021-01", "permno"])       # the month of the figures' date
    kind = lambda c: "depositary receipt" if c in (30, 31) else "partnership unit" if c // 10 == 7 else "common share"  # noqa: E731
    rows = []
    for fig, block in (("3", FIGURE_3), ("4", FIGURE_4)):
        for item in block.replace("\n", "").split("|"):
            keys = [_norm(k) for k in item.split(";")]
            hit = cur[cur.comnam.fillna("").map(lambda c: any(k in _norm(c) for k in keys))]
            label = item.split(";")[0]
            if hit.empty:
                rows.append((fig, label, "never among the 900 largest securities of a month", 0, False))
                continue
            best = max(((int((u21.permno == p).sum()), int(hit[hit.permno == p].shrcd.iloc[-1])) for p in hit.permno.unique()))
            rows.append((fig, label, kind(best[1]), best[0], any(p in jan for p in hit.permno.unique())))
    df = pd.DataFrame(rows, columns=["figure", "name", "security", "months", "january"])
    out = ["## Firms that the paper's figures name, in this replication's universe of 2021", "",
           f"Figures 3 and 4 of the paper name {len(df)} firms among its 500 on the first trading day of 2021. In this",
           "replication's universe (the 500 largest securities by CRSP capitalization of the security at the end of the",
           "previous month, share codes 10, 11, 12, 18, 48, 30, 31):", "",
           f"- in January 2021, the month of that day: {int(df.january.sum())} are members, {int((~df.january).sum())} are not",
           f"- in all 12 months of 2021: {(df.months == 12).sum()}",
           f"- in some months: {((df.months > 0) & (df.months < 12)).sum()}",
           f"- in no month: {(df.months == 0).sum()}", "",
           "The firms that are not members in January 2021:", "",
           "| Firm, as the paper prints it | Security in CRSP | Months of 2021 in our universe |", "|---|---|---|"]
    for r in df[~df.january].sort_values(["security", "name"]).itertuples():
        out.append(f"| {r.name} | {r.security} | {r.months} |")
    return out


def company() -> list[str]:
    """CRSP capitalization of the listed security beside the value of the whole company in Compustat."""
    from attnfactors.wrds_api import Wrds
    w = Wrds()
    tickers = ["EC", "MFG", "NWG", "ORAN", "EQNR", "BSBR", "TM", "BUD", "TTE", "UL", "SNY", "BTI", "BHP"]
    n = pd.DataFrame(w.rows("crsp.msenames", {"ticker__in": ",".join(tickers), "nameendt__gte": "2020-12-31"}, limit=2000))
    n = n[n.shrcd.astype(float).isin([30, 31])].sort_values(["ticker", "namedt"]).drop_duplicates("permno", keep="last")
    m = pd.read_parquet(ROOT / "data" / "raw" / "msf" / "2020-12.parquet")
    m["crsp"] = m.altprc.abs().fillna(m.prc.abs()) * m.shrout / 1e6
    m["rank"] = m.crsp.rank(ascending=False, method="first")
    x = n.merge(m[["permno", "crsp", "rank"]], on="permno")
    links = pd.DataFrame(w.rows("crsp.ccmxpf_lnkhist", {"lpermno__in": ",".join(str(p) for p in x.permno)}, limit=2000))
    links = links[links.linktype.isin(["LC", "LU", "LS"])].drop_duplicates("lpermno", keep="last")[["lpermno", "gvkey"]]
    x = x.merge(links, left_on="permno", right_on="lpermno")
    fa = pd.DataFrame(w.rows("comp.funda", {"gvkey__in": ",".join(x.gvkey.unique()), "fyear__gte": 2019, "fyear__lte": 2020,
                                            "indfmt": "INDL", "datafmt": "STD", "consol": "C", "popsrc": "D"}, limit=2000))
    fa = fa[(fa.indfmt == "INDL") & (fa.datafmt == "STD") & (fa.consol == "C") & (fa.popsrc == "D")]
    fa["company"] = pd.to_numeric(fa.csho, errors="coerce") * pd.to_numeric(fa.prcc_f, errors="coerce") / 1e3
    fa = fa.sort_values(["gvkey", "datadate"]).drop_duplicates("gvkey", keep="last")
    x = x.merge(fa[["gvkey", "company"]], on="gvkey").sort_values("rank")
    out = ["## The listed security and the whole company", "",
           "Depositary receipts of foreign firms that the paper's figures name. CRSP capitalization of the receipt at the end of",
           "2020 (price times receipts outstanding) and its rank among all CRSP securities; beside it the value of the company",
           "in Compustat (shares times price of the last fiscal year). Billion USD, rounded.", "",
           "| Firm | CRSP capitalization | Rank in CRSP | Company in Compustat |", "|---|---|---|---|"]
    for r in x.itertuples():
        out.append(f"| {r.comnam.title()} | {r.crsp:.1f} | {int(round(r.rank, -1)) if r.rank > 1500 else int(r.rank)} | {r.company:.0f} |")
    return out


CHECKS = {"floor": floor, "steps": steps, "universe": universe, "company": company}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (sys.argv[1:] or ["floor", "steps", "universe"]):
        (OUT / f"{name}.md").write_text("\n".join(CHECKS[name]()) + "\n")
        print((OUT / f"{name}.md").read_text())
