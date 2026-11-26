"""The tables of RUNS.md: one row per run with its settings, its results and its commits.

Reads results/**/metrics.json, the git history of each of those files and, where they exist, the folders
under runs/ that hold runs which never reached results/. Writes the tables between the markers of RUNS.md;
the lists of sources in that file are written by hand. The table of the runs under runs/ is left as it is
on a machine that does not have them (runs/ is git-ignored).

    python scripts/runs_table.py            # from replication/
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES, LOCAL, OUT = ROOT / "results", ROOT / "runs", ROOT / "RUNS.md"

# round -> (what it covers, script, results folder)
ROUNDS = {
    "1": ("The first implementation; its open choices were set on the validation years. The numbers on the slides.",
          "scripts/run.py", "runs"),
    "1b": ("Six more runs on the specification of round 1, made on the branch `experiments/open-problems`: 100 factors, one row of "
           "Table 3, the residual history of the decision date, and PCA of the correlation matrix.",
           "scripts/run.py --pca, --lookback, --drop, --k 100", "runs"),
    "2a": ("Round 1 with the Sharpe ratio per day in the objective, as the paper prints it. Nothing else changed.",
           "scripts/replicate_daily_sharpe.py", "daily_sharpe"),
    "2b": ("Round 2a with the objective over the whole training window as one batch and without the bound on the "
           "explained-variance term. Not a valid test: 30 optimizer steps a year.", "scripts/replicate_adjusted_loss.py", "adjusted_loss"),
    "3": ("Choices taken from the sources that the paper cites: Sharpe ratio per day, batches of 125 days, no bound, three "
          "inputs as cited, PCA residuals with loadings from 60 days.", "scripts/replicate_as_cited.py", "as_cited"),
    "4": ("Round 3 on another universe: each month the 500 largest by the value of the whole company, with partnership units "
          "and 120 months of history. The rule was inferred from the 108 firms that the paper's figures name; no firm is put "
          "in by name. The objective and the PCA residuals are each run both ways.",
          "scripts/replicate_company_universe.py", "company"),
}
FOLDER_ROUND = {folder: r for r, (_, _, folder) in ROUNDS.items() if r != "1b"}
# the runs of round 1b share the folder of round 1
ROUND_1B = {"attention_K100_seed0", "attention_K30_look-current_seed0", "attention_K30_no-past_returns_seed0", "pca_K30_corr_seed0",
            "pca_K30_corr_look-current_seed0", "pca_K30_look-current_seed0"}
MODELS = {"attention": ("AF", "Attention Factor model"), "pca": ("PCA", "PCA with the long convolution model"),
          "ou": ("OU", "PCA with an OU threshold rule")}
ORDER = {"attention": 0, "pca": 1, "ou": 2}
# the paper's Table 2: (Sharpe ratio, Sharpe ratio after costs) by model and number of factors
PAPER = {
    "attention": {1: (3.05, 1.68), 3: (3.05, 1.69), 5: (2.92, 1.58), 8: (3.35, 1.94), 15: (3.81, 2.25), 30: (3.97, 2.28), 100: (4.52, 2.19)},
    "ou": {1: (0.40, -2.54), 3: (1.26, -2.72), 5: (0.99, -3.44), 8: (0.78, -4.15), 10: (0.80, -4.32), 15: (0.51, -5.24),
           30: (0.18, -6.45), 100: (-0.35, -7.05)},
    "pca": {1: (2.26, 1.19), 3: (2.76, 1.57), 5: (2.41, 1.30), 8: (2.64, 1.50), 10: (2.66, 1.52), 15: (2.56, 1.41), 30: (2.79, 1.57),
            100: (2.66, 1.44)},
}
PAPER_TABLE3 = {"past_returns": (1.50, 0.59)}                   # the paper's Table 3, 30 factors, one group removed
# validation runs: (results folder, run folder) -> (ID, what the run tests)
VALIDATION = {
    ("runs", "val_att_K30"): ("V1-AF30", "The specification of round 1"),
    ("runs", "val_att_K30_sr-per-day"): ("V1-AF30-day", "... with the Sharpe ratio per day (the printed objective)"),
    ("runs", "val_att_K30_lambda0"): ("V1-AF30-lam0", "... without the explained-variance term"),
    ("runs", "val_att_K30_ridge1e-4"): ("V1-AF30-r4", "... with a ridge penalty of 1e-4"),
    ("runs", "val_att_K30_ridge1e-6"): ("V1-AF30-r6", "... with a ridge penalty of 1e-6"),
    ("runs", "val_att_K30_ridge1e-7"): ("V1-AF30-r7", "... with a ridge penalty of 1e-7"),
    ("runs", "val_att_K8"): ("V1-AF8", "The specification of round 1, 8 factors"),
    ("runs", "val_pca_K30"): ("V1-PCA30", "The benchmark of round 1"),
    ("runs", "val_pca_K8"): ("V1-PCA8", "The benchmark of round 1, 8 factors"),
    ("runs", "val_ou_K8"): ("V1-OU8", "The OU rule of round 1, 8 factors"),
    ("adjusted_loss", "validation/short_uncapped"): ("V2b-AF30-nobound", "Round 2a without the bound"),
    ("adjusted_loss", "validation/full_capped"): ("V2b-AF30-window", "Round 2a with the whole window as one batch"),
    ("adjusted_loss", "validation/full_uncapped"): ("V2b-AF30-window-nobound", "The specification of round 2b"),
    ("as_cited", "validation/att_b125_floor"): ("V3-AF30-b125", "Round 2a with batches of 125 days"),
    ("as_cited", "validation/att_b125_nofloor"): ("V3-AF30-b125-nobound", "... and without the bound"),
    ("as_cited", "validation/att_b64_floor_cited"): ("V3-AF30-cited", "Round 2a with the cited inputs"),
    ("as_cited", "validation/att_b125_nofloor_cited"): ("V3-AF30", "The specification of round 3"),
    ("as_cited", "validation/att_b125_nofloor_cited_annualized"): ("V3-AF30-ann", "The specification of round 3 with the annualized Sharpe ratio"),
    ("as_cited", "validation/att_K8_b125_nofloor_cited"): ("V3-AF8", "The specification of round 3, 8 factors"),
    ("as_cited", "validation/pca_b125_projection"): ("V3-PCA30-cov", "The benchmark of round 1 with batches of 125 days"),
    ("as_cited", "validation/pca_b125_regression252"): ("V3-PCA30-reg252", "Correlation matrix, loadings from all 252 days"),
    ("as_cited", "validation/pca_b64_regression"): ("V3-PCA30-b64", "Loadings from 60 days, batches of 64 days"),
    ("as_cited", "validation/pca_b125_regression"): ("V3-PCA30", "The benchmark of round 3"),
    ("as_cited", "validation/pca_K8_b125_regression"): ("V3-PCA8", "The benchmark of round 3, 8 factors"),
    ("as_cited", "validation/ou_K30_regression"): ("V3-OU30", "The OU rule of round 3"),
    ("as_cited", "validation/ou_K8_regression"): ("V3-OU8", "The OU rule of round 3, 8 factors"),
}
PCA_LABEL = {"projection": "covariance", "corr": "correlation", "regression": "correlation, 60-day loadings",
             "regression252": "correlation, 252-day loadings"}


def settings(folder: str, d: dict) -> dict[str, str]:
    """The settings that differ between runs, as the run recorded them. "-": the setting does not act on the model."""
    c, extra = d["config"], d.get("as_cited") or d.get("round") or {}
    att, trained = c["kind"] == "attention", c["kind"] != "ou"
    # train_pca takes the Sharpe ratio per day whatever sr_scale says; its objective has one term
    sharpe = ("annualized" if c["sr_scale"] > 1 else "per day") if att else "per day" if trained else "-"
    batch = "-" if not trained else "whole window" if c.get("loss_window") == "full training window" else str(c["batch"])
    # a run made before the setting existed has the bound: it was fixed in the code
    bound = "-" if not att else "none" if "ev_floor" in c and c["ev_floor"] is None else "-1"
    inputs = "-"
    if att:
        inputs = {"cited": "cited", "reconstructed": "round 1"}[extra.get("characteristics", "reconstructed")]
        inputs += "".join(f", no {g.replace('_', ' ')}" for g in c["drop_groups"])
    pca = "-" if att else PCA_LABEL["corr" if c.get("pca") == "corr" else extra.get("pca_residuals") or "projection"]
    return {"sharpe": sharpe, "batch": batch, "bound": bound, "inputs": inputs, "pca": pca, "lookback": c.get("lookback", "own"),
            "universe": "company value" if folder == "company" else "security value",
            "ridge": f"{c['ridge']:g}" if att else "-", "lam": f"{c['lambda_var']:g}" if att else "-"}


def round_of(folder: str, name: str) -> str:
    return "1b" if folder == "runs" and name in ROUND_1B else FOLDER_ROUND[folder]


def run_id(folder: str, name: str, c: dict) -> str:
    tag = MODELS[c["kind"]][0] + str(c["k"])
    tag += ("corr" if c.get("pca") == "corr" else "") + ("cur" if c.get("lookback") == "current" else "")
    tag += "".join("no" + g.split("_")[0] for g in c["drop_groups"])
    if folder == "company":
        tag += ("ann" if "annualized" in name else "") + ("cov" if "proj" in name else "")
    return f"{round_of(folder, name)}-{tag}" + ("" if c["kind"] == "ou" else f"-s{c['seed']}")


def stamp(f: Path) -> str:
    return time.strftime("%d %b %H:%M", time.localtime(f.stat().st_mtime))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def commits(f: Path) -> list[tuple[str, str, str]]:
    """(hash, time, note) of every commit that changed the file, oldest first. The note on a later commit says
    whether the run was made again (its daily returns changed too) or only its saved settings were rewritten."""
    out = []
    for line in git("log", "--format=%h|%ad", "--date=format:%d %b %H:%M", "--", str(f)).splitlines()[::-1]:
        h, t = line.split("|")
        rerun = bool(git("diff-tree", "--no-commit-id", "--name-only", "-r", h, "--", str(f.with_name("daily.csv"))).strip())
        out.append((h, t, "" if not out else " (rerun)" if rerun else " (saved settings rewritten, no rerun)"))
    return out


def commit_cell(f: Path) -> str:
    cs = commits(f)
    return "; ".join(f"`{h}` {t}{note}" for h, t, note in cs) if cs else f"not committed (file of {stamp(f)})"


def load() -> list[dict]:
    rows = []
    for folder in FOLDER_ROUND:
        for f in sorted((RES / folder).glob("*/metrics.json")) + sorted((RES / folder).glob("validation/*/metrics.json")):
            d = json.loads(f.read_text())
            c, name = d["config"], str(f.parent.relative_to(RES / folder))
            val = c["first_year"] == 1996
            rid, covers = VALIDATION[(folder, name)] if val else (run_id(folder, name, c), "")
            rows.append({"id": rid, "covers": covers, "folder": folder, "name": name, "val": val, "d": d, "c": c, "s": settings(folder, d),
                         "round": round_of(folder, name), "file": f, "commit": commit_cell(f)})
    ids = [x["id"] for x in rows]
    assert len(set(ids)) == len(ids), "two runs with one ID"
    return rows


def n(x: float) -> str:
    return f"{x:.2f}"


def pair(a: float, b: float) -> str:
    return f"{n(a)} / {n(b)}"


def table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def rounds(rows: list[dict]) -> list[str]:
    out = []
    for r, (what, script, folder) in ROUNDS.items():
        mine = [x for x in rows if x["round"] == r]
        seen = sorted({(t, h) for x in mine for h, t, _ in commits(x["file"])}, key=lambda p: time.strptime(p[0], "%d %b %H:%M"))
        when = "; ".join(f"`{h}` {t}" for t, h in seen) if seen else "not committed"
        out.append([r, what, f"`{script}`", f"`results/{folder}/`", f"{sum(not x['val'] for x in mine)} + {sum(x['val'] for x in mine)}", when])
    return table(["Round", "What it covers", "Script", "Results", "Runs: out of sample + validation", "Commits (hash, time)"], out)


def key(x: dict):
    return list(ROUNDS).index(x["round"]), ORDER[x["c"]["kind"]], x["c"]["k"], x["id"]


def paper(c: dict) -> str:
    p = PAPER_TABLE3.get(c["drop_groups"][0]) if c["drop_groups"] else PAPER[c["kind"]].get(c["k"])
    return pair(*p) if p else ""


def covers(c: dict) -> str:
    s = f"{MODELS[c['kind']][1]}, K = {c['k']}" + ("" if c["kind"] == "ou" else f", seed {c['seed']}")
    return s + (" (Table 3: " + ", ".join(g.replace("_", " ") for g in c["drop_groups"]) + " removed)" if c["drop_groups"] else "")


def main_table(rows: list[dict]) -> list[str]:
    out = []
    for x in sorted((x for x in rows if not x["val"]), key=key):
        d, c, s = x["d"], x["c"], x["s"]
        out.append([f"**{x['id']}**", covers(c), s["sharpe"], s["batch"], s["bound"], s["inputs"], s["lookback"], s["pca"], s["universe"],
                    pair(d["SR"], d["SR_net"]), paper(c), pair(d["mu"], d["mu_net"]), n(d["sigma"]), n(d["turnover_per_day"]),
                    f"`{x['folder']}/{x['name']}`", x["commit"]])
    return table(["ID", "Covers", "Sharpe ratio in the objective", "Batch, days", "Bound", "Inputs", "Residual history",
                  "PCA residuals", "Universe", "SR before / after costs", "Paper", "Mean % before / after costs",
                  "Volatility %", "Turnover a day", "Folder under `results/`", "Commit (hash, time)"], out)


def validation_table(rows: list[dict]) -> list[str]:
    out = []
    order = list(VALIDATION.values())
    for x in sorted((x for x in rows if x["val"]), key=lambda x: order.index((x["id"], x["covers"]))):
        d, c, s = x["d"], x["c"], x["s"]
        out.append([f"**{x['id']}**", x["covers"], f"{MODELS[c['kind']][0]}, K = {c['k']}", s["sharpe"], s["batch"], s["bound"], s["inputs"],
                    s["pca"], s["ridge"], s["lam"], pair(d["SR"], d["SR_net"]), n(d["mu_net"]), n(d["sigma"]), n(d["turnover_per_day"]),
                    f"`{x['folder']}/{x['name']}`", x["commit"]])
    return table(["ID", "Covers (what the run tests)", "Model", "Sharpe ratio in the objective", "Batch, days", "Bound", "Inputs",
                  "PCA residuals", "Ridge penalty", "Weight on explained variance", "SR before / after costs", "Mean after costs %",
                  "Volatility %", "Turnover a day", "Folder under `results/`", "Commit (hash, time)"], out)


def averages(rows: list[dict]) -> list[str]:
    groups: dict[str, list[dict]] = {}
    for x in sorted((x for x in rows if not x["val"] and x["c"]["kind"] != "ou"), key=key):
        groups.setdefault(x["id"].rsplit("-s", 1)[0], []).append(x)
    out = []
    for xs in groups.values():
        if len(xs) > 1:
            c = xs[0]["c"]
            out.append([" and ".join(f"**{x['id']}**" for x in xs), f"{MODELS[c['kind']][1]}, K = {c['k']}",
                        pair(sum(x["d"]["SR"] for x in xs) / len(xs), sum(x["d"]["SR_net"] for x in xs) / len(xs)), paper(c)])
    return table(["Runs", "Covers", "Mean SR before / after costs", "Paper"], out)


def local_table() -> list[str] | None:
    """Runs under runs/ that never reached results/: the validation sweeps before the code audit, the first versions of
    the benchmark runs of round 1, and the runs on an earlier build of the data of round 4."""
    groups = [("_pre_audit", "P", "Validation before the code audit of round 1"),
              ("_before_rerun", "B", "First version of a benchmark run of round 1, before its rerun"),
              ("company_discarded", "D", "Round 4 on an earlier build of its data, discarded")]
    if not all((LOCAL / g).is_dir() for g, _, _ in groups):
        return None
    out = []
    for g, letter, what in groups:
        for p in sorted(x for x in (LOCAL / g).iterdir() if x.is_dir() or x.name.endswith(".metrics.json")):
            name = p.name.replace(".metrics.json", "")
            f = p if p.is_file() else p / "metrics.json"
            if f.exists():
                d = json.loads(f.read_text())
                c = d["config"]
            elif (p / "config.json").exists():      # stopped before its last year: only the settings were saved
                d, c = None, json.loads((p / "config.json").read_text())
            else:
                continue
            att, trained = c["kind"] == "attention", c["kind"] != "ou"
            sharpe = ("annualized" if c["sr_scale"] > 1 else "per day") if att else "per day" if trained else "-"
            pca = "-" if att else PCA_LABEL["projection" if "proj" in name or g != "company_discarded" else "regression"]
            years = f"{c['first_year']} to {c['last_year']}" if "first_year" in c else "1998 to 2021"
            done = sorted(x.name for x in p.glob("weights_*.npy")) if p.is_dir() else []
            out.append([f"**{letter}-{name}**", what, f"{MODELS[c['kind']][0]}, K = {c['k']}" + (f", seed {c['seed']}" if trained else ""),
                        years, sharpe, str(c["batch"]) if trained else "-", f"{c['ridge']:g}" if att else "-",
                        f"{c['lambda_var']:g}" if att else "-", pca, pair(d["SR"], d["SR_net"]) if d else "",
                        stamp(f) if d else f"stopped; last year saved: {done[-1][8:12]}" if done else "stopped; no year saved",
                        f"`{g}/{p.name}`"])
    return table(["ID", "What it is", "Model", "Test years", "Sharpe ratio in the objective", "Batch, days", "Ridge penalty",
                  "Weight on explained variance", "PCA residuals", "SR before / after costs", "Finished (file time)",
                  "Folder under `runs/`"], out)


def report() -> None:
    rows = load()
    blocks = {"rounds": rounds(rows), "main": main_table(rows), "validation": validation_table(rows), "averages": averages(rows),
              "local": local_table()}
    s = OUT.read_text()
    for name, lines in blocks.items():
        a, b = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
        if lines is not None and a in s and b in s:
            s = s[:s.index(a) + len(a)] + "\n" + "\n".join(lines).rstrip() + "\n" + s[s.index(b):]
    OUT.write_text(s)
    print(f"updated the tables of {OUT.name}: {sum(not x['val'] for x in rows)} runs out of sample, {sum(x['val'] for x in rows)} on "
          f"validation" + ("" if blocks["local"] is None else f", {len(blocks['local']) - 3} under runs/ only"))


if __name__ == "__main__":
    report()
