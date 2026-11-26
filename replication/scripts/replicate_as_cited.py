"""Third round of runs: where the paper's own sources settle a choice, take the choice from them.

The first round (scripts/run.py, the numbers on the slides) made its choices on validation data.
The second round (replicate_daily_sharpe.py, replicate_adjusted_loss.py) changed the loss. This
round changes four things together, each to what a source that the paper cites says or does:

  loss     the Sharpe ratio per day, as the paper prints it and as the public code of the
           predecessor paper computes it; batches of 125 consecutive days, the batch of that
           code; no lower bound on one security's explained-variance term.
  inputs   Beta, Variance and Resid_Var as the paper's source for the characteristics defines
           them (chars.build("cited")), in a second panel folder, data/panel_as_cited.
  PCA      residuals as the predecessor's code builds them: correlation matrix, loadings from a
           regression on 60 days (attnfactors/pca_regression.py).
  nothing  else: universe, costs, model, optimizer, rolling windows and seeds are unchanged.

Run from replication/:

    python scripts/replicate_as_cited.py build         # second panel folder and the new PCA residuals
    python scripts/replicate_as_cited.py validation    # 1996-1997, one change at a time
    python scripts/replicate_as_cited.py main          # 1998-2021, the eight runs of this round
    python scripts/replicate_as_cited.py report        # results/as_cited/comparison.md from saved files

Full output goes to runs/as_cited/ (not committed), the two small files of every run to
results/as_cited/ (committed). Nothing of the earlier rounds is overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PUBLIC = ROOT / "results" / "as_cited"
PRIVATE = "as_cited"                       # under runs/
CITED_PANEL = "panel_as_cited"
# files of data/panel that the characteristics and the inputs do not write: linked, never copied
LINKED = ["daily_*.npy", "monthly_*.npy", "member.npy", "meta.npz", "ff.npy", "pca_cols.npy", "pca_v.npy",
          "pca_eps_K*.npy", "pcareg*.npy"]


def case(kind, k, seed=0, batch=125, floor=None, chars="cited", pca="regression", sr_scale=1.0):
    return dict(kind=kind, k=k, seed=seed, batch=batch, ev_floor=floor, chars=chars, pca=pca if kind != "attention" else None,
                sr_scale=sr_scale)


# the runs of this round, 1998 to 2021
MAIN = {
    "pca_K30_seed0": case("pca", 30, 0), "pca_K30_seed1": case("pca", 30, 1), "pca_K8_seed0": case("pca", 8, 0),
    "ou_K8": case("ou", 8), "ou_K30": case("ou", 30),
    "attention_K30_seed0": case("attention", 30, 0), "attention_K30_seed1": case("attention", 30, 1),
    "attention_K8_seed0": case("attention", 8, 0),
}
# 1996 to 1997, six training years: each change alone, then all together
VALIDATION = {
    "att_b125_floor": case("attention", 30, batch=125, floor=-1.0, chars="reconstructed"),
    "att_b125_nofloor": case("attention", 30, batch=125, floor=None, chars="reconstructed"),
    "att_b64_floor_cited": case("attention", 30, batch=64, floor=-1.0, chars="cited"),
    "att_b125_nofloor_cited": case("attention", 30),
    "att_b125_nofloor_cited_annualized": case("attention", 30, sr_scale=15.874507866387544),
    "att_K8_b125_nofloor_cited": case("attention", 8),
    "pca_b64_regression": case("pca", 30, batch=64),
    "pca_b125_projection": case("pca", 30, pca="projection"),
    "pca_b125_regression": case("pca", 30),
    "pca_K8_b125_regression": case("pca", 8),
    "pca_b125_regression252": case("pca", 30, pca="regression252"),      # correlation PCA, loadings from all 252 days
    "ou_K8_regression": case("ou", 8), "ou_K30_regression": case("ou", 30),
}


def build(force: bool = False) -> None:
    """data/panel_as_cited: links to the unchanged arrays, characteristics with the cited definitions, inputs."""
    from attnfactors.panel import DATA
    src, dst = DATA / "panel", DATA / CITED_PANEL
    for window, pre in ((60, "pcareg"), (252, "pcareg252")):
        if not any(src.glob(f"{pre}_eps_K*.npy")):
            subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'src'); from attnfactors import "
                            f"pca_regression; pca_regression.build(loading_window={window})"], cwd=ROOT, check=True)
    dst.mkdir(exist_ok=True)
    for pattern in LINKED:
        for f in sorted(src.glob(pattern)):
            link = dst / f.name
            if not link.exists():
                link.symlink_to(os.path.relpath(f, dst))
    if (dst / "rank_monthly.npy").exists() and not force:
        print(f"data/{CITED_PANEL} holds its inputs already (use --force to rebuild them)")
        return
    code = ("import sys; sys.path.insert(0, 'src'); from attnfactors import chars, features; "
            "chars.build('cited'); features.build_inputs()")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, env={**os.environ, "ATTNFACTORS_PANEL": CITED_PANEL})


def run_case(name: str, validation: bool, threads: int) -> None:
    """One run, in this process. The panel folder is fixed by the environment before attnfactors is imported."""
    spec = (VALIDATION if validation else MAIN)[name]
    want = CITED_PANEL if spec["chars"] == "cited" else "panel"
    if os.environ.get("ATTNFACTORS_PANEL", "panel") != want:
        raise RuntimeError(f"{name} needs ATTNFACTORS_PANEL={want}")
    import torch
    from attnfactors.train import Config, Data, PcaData, run
    from attnfactors.pca_regression import PcaRegressionData
    torch.set_num_threads(threads)
    sub = f"validation/{name}" if validation else name
    cfg = Config(kind=spec["kind"], k=spec["k"], seed=spec["seed"], batch=spec["batch"], ev_floor=spec["ev_floor"],
                 sr_scale=spec["sr_scale"], name=f"{PRIVATE}/{sub}")
    if validation:
        cfg.first_year, cfg.last_year, cfg.train_years = 1996, 1997, 6
    data = Data()
    pd_ = None
    if spec["kind"] != "attention":
        pd_ = {"regression": lambda: PcaRegressionData(data), "regression252": lambda: PcaRegressionData(data, 252),
               "projection": lambda: PcaData(data)}[spec["pca"]]()
    run(cfg, data, pd_)
    out = PUBLIC / sub
    out.mkdir(parents=True, exist_ok=True)
    src = ROOT / "runs" / PRIVATE / sub
    metrics = json.loads((src / "metrics.json").read_text())
    metrics["as_cited"] = {"characteristics": spec["chars"], "pca_residuals": spec["pca"], "panel_folder": want}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    shutil.copyfile(src / "daily.csv", out / "daily.csv")


def run_many(names: list[str], validation: bool, workers: int, threads: int) -> None:
    table = VALIDATION if validation else MAIN

    def one(name: str) -> tuple[str, int]:
        env = {**os.environ, "ATTNFACTORS_PANEL": CITED_PANEL if table[name]["chars"] == "cited" else "panel"}
        cmd = [sys.executable, str(Path(__file__).resolve()), "case", "--name", name, "--threads", str(threads)]
        log = ROOT / "runs" / PRIVATE / ("validation" if validation else "") / f"job-{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            return name, subprocess.run(cmd + (["--validation"] if validation else []), cwd=ROOT, env=env, stdout=f,
                                        stderr=subprocess.STDOUT).returncode

    with ThreadPoolExecutor(workers) as pool:
        failed = [n for n, code in pool.map(one, names) if code != 0]
    if failed:
        raise SystemExit(f"failed: {failed} (see runs/{PRIVATE}/job-*.log)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "validation", "main", "case", "report"])
    ap.add_argument("--name", help="with the stage 'case': the run; with 'validation' or 'main': a comma-separated subset")
    ap.add_argument("--validation", action="store_true")
    ap.add_argument("--workers", type=int, default=3, help="runs in parallel")
    ap.add_argument("--threads", type=int, default=2, help="threads of one run")
    ap.add_argument("--force", action="store_true", help="with 'build': compute the inputs of the second panel again")
    a = ap.parse_args()
    if a.stage == "build":
        build(a.force)
    elif a.stage == "case":
        run_case(a.name, a.validation, a.threads)
    elif a.stage in ("validation", "main"):
        table = VALIDATION if a.stage == "validation" else MAIN
        names = a.name.split(",") if a.name else list(table)
        run_many(names, a.stage == "validation", a.workers, a.threads)
    if a.stage in ("validation", "main", "report"):
        from as_cited_comparison import report
        report()


if __name__ == "__main__":
    main()
