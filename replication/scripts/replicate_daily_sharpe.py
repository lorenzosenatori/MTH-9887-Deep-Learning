"""Refit the attention models used in the slides with daily Sharpe in the objective.

Run with replication/.venv/bin/python replication/scripts/replicate_daily_sharpe.py.
Yearly checkpoints live under replication/runs/daily_sharpe/. The public results
contain only portfolio return series, aggregate metrics and their comparison.
"""
from __future__ import annotations

import argparse
try:
    import fcntl  # POSIX only: a lock keeps two copies of this script from training at once
except ImportError:  # Windows: run without the lock
    fcntl = None
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from daily_sharpe_comparison import write_comparison

ORIGINAL = ROOT / "results" / "runs"
RESULTS = ROOT / "results" / "daily_sharpe"
CASES = ("attention_K30_seed0", "attention_K30_seed1", "attention_K8_seed0")


def refit(case: str) -> None:
    import torch
    from attnfactors.train import Config, run

    torch.set_num_threads(2)
    old = json.loads((ORIGINAL / case / "metrics.json").read_text())["config"]
    cfg = Config(**old)
    cfg.sr_scale = 1.0
    cfg.name = f"daily_sharpe/{case}"
    if cfg.lambda_var != 100:
        raise ValueError(f"Unexpected variance weight for {case}")
    run(cfg)
    public = RESULTS / case
    public.mkdir(parents=True, exist_ok=True)
    for filename in ("daily.csv", "metrics.json"):
        tmp = public / f"{filename}.tmp"
        shutil.copyfile(ROOT / "runs" / cfg.name / filename, tmp)
        os.replace(tmp, public / filename)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-only", action="store_true", help="use the saved daily returns and metrics")
    args = ap.parse_args()
    if not args.report_only:
        checkpoint_dir = ROOT / "runs" / "daily_sharpe"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (checkpoint_dir / "training.lock").open("a") as lock:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_EX)
            with ProcessPoolExecutor(max_workers=len(CASES)) as pool:
                list(pool.map(refit, CASES))
    write_comparison(ORIGINAL, RESULTS, CASES)


if __name__ == "__main__":
    main()
