"""Train and evaluate one configuration over the rolling windows.

Examples (run from replication/):
    python scripts/run.py attention --k 30
    python scripts/run.py pca --k 30
    python scripts/run.py ou --k 3
    python scripts/run.py attention --k 30 --drop past_returns        # a row of Table 3
    python scripts/run.py attention --k 8 --first-year 1996 --last-year 1997 --train-years 6 --name val_K8
    python scripts/run.py pca --k 30 --lookback current          # variants of open choices (README)
    python scripts/run.py pca --k 30 --pca corr                   # after: python scripts/build.py pca_corr
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attnfactors.train import Config, run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["attention", "pca", "ou"])
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--first-year", type=int, default=1998)
    ap.add_argument("--last-year", type=int, default=2021)
    ap.add_argument("--train-years", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ridge", type=float, default=1e-5)
    ap.add_argument("--lambda-var", type=float, default=100.0)
    ap.add_argument("--sr-scale", type=float, default=15.874507866387544, help="scale of the Sharpe ratio in the objective; 1 = per day")
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--no-common", action="store_true", help="leave out the 40 inputs that are equal for all securities")
    ap.add_argument("--drop", nargs="*", default=[], help="characteristic groups to leave out")
    ap.add_argument("--pca", choices=["cov", "corr"], default="cov",
                    help="PCA benchmarks: covariance (default) or correlation matrix with Avellaneda-Lee eigenportfolios "
                         "(needs scripts/build.py pca_corr)")
    ap.add_argument("--lookback", choices=["own", "current"], default="own",
                    help="residual history of the signal: each day's own factor model (default) or the decision date's")
    ap.add_argument("--name", default="")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = Config(kind=a.kind, k=a.k, seed=a.seed, first_year=a.first_year, last_year=a.last_year, train_years=a.train_years,
                 epochs=a.epochs, batch=a.batch, ridge=a.ridge, lambda_var=a.lambda_var, sr_scale=a.sr_scale, lr=a.lr,
                 use_common=not a.no_common, drop_groups=a.drop, pca=a.pca, lookback=a.lookback, name=a.name)
    run(cfg)


if __name__ == "__main__":
    main()
