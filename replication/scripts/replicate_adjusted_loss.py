"""Refit attention factors with a whole-window daily Sharpe and uncapped explained variance.

Run with replication/.venv/bin/python replication/scripts/replicate_adjusted_loss.py.
Only the variance floor and the time window used to evaluate the loss are varied.
"""
from __future__ import annotations

import argparse
try:
    import fcntl  # POSIX only: a lock keeps two copies of this script from training at once
except ImportError:  # Windows: run without the lock
    fcntl = None
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.checkpoint import checkpoint

from replicate_daily_sharpe import CASES, ROOT
from attnfactors.model import AttentionFactorModel, sharpe, trading_cost
from attnfactors.train import Config, Data, PANEL, apply_attention, att_forward, batches, performance

PUBLIC = ROOT / "results" / "adjusted_loss"
PRIVATE = ROOT / "runs" / "adjusted_loss"
CHUNK = 64  # memory boundary only for full-window training
VALIDATION = ("short_uncapped", "full_capped", "full_uncapped")


def variance_terms(eps, returns, valid):
    """Same eligibility and centered variance estimator as the original replication."""
    count = valid.sum(0).clamp(min=1)
    def variance(x):
        centered = x - (x * valid).sum(0) / count
        return (centered.square() * valid).sum(0) / count
    var_r = variance(returns)
    keep = (count >= 0.5 * len(valid)) & (var_r > 1e-10)
    if not keep.any():
        raise ValueError("No eligible stocks in the loss window")
    return 1 - variance(eps)[keep] / var_r[keep]


def loss(w, eps, returns, valid, rf, cfg, capped):
    net = (w * returns).sum(1)[1:] - trading_cost(w)
    terms = variance_terms(eps, returns, valid)
    if capped:
        terms = terms.clamp(min=-1)
    return -(sharpe(net, rf[1:]) + cfg.lambda_var * terms.mean())


def prepare_window(data, model, lo, hi, stats):
    """Keep stock identities consistent across chunks, including entries and exits."""
    union = data.member[lo-model.s:hi].any(0).nonzero().flatten()
    blocks = []
    for start in range(lo, hi, CHUNK):
        end = min(start + CHUNK, hi)
        begin = start - model.s
        cols = data.member[begin:end].any(0).nonzero().flatten()
        positions = torch.searchsorted(union, cols)
        blocks.append((positions, (data.x_firm(begin, end, cols), data.x_common(begin, end, stats),
                                    data.member[begin:end][:, cols], data.r_next[begin:end][:, cols],
                                    data.has_ret_next[begin:end][:, cols])))
    return blocks, data.r_next[lo:hi][:, union], (data.member[lo:hi] & data.has_ret_next[lo:hi])[:, union]


def full_loss(model, window, rf, cfg, capped):
    blocks, returns, valid = window
    def forward(*inputs):
        out = model(*inputs)
        return out["w"], out["eps"]
    weights, residuals = [], []
    for positions, inputs in blocks:
        # Replay the same dropout mask during backpropagation, keeping only one block's activations.
        w, eps = checkpoint(forward, *inputs, use_reentrant=False, preserve_rng_state=True)
        shape = (w.shape[0], returns.shape[1])
        weights.append(w.new_zeros(shape).index_copy(1, positions, w))
        residuals.append(eps.new_zeros(shape).index_copy(1, positions, eps))
    return loss(torch.cat(weights), torch.cat(residuals), returns, valid, rf, cfg, capped)


def train_year(data, cfg, year, spec):
    seed = cfg.seed * 100003 + year
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lo, hi = data.train_range(year, cfg.train_years)
    lo = max(lo, int((data.month_row >= 0).nonzero()[0]) + cfg.s + 1)
    stats = data.common_stats(lo, hi)
    model = AttentionFactorModel(data.m_firm, data.m_common, cfg.k, cfg.d, cfg.s, cfg.d_h,
                                 cfg.dropout, cfg.squash, cfg.ridge)
    opt = torch.optim.AdamW([{"params": model.factors.parameters(), "weight_decay": 0.0},
                             {"params": model.policy.parameters(), "weight_decay": cfg.weight_decay}], lr=cfg.lr)
    full = spec.startswith("full_")
    capped = spec == "full_capped"
    window = prepare_window(data, model, lo, hi, stats) if full else None
    steps = 0
    for epoch in range(cfg.epochs):
        ranges = [(lo, hi)] if full else batches(lo, hi, cfg.batch, rng)
        for a, b in ranges:
            opt.zero_grad(set_to_none=True)
            if full:
                value = full_loss(model, window, data.rf_next[a:b], cfg, capped)
            else:
                out, _ = att_forward(model, data, a, b, stats, True)
                value = loss(out["w"], out["eps"], out["r"], out["valid"], data.rf_next[a:b], cfg, capped)
            if not value.isfinite():
                raise ArithmeticError(f"Nonfinite loss for {year}, epoch {epoch + 1}")
            value.backward()
            model.factors.W.grad[model.factors.m_firm:] = 0
            opt.step()
            steps += 1
    a, b = data.test_range(year)
    w, diag = apply_attention(model, data, a, b, stats, cfg)
    if not np.isfinite(w).all() or not np.isfinite(list(diag.values())).all():
        raise ArithmeticError(f"Nonfinite portfolio or diagnostics for {year}")
    return dict(w=torch.from_numpy(w), model=model.state_dict(), diag=diag, steps=steps)


def atomic_write(path, writer):
    tmp = path.with_suffix(path.suffix + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


def refit(job):
    case, spec, validation = job
    torch.set_num_threads(2)
    source = (ROOT / "results" / "runs" / "val_att_K30_sr-per-day" if validation
              else ROOT / "results" / "daily_sharpe" / case)
    cfg = Config(**json.loads((source / "metrics.json").read_text())["config"])
    name = f"validation/{spec}" if validation else case
    cfg.name = f"adjusted_loss/{name}"
    settings = {**asdict(cfg), "loss_window": "full training window" if spec.startswith("full_") else "minibatch",
                "batch": None if spec.startswith("full_") else cfg.batch, "ev_floor": -1 if spec == "full_capped" else None,
                "min_valid_share": 0.5, "min_return_variance": 1e-10, "forward_chunk_days": CHUNK}
    if cfg.sr_scale != 1 or cfg.lambda_var != 100:
        raise ValueError("Expected daily Sharpe and variance weight 100")
    private = PRIVATE / name
    private.mkdir(parents=True, exist_ok=True)
    inputs = ["rank_monthly.npy", "rank_daily.npy", "common_raw.npy", "meta.npz", "feature_meta.npz",
              "ff.npy", "member.npy", "daily_ret.npy", "daily_prc.npy"]
    signature = {"settings": settings, "inputs": {f: (PANEL/f).stat().st_mtime_ns for f in inputs},
                 "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    config_file = private / "config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != signature:
        raise ValueError(f"Incompatible checkpoints in {private}")
    atomic_write(config_file, lambda p: p.write_text(json.dumps(signature, indent=2)))
    data = Data()
    weights, diagnostics, updates = [], [], {}
    for y in range(cfg.first_year, cfg.last_year + 1):
        path = private / f"{y}.pt"
        if path.exists():
            result = torch.load(path, weights_only=True)
        else:
            result = train_year(data, cfg, y, spec)
            atomic_write(path, lambda p: torch.save(result, p))
        weights.append(result["w"].numpy())
        diagnostics.append(result["diag"])
        updates[str(y)] = result["steps"]
        print(f"{name} {y} complete ({result['steps']} updates)", flush=True)
    lo = data.test_range(cfg.first_year)[0]
    hi = data.test_range(cfg.last_year)[1]
    metrics, series = performance(np.vstack(weights), data, lo, hi)
    metrics.update({k: float(np.mean([d[k] for d in diagnostics])) for k in diagnostics[0]})
    metrics.update(config=settings, optimizer_steps=updates)
    public = PUBLIC / name
    public.mkdir(parents=True, exist_ok=True)
    atomic_write(public / "daily.csv", lambda p: pd.DataFrame(series).to_csv(p, index=False))
    atomic_write(public / "metrics.json", lambda p: p.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.report_only:
        PRIVATE.mkdir(parents=True, exist_ok=True)
        with (PRIVATE / "training.lock").open("a") as lock:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_EX)
            with ProcessPoolExecutor(max_workers=3) as pool:
                list(pool.map(refit, [(CASES[0], spec, True) for spec in VALIDATION]))
                list(pool.map(refit, [(case, "full_uncapped", False) for case in CASES]))
    from adjusted_loss_comparison import report
    report()


if __name__ == "__main__":
    main()
