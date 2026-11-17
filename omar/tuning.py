"""Hyperparameter tuning (protocol Sec. 1.5).

Tuning data: a *separate* synthetic stream (seed ``config.TUNE_SEED``) for synthetic datasets, and
the first 20 % of each real series. The evaluation streams / the remaining 80 % are never touched.

Stage A  - for every forgetting factor lam: expert forecasts, then Algorithm 1 on the whole
           (eta, alpha) grid (vectorized). Select (lam, eta, alpha) jointly by the cumulative
           MSE of Algorithm 1. Hedge's eta = best eta with alpha = 0 at the selected lam.
           All methods then share the experts of the selected lam.
Stage B  - at the selected lam: residual gate (lr, wd), plain MLP gate (lr, wd), learned
           (eta, alpha) Adam lr, and the smoothing constant of the SES expert (ablation c).
"""
from __future__ import annotations

import numpy as np

from . import config
from .combiners.fixed_share import fixed_share_grid
from .data.datasets import load_dataset
from .experts.simple_experts import exponential_smoothing
from .pipeline import get_stream, run_method


def tuning_seed(key: str) -> int:
    return config.TUNE_SEED if config.DATASETS[key]["kind"] in ("regime", "drift") else 0


def stage_a_job(key: str, lam: float, etas, alphas) -> dict:
    ds, st = get_stream(key, tuning_seed(key), config.L_DEFAULT, lam)
    grid = fixed_share_grid(st, ds.tune_start, ds.tune_end, etas, alphas)
    return dict(key=key, lam=lam, grid=grid)


def stage_b_job(key: str, lam: float, eta: float, alpha: float, method: str, params: dict) -> dict:
    seed = tuning_seed(key)
    ds, st = get_stream(key, seed, config.L_DEFAULT, lam)
    hp = dict(lam=lam, eta=eta, alpha=alpha, hedge_eta=eta, lr=1e-3, wd=0.1, plain_lr=1e-3, plain_wd=0.1,
              learned_lr=3e-3)
    hp.update(params)
    res = run_method(method, st, ds.T0, ds.tune_end, hp, seed)
    loss = float(np.mean(res.loss[ds.tune_start:ds.tune_end]))
    return dict(key=key, method=method, loss=loss, **params)


def tune_ses(key: str, alphas) -> tuple[float, list]:
    ds = load_dataset(key, tuning_seed(key))
    s, e = ds.tune_start, ds.tune_end
    rows = []
    for a in alphas:
        Y = exponential_smoothing(ds.X, a)
        rows.append(dict(key=key, ses_alpha=a, loss=float(((ds.X[s:e] - Y[s:e]) ** 2).sum(1).mean())))
    best = min(rows, key=lambda r: r["loss"])["ses_alpha"]
    return best, rows
