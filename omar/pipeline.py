"""Glue code shared by all experiment scripts.

* ``get_stream``   - dataset + cached expert forecasts  ->  ExpertStream
* ``run_method``   - one combiner by name (the method registry)
* ``run_job``      - all requested methods on one (dataset, seed), with metrics
* ``parallel_map`` - process-level parallelism over independent jobs
* hyperparameter I/O (``load_hparams`` / ``save_hparams``)
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import config
from .combiners.baselines import run_fpe, run_rls, run_uniform
from .combiners.fixed_share import run_fixed_share
from .combiners.learned_share import run_learned_share
from .combiners.neural_gate import run_neural_gate
from .data.datasets import Dataset, data_seed, load_dataset
from .experts.ar_bank import run_expert_bank
from .experts.simple_experts import exponential_smoothing, seasonal_naive
from .metrics import evaluate, reference_losses
from .stream import ExpertStream, RunResult

# --------------------------------------------------------------------------------------
# method registry
# --------------------------------------------------------------------------------------
METHOD_LABELS = {
    "proposed": "Alg. 1 + residual MLP (proposed)",
    "alg1": "Algorithm 1 (fixed share)",
    "fpe": "FPE order selection",
    "rls": "RLS stacking",
    "best_expert": "Best single expert (hindsight)",
    "uniform": "Uniform average (ref.)",
    "segment_oracle": "Segment oracle (ref.)",
    "step_oracle": "Per-step oracle (ref.)",
    "hedge": "Hedge (alpha=0)",
    "alg1_learned": "Alg. 1, learned (eta, alpha)",
    "plain_mlp": "Plain MLP gate (no backbone)",
}
LADDER_METHODS = ["hedge", "alg1", "alg1_learned", "proposed", "plain_mlp"]
ALL_ONLINE_METHODS = ["proposed", "alg1", "fpe", "rls", "uniform", "hedge", "alg1_learned", "plain_mlp"]
REFERENCES = ["best_expert", "segment_oracle", "step_oracle"]

GATE_KW_DEFAULT = dict(K=config.K_DEFAULT, feature_set=config.GATE_FEATURES, hidden=config.GATE_HIDDEN,
                       replay=config.GATE_REPLAY, top_k=None, betas=config.GATE_BETAS,
                       x_lags=config.GATE_X_LAGS)


def run_method(name: str, stream: ExpertStream, start: int, end: int, hp: dict, seed: int,
               gate_kw: dict | None = None) -> RunResult:
    """Run combiner ``name`` on ``stream`` over [start, end) with hyperparameters ``hp``."""
    kw = dict(GATE_KW_DEFAULT)
    kw.update(gate_kw or {})
    if name == "alg1":
        return run_fixed_share(stream, start, end, hp["eta"], hp["alpha"], top_k=kw["top_k"], name=name)
    if name == "hedge":
        return run_fixed_share(stream, start, end, hp["hedge_eta"], 0.0, name=name)
    if name == "alg1_learned":
        return run_learned_share(stream, start, end, hp["eta"], hp["alpha"], hp["learned_lr"], name=name)
    if name == "proposed":
        return run_neural_gate(stream, start, end, backbone=(hp["eta"], hp["alpha"]), lr=hp["lr"], wd=hp["wd"],
                               seed=seed, name=name, **kw)
    if name == "proposed_frozen":  # sanity check (a): residual frozen at zero
        return run_neural_gate(stream, start, end, backbone=(hp["eta"], hp["alpha"]), lr=0.0, wd=0.0,
                               seed=seed, name=name, train=False, **kw)
    if name == "plain_mlp":
        return run_neural_gate(stream, start, end, backbone=None, lr=hp["plain_lr"], wd=hp["plain_wd"],
                               seed=seed, name=name, **kw)
    if name == "fpe":
        return run_fpe(stream, start, end, name=name)
    if name == "rls":
        return run_rls(stream, start, end, name=name)
    if name == "uniform":
        return run_uniform(stream, start, end, name=name)
    raise ValueError(f"unknown method {name!r}")


# --------------------------------------------------------------------------------------
# expert streams (cached on disk: experts do not depend on the combiner)
# --------------------------------------------------------------------------------------
def _stream_cache_path(key: str, dseed: int, L: int, lam: float, eps: float) -> Path:
    return config.CACHE / "experts" / f"{key}_d{dseed}_L{L}_lam{lam:g}_eps{eps:g}.npz"


def get_stream(key: str, seed: int, L: int = config.L_DEFAULT, lam: float = 0.99, extras: list | None = None,
               eps: float = config.LD_EPS, eval_start: int | None = None,
               use_cache: bool = True) -> tuple[Dataset, ExpertStream]:
    """Load dataset ``key`` and the forecasts of its L AR experts (+ optional extra experts).

    extras: list of ("snaive", lag) / ("ses", smoothing) non-AR experts appended after the AR ones.
    """
    ds = load_dataset(key, seed, L, eval_start=eval_start)
    path = _stream_cache_path(key, data_seed(key, seed), L, lam, eps)
    if use_cache and path.exists():
        with np.load(path) as d:
            arr = {k: d[k] for k in d.files}
    else:
        arr = run_expert_bank(ds.X, L, lam, eps, config.LD_TOL)
        arr = {k: arr[k] for k in ("Y", "logdet", "valid", "refl", "coef1")}
        if use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
            np.savez(tmp, **arr)
            os.replace(tmp, path)
    Y = arr["Y"]
    names = [f"AR({p})" for p in range(1, L + 1)]
    for kind, param in extras or []:
        if kind == "snaive":
            Ye, nm = seasonal_naive(ds.X, int(param)), f"SNaive({int(param)})"
        elif kind == "ses":
            Ye, nm = exponential_smoothing(ds.X, float(param)), f"SES({float(param):g})"
        else:
            raise ValueError(kind)
        Y = np.concatenate([Y, Ye[:, None, :]], axis=1)
        names.append(nm)
    stream = ExpertStream(X=ds.X, Y=Y, logdet=arr["logdet"], names=names, n_ar=L, lam=lam,
                          valid=arr["valid"], coef1=arr["coef1"])
    return ds, stream


# --------------------------------------------------------------------------------------
# one (dataset, seed) job
# --------------------------------------------------------------------------------------
def run_job(key: str, seed: int, methods: list, hp: dict, L: int = config.L_DEFAULT, lam: float | None = None,
            extras: list | None = None, gate_kw: dict | None = None, method_hp: dict | None = None,
            method_gate_kw: dict | None = None, eval_start: int | None = None, keep_gamma: tuple = (),
            tag: str = "") -> dict:
    """Run ``methods`` on one stream and evaluate them on the evaluation window.

    The online pass starts right after warm-up (m = T0) for every method; on real data the tuning
    part of the series is therefore processed online but not scored.

    A method name may carry an alias suffix, e.g. ``"proposed@K=10"``: the part before ``@`` selects
    the implementation, and ``method_hp`` / ``method_gate_kw`` (keyed by the full name) override
    hyperparameters / gate options for that entry; ``gate_kw`` applies to every neural gate.
    Returns ``rows`` (metrics), ``loss`` (per-step losses of methods and references) and ``gamma``
    (weights of the methods listed in ``keep_gamma``).
    """
    t0 = time.time()
    lam = hp["lam"] if lam is None else lam
    ds, st = get_stream(key, seed, L, lam, extras, eval_start=eval_start)
    start, end = ds.eval_start, ds.eval_end
    losses, gammas, extra = {}, {}, {}
    for name in methods:
        base = name.split("@")[0]
        h = {**hp, **(method_hp or {}).get(name, {})}
        gk = {**(gate_kw or {}), **(method_gate_kw or {}).get(name, {})}
        res = run_method(base, st, ds.T0, end, h, seed, gate_kw=gk)
        losses[name] = res.loss
        if name in keep_gamma:
            gammas[name] = res.gamma
        for k, v in res.extra.items():
            extra[f"{name}:{k}"] = v
    refs = reference_losses(st.E, ds.segments, start, end)
    rows = []
    common = dict(dataset=key, seed=seed, tag=tag, L=L, lam=lam, M=st.M)
    for name, loss in losses.items():
        rows.append(dict(common, method=name, **evaluate(loss, st.E, ds.segments, start, end)))
    for name in REFERENCES:
        rows.append(dict(common, method=name, **evaluate(refs[name], st.E, ds.segments, start, end)))
    for name in REFERENCES:
        losses[name] = refs[name]
    return dict(rows=rows, loss=losses, gamma=gammas, extra=extra, best_index=refs["_best_index"],
                expert_names=st.names, seconds=time.time() - t0)


def save_run(path: Path, out: dict) -> None:
    """Persist the per-step arrays of a ``run_job`` result (losses, kept weights, extras)."""
    arrays = {f"loss__{k}": v for k, v in out["loss"].items()}
    arrays.update({f"gamma__{k}": v for k, v in out["gamma"].items()})
    arrays.update({f"extra__{k.replace(':', '__')}": v for k, v in out["extra"].items()})
    arrays["best_index"] = np.array(out["best_index"])
    arrays["expert_names"] = np.array(out["expert_names"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_run(path: Path) -> dict:
    out = dict(loss={}, gamma={}, extra={})
    with np.load(path) as d:
        for k in d.files:
            if k.startswith("loss__"):
                out["loss"][k[6:]] = d[k]
            elif k.startswith("gamma__"):
                out["gamma"][k[7:]] = d[k]
            elif k.startswith("extra__"):
                out["extra"][k[7:]] = d[k]
        out["best_index"] = int(d["best_index"])
        out["expert_names"] = [str(s) for s in d["expert_names"]]
    return out


# --------------------------------------------------------------------------------------
# hyperparameters
# --------------------------------------------------------------------------------------
HPARAM_FILE = config.OUT_TUNING / "hparams.json"


def save_hparams(hp: dict) -> None:
    HPARAM_FILE.parent.mkdir(parents=True, exist_ok=True)
    HPARAM_FILE.write_text(json.dumps(hp, indent=2))


def load_hparams() -> dict:
    if not HPARAM_FILE.exists():
        raise FileNotFoundError(f"{HPARAM_FILE} not found - run experiments/03_tune.py first")
    return json.loads(HPARAM_FILE.read_text())


# --------------------------------------------------------------------------------------
# parallel execution
# --------------------------------------------------------------------------------------
def _worker_init() -> None:
    import torch

    torch.set_num_threads(1)


def default_jobs() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def parallel_map(fn, jobs: list[dict], n_jobs: int | None = None, log=None) -> list:
    """Call ``fn(**job)`` for every job (in parallel processes when n_jobs > 1); results keep job order."""
    n_jobs = default_jobs() if n_jobs is None else n_jobs
    t0 = time.time()
    if n_jobs <= 1 or len(jobs) <= 1:
        _worker_init()
        out = []
        for i, job in enumerate(jobs):
            out.append(fn(**job))
            if log:
                log(f"  [{i + 1}/{len(jobs)}] done ({time.time() - t0:.0f}s)")
        return out
    out = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_worker_init) as ex:
        futs = {ex.submit(fn, **job): i for i, job in enumerate(jobs)}
        done = 0
        from concurrent.futures import as_completed

        for fut in as_completed(futs):
            out[futs[fut]] = fut.result()
            done += 1
            if log and (done % max(1, len(jobs) // 10) == 0 or done == len(jobs)):
                log(f"  [{done}/{len(jobs)}] jobs done ({time.time() - t0:.0f}s)")
    return out
