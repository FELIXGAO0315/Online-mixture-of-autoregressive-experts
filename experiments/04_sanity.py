"""Step 04 - sanity checks (project Step 3).

  (a) With the residual frozen at zero the mixture must reproduce Algorithm 1 exactly
      (every dataset, seed 0): max |gamma - w| and max |loss difference|.
  (b) On stationary AR(p0) data (T = inf) the mixture should match the best single expert and FPE
      selection within noise (3 seeds).
  (c) No look-ahead: perturb the raw series from step m* on, recompute everything (causal
      standardization -> experts -> every combiner) and check that all weights used up to m* and all
      losses before m* are bit-for-bit unchanged.

Outputs -> outputs/04_sanity/tables/
"""
import _bootstrap  # noqa: F401

import argparse

import numpy as np
import pandas as pd

from omar import config
from omar.data.datasets import load_dataset
from omar.data.standardize import standardize_online
from omar.experts.ar_bank import run_expert_bank
from omar.pipeline import ALL_ONLINE_METHODS, get_stream, load_hparams, parallel_map, run_job, run_method
from omar.reporting import Logger, mean_std, save_table
from omar.stream import ExpertStream

OUT = config.OUT_SANITY


def frozen_job(key: str, hp: dict) -> dict:
    ds, st = get_stream(key, 0, config.L_DEFAULT, hp["lam"])
    a = run_method("alg1", st, ds.T0, ds.eval_end, hp, 0)
    f = run_method("proposed_frozen", st, ds.T0, ds.eval_end, hp, 0)
    sl = slice(ds.T0, ds.eval_end)
    return dict(dataset=key, steps=ds.eval_end - ds.T0,
                max_abs_gamma_diff=float(np.abs(a.gamma[sl] - f.gamma[sl]).max()),
                max_abs_loss_diff=float(np.abs(a.loss[sl] - f.loss[sl]).max()))


def stationary_job(key: str, seed: int, hp: dict) -> dict:
    return run_job(key, seed, ["proposed", "alg1", "fpe"], hp)["rows"]


def lookahead_job(key: str, hp: dict, cut_frac: float = 0.5) -> list:
    ds = load_dataset(key, 0)
    L, lam = config.L_DEFAULT, hp["lam"]
    m_star = int(cut_frac * ds.N)
    rng = np.random.default_rng(7)
    X2 = ds.X_raw.copy()
    X2[m_star:] += 5.0 * rng.standard_normal(X2[m_star:].shape) * ds.X_raw.std(0)

    def stream(raw):
        Z = standardize_online(raw, ds.T0, "expanding")
        arr = run_expert_bank(Z, L, lam, config.LD_EPS, config.LD_TOL)
        return ExpertStream(Z, arr["Y"], arr["logdet"], [f"AR({p})" for p in range(1, L + 1)], L, lam,
                            arr["valid"], arr["coef1"])

    s1, s2 = stream(ds.X_raw), stream(X2)
    rows = [dict(dataset=key, method="experts (Y)", max_abs_diff_before=float(np.abs(s1.Y[:m_star + 1] - s2.Y[:m_star + 1]).max()),
                 changed_after=bool(np.abs(s1.Y[m_star + 1:] - s2.Y[m_star + 1:]).max() > 0))]
    for name in ALL_ONLINE_METHODS:
        r1 = run_method(name, s1, ds.T0, ds.N, hp, 0)
        r2 = run_method(name, s2, ds.T0, ds.N, hp, 0)
        dg = np.abs(r1.gamma[ds.T0:m_star + 1] - r2.gamma[ds.T0:m_star + 1]).max()
        dl = np.abs(r1.loss[ds.T0:m_star] - r2.loss[ds.T0:m_star]).max()
        rows.append(dict(dataset=key, method=name, max_abs_diff_before=float(max(dg, dl)),
                         changed_after=bool(np.nanmax(np.abs(r1.loss[m_star:] - r2.loss[m_star:])) > 0)))
    for r in rows:
        r["m_star"] = m_star
        r["pass"] = r["max_abs_diff_before"] == 0.0 and r["changed_after"]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    log = Logger(config.OUTPUTS / "logs" / "04_sanity.log")
    hps = load_hparams()

    # (a)
    res = parallel_map(frozen_job, [dict(key=k, hp=hps[k]) for k in config.DATASETS], args.jobs)
    df = pd.DataFrame(res)
    df["pass"] = (df.max_abs_gamma_diff < 1e-10) & (df.max_abs_loss_diff < 1e-10)
    save_table(df, OUT / "tables" / "a_frozen_residual_equals_alg1",
               "(a) Residual frozen at zero vs Algorithm 1 (seed 0, all steps after warm-up)")
    log("(a) frozen residual == Algorithm 1:\n" + df.to_string(index=False))

    # (b)
    jobs = [dict(key=k, seed=s, hp=hps[k]) for k in config.STATIONARY_KEYS for s in config.SEEDS]
    rows = pd.DataFrame([r for rr in parallel_map(stationary_job, jobs, args.jobs) for r in rr])
    out = []
    for k in config.STATIONARY_KEYS:
        d = rows[rows.dataset == k].pivot(index="seed", columns="method", values="cum_mse")
        for mth in ("proposed", "alg1", "fpe", "best_expert"):
            rel = 100 * (d[mth] / d["best_expert"] - 1)
            out.append(dict(dataset=k, method=mth, cum_mse=mean_std(d[mth]), rel_diff_vs_best_expert_pct=mean_std(rel, 3),
                            rel_diff_vs_fpe_pct=mean_std(100 * (d[mth] / d["fpe"] - 1), 3),
                            within_1pct_of_best=bool(np.abs(rel.mean()) < 1.0)))
    tab = pd.DataFrame(out)
    save_table(tab, OUT / "tables" / "b_stationary_matches_best_and_fpe",
               "(b) Stationary AR(4) data: mixture vs best single expert vs FPE (mean ± std over 3 seeds)",
               note="Criterion: mean relative difference to the best expert in hindsight below 1 %.")
    log("(b) stationary case:\n" + tab.to_string(index=False))

    # (c)
    keys = ["syn_n3_T500", "etth1"]
    rows = [r for rr in parallel_map(lookahead_job, [dict(key=k, hp=hps[k]) for k in keys], args.jobs) for r in rr]
    tab = pd.DataFrame(rows)
    save_table(tab, OUT / "tables" / "c_no_lookahead",
               "(c) No look-ahead: the raw series is perturbed from m* on; everything used up to m* must be identical",
               note="max_abs_diff_before = max |difference| of expert forecasts / weights up to m* and losses before m*.")
    log("(c) no look-ahead:\n" + tab.to_string(index=False))
    log(f"done -> {OUT}")


if __name__ == "__main__":
    main()
