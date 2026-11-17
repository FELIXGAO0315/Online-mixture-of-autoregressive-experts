"""Step 05 - main experiments (project Step 4).

Every dataset x 3 seeds x every method on the same expert stream (selected lam), evaluated strictly
online on the evaluation window. Reported separately for synthetic n=1, synthetic n=3 and real data.

Outputs -> outputs/05_main/
  tables/main_results_long.csv          every (dataset, seed, method) row with all metrics
  tables/main_synthetic_n1.{csv,md}     mean ± std over seeds: cum. MSE and the three regrets
  tables/main_synthetic_n3.{csv,md}
  tables/main_real.{csv,md}
  tables/relative_to_best_expert.{csv,md}   cum. MSE relative to the best single expert (%)
  tables/readaptation.{csv,md}          gate re-adaptation time after regime switches
  figures/gamma_<dataset>.png           gate weights over time vs the true order (seed 0)
  figures/gamma_zoom_<dataset>.png      same, zoomed on the first regimes
  figures/drift_tracking.png            drifting AR(1): coefficient tracking and gate weights
  figures/error_curves_<group>.png      decayed running average of the loss (and its excess over the best expert)
  figures/regret_curves_<group>.png     cumulative regret to the best fixed expert over time
  figures/switch_response.png           loss aligned at regime switches (mean over switches and seeds)
"""
import _bootstrap  # noqa: F401

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omar import config
from omar.analysis import mean_order, switch_aligned, switch_recovery
from omar.data.datasets import load_dataset
from omar.metrics import METRICS, decayed_average
from omar.pipeline import (ALL_ONLINE_METHODS, METHOD_LABELS, load_hparams, load_run, parallel_map, run_job,
                           save_run)
from omar.plotting import INK, INK2, moving_average, savefig, setup_style, stacked_gamma, style
from omar.reporting import Logger, mean_std, save_table, summarize

OUT = config.OUT_MAIN
RUNS = config.CACHE / "runs" / "main"
GROUPS = {
    "synthetic_n1": ["syn_n1_T500", "syn_n1_T2000", "syn_n1_Tinf", "drift_ar1"],
    "synthetic_n3": ["syn_n3_T500", "syn_n3_T2000", "syn_n3_Tinf"],
    "real": ["etth1", "sunspots"],
}
TABLE_METHODS = ["proposed", "alg1", "fpe", "rls", "best_expert", "uniform", "segment_oracle", "step_oracle"]
# drawing order: the proposed method last so that it is on top
CURVE_METHODS = ["uniform", "rls", "fpe", "best_expert", "alg1", "proposed"]
DIFF_BETA = 0.999  # smoother decay for the "excess over best expert" panels only
RECOVERY_METHODS = ["proposed", "alg1", "fpe"]


def run_path(key: str, seed: int):
    return RUNS / f"{key}_s{seed}.npz"


def main_job(key: str, seed: int, hp: dict) -> dict:
    keep = ("proposed", "alg1", "fpe", "alg1_learned")
    out = run_job(key, seed, ALL_ONLINE_METHODS, hp, keep_gamma=keep, tag="main")
    save_run(run_path(key, seed), out)
    rec = []
    if key in config.SWITCHING_KEYS:
        ds = load_dataset(key, seed)
        for name in RECOVERY_METHODS:
            for r in switch_recovery(out["gamma"][name], ds.true_order, ds.switch_points, ds.eval_start,
                                     ds.eval_end, config.L_DEFAULT):
                # cost of the switch: excess loss over the segment oracle in the first 200 steps
                s = r["switch"]
                excess = out["loss"][name][s:s + 200] - out["loss"]["segment_oracle"][s:s + 200]
                rec.append(dict(dataset=key, seed=seed, method=name, switch_cost_200=float(excess.sum()), **r))
    return dict(rows=out["rows"], recovery=rec, seconds=out["seconds"])


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------
def make_tables(rows: pd.DataFrame, rec: pd.DataFrame, log) -> None:
    rows.to_csv(OUT / "tables" / "main_results_long.csv", index=False)
    for group, keys in GROUPS.items():
        d = rows[rows.dataset.isin(keys) & rows.method.isin(TABLE_METHODS)].copy()
        d["dataset"] = pd.Categorical(d.dataset, keys, ordered=True)
        d["method"] = pd.Categorical(d.method, TABLE_METHODS, ordered=True)
        d = d.sort_values(["dataset", "method", "seed"])
        tab = summarize(d, ["dataset", "method"], list(METRICS))
        tab["method"] = tab["method"].map(lambda m: METHOD_LABELS.get(m, m))
        save_table(tab, OUT / "tables" / f"main_{group}",
                   f"Main results - {group.replace('_', ' ')} (mean ± std over {rows.seed.nunique()} seeds)",
                   note="Cum. MSE = mean per-step loss ||X_m - X_hat_m||^2 on the evaluation window (standardized "
                        "units). Regrets are cumulative sums over the window; negative = better than the reference. "
                        "Rows marked (ref.) are reference lines, not causal baselines.")
        log(f"\n{group}:\n" + tab.to_string(index=False))

    # compact summary: cum. MSE relative to the best expert in hindsight
    piv = rows.pivot_table(index=["dataset", "seed"], columns="method", values="cum_mse")
    rel = 100 * (piv.div(piv["best_expert"], axis=0) - 1)
    out = []
    for m in ["proposed", "alg1", "alg1_learned", "hedge", "plain_mlp", "fpe", "rls", "uniform", "segment_oracle"]:
        rec_ = {"method": METHOD_LABELS.get(m, m)}
        for k in config.DATASETS:
            rec_[k] = mean_std(rel.loc[k][m], 2)
        out.append(rec_)
    save_table(pd.DataFrame(out), OUT / "tables" / "relative_to_best_expert",
               "Cumulative MSE relative to the best single expert in hindsight (%, mean ± std over seeds; "
               "negative = the method beats the best expert)")

    # paired comparison proposed vs Algorithm 1
    d = 100 * (piv["proposed"] / piv["alg1"] - 1)
    pc = pd.DataFrame([dict(dataset=k, rel_diff_pct=mean_std(d.loc[k], 3), seeds_better=int((d.loc[k] < 0).sum()),
                            seeds=int(d.loc[k].shape[0])) for k in config.DATASETS])
    save_table(pc, OUT / "tables" / "proposed_vs_alg1",
               "Proposed vs Algorithm 1: relative cum. MSE difference (%) and number of seeds where it is better")
    log("\nproposed vs alg1:\n" + pc.to_string(index=False))

    if len(rec):
        agg = []
        for (k, m), g in rec.groupby(["dataset", "method"], sort=False):
            per_seed = g.groupby("seed").tau.mean()
            agg.append(dict(dataset=k, method=METHOD_LABELS.get(m, m), switches=len(g),
                            frac_recovered=float(g.tau.notna().mean()),
                            mean_tau=mean_std(per_seed, 1), median_tau=float(np.nanmedian(g.tau)),
                            mean_tau_up=float(np.nanmean(g[g.p_new > g.p_old].tau)),
                            mean_tau_down=float(np.nanmean(g[g.p_new < g.p_old].tau)),
                            switch_cost_200=mean_std(g.groupby("seed").switch_cost_200.mean(), 2)))
        tab = pd.DataFrame(agg)
        save_table(tab, OUT / "tables" / "readaptation",
                   "Re-adaptation after regime switches",
                   note="tau = steps until the gate's mean order sum_p gamma_p p crosses the midpoint between the old "
                        "and the new true order and stays there for 25 steps (NaN = never before the next switch); "
                        "mean_tau: mean ± std over seeds of the per-seed average; up/down = switches to a higher / lower "
                        "order. switch_cost_200 = excess loss over the segment oracle in the 200 steps after a switch.",
                   digits=1)
        rec.to_csv(OUT / "tables" / "readaptation_per_switch.csv", index=False)
        log("\nre-adaptation:\n" + tab.to_string(index=False))


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def fig_gamma(key: str, zoom: tuple | None = None) -> None:
    ds = load_dataset(key, 0)
    run = load_run(run_path(key, 0))
    s, e = ds.eval_start, ds.eval_end
    if zoom:
        s, e = max(s, zoom[0]), min(e, zoom[1])
    t = np.arange(s, e)
    names = run["expert_names"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1.1]))
    for ax, m in zip(axes[:2], ("proposed", "alg1")):
        stacked_gamma(ax, t, run["gamma"][m][s:e], names, legend=(m == "proposed"))
        ax.set_ylabel("gate weight")
        ax.set_title(f"{METHOD_LABELS[m]}: weights gamma_i over time", loc="left")
    ax = axes[2]
    ax.step(t, ds.true_order[s:e], where="post", color=INK, lw=2, label="true order")
    for m in ("fpe", "alg1", "proposed"):
        c, ls, lab = style(m)
        mo = mean_order(run["gamma"][m][s:e], config.L_DEFAULT)
        ax.plot(t, mo, color=c, ls=ls, lw=1.0 if m != "fpe" else 0.6, alpha=0.9 if m != "fpe" else 0.6,
                label=("FPE selected order" if m == "fpe" else f"{lab}: mean order"))
    ax.set_ylabel("AR order")
    ax.set_xlabel("time step m")
    ax.set_ylim(0.5, config.L_DEFAULT + 0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Gate-weighted mean order vs true order", loc="left")
    fig.suptitle(f"{ds.label}, seed 0", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    savefig(fig, OUT / "figures" / (f"gamma_zoom_{key}" if zoom else f"gamma_{key}"))


def fig_drift(hps: dict) -> None:
    from omar.pipeline import get_stream

    ds, st = get_stream("drift_ar1", 0, config.L_DEFAULT, hps["drift_ar1"]["lam"])
    run = load_run(run_path("drift_ar1", 0))
    s, e = ds.eval_start, ds.eval_end
    t = np.arange(s, e)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    ax = axes[0]
    ax.plot(t, ds.true_coef[s:e], color=INK, lw=2, label="true a_t")
    ax.plot(t, st.coef1[s:e, 0, 0], color=style("proposed")[0], lw=0.8,
            label=f"AR(1) expert estimate (lam={hps['drift_ar1']['lam']:g})")
    ax.set_ylabel("AR(1) coefficient")
    ax.legend(loc="lower left")
    ax.set_title("Coefficient tracking", loc="left")
    stacked_gamma(axes[1], t, run["gamma"]["proposed"][s:e], run["expert_names"])
    axes[1].set_ylabel("gate weight")
    axes[1].set_title(f"{METHOD_LABELS['proposed']}: weights", loc="left")
    ax = axes[2]
    for m in ("best_expert", "fpe", "alg1", "proposed"):
        c, ls, lab = style(m)
        ax.plot(t, decayed_average(run["loss"][m], s, e)[s:e], color=c, ls=ls, lw=1, label=lab)
    ax.set_ylabel("decayed avg. loss")
    ax.set_xlabel("time step m")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Decayed running average of the loss (beta=0.99)", loc="left")
    fig.tight_layout()
    savefig(fig, OUT / "figures" / "drift_tracking")


def fig_curves(group: str, keys: list) -> None:
    fig, axes = plt.subplots(2, len(keys), figsize=(3.6 * len(keys) + 2, 6), squeeze=False)
    for j, key in enumerate(keys):
        ds = load_dataset(key, 0)
        run = load_run(run_path(key, 0))
        s, e = ds.eval_start, ds.eval_end
        t = np.arange(s, e)
        best = decayed_average(run["loss"]["best_expert"], s, e, DIFF_BETA)[s:e]
        diffs = []
        for m in CURVE_METHODS:
            c, ls, lab = style(m)
            avg = decayed_average(run["loss"][m], s, e)[s:e]
            axes[0, j].plot(t, avg, color=c, ls=ls, lw=0.9, label=lab)
            if m != "best_expert":
                d = decayed_average(run["loss"][m], s, e, DIFF_BETA)[s:e] - best
                diffs.append(d)
                axes[1, j].plot(t, d, color=c, ls=ls, lw=0.9, label=lab)
        axes[1, j].axhline(0, color=style("best_expert")[0], lw=1.2)
        axes[0, j].set_title(ds.label, loc="left")
        axes[1, j].set_xlabel("time step m")
        lo, hi = np.nanpercentile(np.concatenate([d[200:] for d in diffs]), [0.5, 99.5])
        axes[1, j].set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    axes[0, 0].set_ylabel("decayed avg. loss (beta=0.99)")
    axes[1, 0].set_ylabel("excess over best expert\n(decayed avg., beta=0.999)")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"Error vs time ({group.replace('_', ' ')}, seed 0)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    savefig(fig, OUT / "figures" / f"error_curves_{group}")

    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys) + 2, 3.4), squeeze=False)
    for j, key in enumerate(keys):
        ds = load_dataset(key, 0)
        run = load_run(run_path(key, 0))
        s, e = ds.eval_start, ds.eval_end
        t = np.arange(s, e)
        for m in CURVE_METHODS + ["segment_oracle"]:
            if m == "best_expert":
                continue
            c, ls, lab = style(m)
            axes[0, j].plot(t, np.cumsum(run["loss"][m][s:e] - run["loss"]["best_expert"][s:e]), color=c, ls=ls,
                            lw=1, label=lab)
        axes[0, j].axhline(0, color=style("best_expert")[0], lw=1.2)
        axes[0, j].set_title(ds.label, loc="left")
        axes[0, j].set_xlabel("time step m")
    axes[0, 0].set_ylabel("cumulative regret to best expert")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    savefig(fig, OUT / "figures" / f"regret_curves_{group}")


def fig_switch_response() -> None:
    keys = config.SWITCHING_KEYS
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys) + 2, 3.4), squeeze=False, sharey=False)
    for j, key in enumerate(keys):
        ax = axes[0, j]
        for m in ["segment_oracle", "fpe", "alg1", "proposed"]:
            mats = []
            for seed in config.SEEDS:
                ds = load_dataset(key, seed)
                run = load_run(run_path(key, seed))
                offs, mat = switch_aligned(run["loss"][m], ds.switch_points, ds.eval_start, ds.eval_end, 50, 450)
                mats.append(mat)
            curve = moving_average(np.concatenate(mats).mean(0), 15)
            c, ls, lab = style(m)
            ax.plot(offs[10:-10], curve[10:-10], color=c, ls=ls, lw=1.2, label=lab)
        ax.axvline(0, color=INK2, lw=0.8, ls=":")
        ax.set_title(config.DATASETS[key]["label"], loc="left")
        ax.set_xlabel("steps after switch")
    axes[0, 0].set_ylabel("loss (mean over switches & seeds)")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    savefig(fig, OUT / "figures" / "switch_response")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--plots-only", action="store_true", help="re-draw figures/tables from cached runs")
    args = ap.parse_args()
    log = Logger(config.OUTPUTS / "logs" / "05_main.log")
    setup_style()
    hps = load_hparams()
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    if not args.plots_only:
        keys = sorted(config.DATASETS, key=lambda k: -config.DATASETS[k]["n"])
        jobs = [dict(key=k, seed=s, hp=hps[k]) for k in keys for s in config.SEEDS]
        log(f"running {len(jobs)} (dataset, seed) jobs x {len(ALL_ONLINE_METHODS)} methods")
        res = parallel_map(main_job, jobs, args.jobs, log)
        rows = pd.DataFrame([r for x in res for r in x["rows"]])
        rec = pd.DataFrame([r for x in res for r in x["recovery"]])
        rows.to_csv(OUT / "tables" / "main_results_long.csv", index=False)
        rec.to_csv(OUT / "tables" / "readaptation_per_switch.csv", index=False)
    else:
        rows = pd.read_csv(OUT / "tables" / "main_results_long.csv")
        rec = pd.read_csv(OUT / "tables" / "readaptation_per_switch.csv")
    make_tables(rows, rec, log)
    for key in config.SWITCHING_KEYS:
        fig_gamma(key)
    for key in ("syn_n1_T500", "syn_n3_T500"):
        fig_gamma(key, zoom=(0, 2600))
    fig_drift(hps)
    for group, keys in GROUPS.items():
        fig_curves(group, keys)
    fig_switch_response()
    log(f"done -> {OUT}")


if __name__ == "__main__":
    main()
