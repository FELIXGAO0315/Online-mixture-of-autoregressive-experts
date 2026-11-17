"""Step 06 - ablations and discussion material (project Step 5).

  (a) vary L in {4, 8, 12}, lam, the error window K, the gate input features
      (last-step errors / window / window + EWMA / + recent X / + log det Sigma_p) and the replay buffer
  (b) the learning ladder: Hedge -> Algorithm 1 -> Algorithm 1 with (eta, alpha) learned online ->
      Algorithm 1 + residual MLP (proposed) -> plain MLP gate (read from the main run, all 9 datasets);
      sweep of the residual's learning rate x weight decay; top-k sparse gating
  (c) heterogeneous experts: + seasonal naive (lag 24, ETTh1) and simple exponential smoothing
  (d) when does the learned mixture beat the best single AR expert?

All other hyperparameters stay at their tuned values. Ablations (a)-(c) run on
config.ABLATION_KEYS x 3 seeds. Step 05 must have been run first (the ladder and (d) reuse its results).

Outputs -> outputs/06_ablation/{tables,figures}
"""
import _bootstrap  # noqa: F401

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omar import config
from omar.pipeline import LADDER_METHODS, METHOD_LABELS, get_stream, load_hparams, parallel_map, run_job
from omar.plotting import DIVERGING, INK, INK2, MUTED, savefig, setup_style, style
from omar.reporting import Logger, fmt_num, mean_std, save_table

OUT = config.OUT_ABLATION
SYN_EVAL_START = 5 * 12  # common evaluation start for every L in the L ablation (T0 of L = 12)

L_VALUES = (4, 8, 12)
K_VALUES = (1, 3, 5, 10, 20)
FEATURE_SETS = ("last", "window", "window_ewma", "window_ewma_x", "window_ewma_logdet")
REPLAY_VALUES = (1, 8, 32)
SWEEP_LR = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
SWEEP_WD = (0.0, 0.01, 0.1, 1.0)
TOPK_VALUES = (1, 2, 3)


# --------------------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------------------
def stream_job(key: str, seed: int, L: int, lam: float) -> None:
    get_stream(key, seed, L, lam)  # computes and caches the expert forecasts


def abl_job(ablation: str, setting: str, key: str, seed: int, hp: dict, methods: list, nonar_weight: bool = False,
            **kw) -> list:
    keep = tuple(m for m in methods if m.split("@")[0] in ("proposed", "alg1")) if nonar_weight else ()
    out = run_job(key, seed, methods, hp, tag=ablation, keep_gamma=keep, **kw)
    names = out["expert_names"]
    rows = []
    for r in out["rows"]:
        base, _, alias = r["method"].partition("@")
        row = dict(r, ablation=ablation, job_setting=setting, setting=alias or setting, method=base,
                   best_expert_name=names[out["best_index"]])
        if nonar_weight and r["method"] in out["gamma"]:
            g = out["gamma"][r["method"]]
            n_ar = sum(nm.startswith("AR(") for nm in names)
            s0 = int(np.flatnonzero(~np.isnan(out["loss"]["best_expert"]))[0])
            row["mean_weight_nonar"] = float(np.nanmean(g[s0:, n_ar:].sum(1)))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------------------
def with_rel_to_alg1(df: pd.DataFrame) -> pd.DataFrame:
    """Add rel_vs_alg1_pct: cum. MSE relative to Algorithm 1 (no alias) of the same dataset/seed/job setting."""
    ref = df[(df.method == "alg1") & (df.setting == df.job_setting)][["dataset", "seed", "job_setting", "cum_mse"]]
    ref = ref.rename(columns={"cum_mse": "alg1_cum_mse"})
    d = df.merge(ref, on=["dataset", "seed", "job_setting"], how="left")
    d["rel_vs_alg1_pct"] = 100 * (d.cum_mse / d.alg1_cum_mse - 1)
    return d


def pivot_table(d: pd.DataFrame, methods: list, value: str = "cum_mse", digits: int = 4,
                setting_order: list | None = None) -> pd.DataFrame:
    out = []
    keys = [k for k in config.DATASETS if k in set(d.dataset)]
    settings = setting_order or list(dict.fromkeys(d.setting))
    for k in keys:
        for s in settings:
            g = d[(d.dataset == k) & (d.setting == s)]
            if g.empty:
                continue
            rec = dict(dataset=k, setting=s)
            for m in methods:
                v = g[g.method == m][value]
                if len(v):
                    rec[METHOD_LABELS.get(m, m)] = mean_std(v, digits)
            out.append(rec)
    return pd.DataFrame(out)


def save_pair(d: pd.DataFrame, name: str, title: str, methods: list, setting_order=None, note=None) -> None:
    save_table(pivot_table(d, methods, "cum_mse", 4, setting_order), OUT / "tables" / f"{name}_cum_mse",
               f"{title}: cumulative MSE (mean ± std over seeds)", note=note)
    rel_methods = [m for m in methods if m not in ("best_expert", "segment_oracle", "step_oracle")]
    save_table(pivot_table(d, rel_methods, "rel_vs_alg1_pct", 3, setting_order), OUT / "tables" / f"{name}_rel_alg1",
               f"{title}: cum. MSE relative to Algorithm 1 at the same setting (%, negative = better)", note=note)


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def fig_ladder(main: pd.DataFrame) -> None:
    keys = list(config.DATASETS)
    piv = main.pivot_table(index=["dataset", "seed"], columns="method", values="cum_mse")
    rel = 100 * (piv.div(piv["alg1"], axis=0) - 1)
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.5))
    x = np.arange(len(LADDER_METHODS))
    for ax, k in zip(axes.ravel(), keys):
        r = rel.loc[k]
        for i, m in enumerate(LADDER_METHODS):
            c = style(m)[0]
            ax.errorbar([i], [r[m].mean()], yerr=[r[m].std(ddof=1)], fmt="o", ms=7, color=c, capsize=3, lw=1.5)
            ax.scatter(np.full(len(r), i) + 0.18, r[m], s=10, color=c, alpha=0.6)
        ax.axhline(0, color=INK2, lw=0.8, ls=":")
        ax.set_xticks(x, ["Hedge", "Alg. 1", "Alg. 1\nlearned", "Alg. 1 +\nresidual", "plain\nMLP"], fontsize=7)
        ax.set_title(config.DATASETS[k]["label"], loc="left", fontsize=9)
        lo, hi = np.nanmin(r[LADDER_METHODS].values), np.nanmax(r[LADDER_METHODS].values)
        pad = 0.1 * max(hi - lo, 0.05)
        ax.set_ylim(lo - pad, hi + pad)
    for ax in axes[:, 0]:
        ax.set_ylabel("cum. MSE vs Alg. 1 (%)")
    fig.suptitle("Learning ladder: mean ± std over seeds (dots = individual seeds)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    savefig(fig, OUT / "figures" / "learning_ladder")


def fig_sweep(d: pd.DataFrame) -> None:
    keys = [k for k in config.DATASETS if k in set(d.dataset)]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.3 * len(keys) + 1, 3.4), squeeze=False)
    vmax = 1.0
    for ax, k in zip(axes[0], keys):
        g = d[(d.dataset == k) & (d.method == "proposed")]
        M = np.full((len(SWEEP_LR), len(SWEEP_WD)), np.nan)
        for i, lr in enumerate(SWEEP_LR):
            for j, wd in enumerate(SWEEP_WD):
                v = g[(g.setting == f"lr={lr:g},wd={wd:g}")].rel_vs_alg1_pct
                M[i, j] = v.mean() if len(v) else np.nan
        im = ax.imshow(np.clip(M, -vmax, vmax), cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if abs(M[i, j]) > 0.6 * vmax else INK)
        ax.set_xticks(range(len(SWEEP_WD)), [f"{w:g}" for w in SWEEP_WD])
        ax.set_yticks(range(len(SWEEP_LR)), [f"{lr:g}" for lr in SWEEP_LR])
        ax.set_xlabel("weight decay")
        ax.set_title(config.DATASETS[k]["label"], loc="left", fontsize=9)
        ax.grid(False)
    axes[0, 0].set_ylabel("learning rate")
    fig.colorbar(im, ax=axes, shrink=0.8, label="cum. MSE vs Alg. 1 (%), clipped at ±1")
    savefig(fig, OUT / "figures" / "lr_wd_sweep")


def fig_lambda(d: pd.DataFrame) -> None:
    keys = [k for k in config.DATASETS if k in set(d.dataset)]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys) + 2, 3.4), squeeze=False)
    for ax, k in zip(axes[0], keys):
        for m in ("best_expert", "rls", "fpe", "alg1", "proposed"):
            g = d[(d.dataset == k) & (d.method == m)].groupby("lam").cum_mse.mean()
            c, ls, lab = style(m)
            ax.plot(1.0 / (1.0 - g.index.values), g.values, color=c, ls=ls, marker="o", ms=4, label=lab)
        ax.set_xscale("log")
        ax.set_xlabel("N_eff = 1/(1-lam)")
        ax.set_title(config.DATASETS[k]["label"], loc="left", fontsize=9)
    axes[0, 0].set_ylabel("cum. MSE (mean over seeds)")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    savefig(fig, OUT / "figures" / "lambda_ablation")


def characterization(main: pd.DataFrame, log) -> None:
    piv = main.pivot_table(index=["dataset", "seed"], columns="method", values="cum_mse")
    rows = []
    for k in config.DATASETS:
        p = piv.loc[k]
        rows.append(dict(
            dataset=k, n=config.DATASETS[k]["n"],
            switching_potential_pct=mean_std(100 * (1 - p.segment_oracle / p.best_expert), 2),
            uniform_vs_best_pct=mean_std(100 * (p.uniform / p.best_expert - 1), 2),
            alg1_vs_best_pct=mean_std(100 * (p.alg1 / p.best_expert - 1), 2),
            proposed_vs_best_pct=mean_std(100 * (p.proposed / p.best_expert - 1), 2),
            proposed_vs_alg1_pct=mean_std(100 * (p.proposed / p.alg1 - 1), 3),
            rls_vs_best_pct=mean_std(100 * (p.rls / p.best_expert - 1), 2),
            winner=min(["proposed", "alg1", "fpe", "rls", "best_expert"], key=lambda m: p[m].mean()),
        ))
    tab = pd.DataFrame(rows)
    save_table(tab, OUT / "tables" / "d_characterization",
               "(d) When does the learned mixture beat the best single expert?",
               note="switching_potential = 1 - (segment oracle / best fixed expert): how much a method that tracks the "
                    "best expert per segment could gain. *_vs_best = cum. MSE relative to the best single expert (%). "
                    "winner = lowest mean cum. MSE among the causal methods and the best expert in hindsight.")
    log("(d) characterization:\n" + tab.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m in ("alg1", "proposed"):
        xs, ys = [], []
        for k in config.DATASETS:
            p = piv.loc[k]
            xs.append(100 * (1 - p.segment_oracle / p.best_expert).mean())
            ys.append(100 * (1 - p[m] / p.best_expert).mean())
        c, _, lab = style(m)
        ax.scatter(xs, ys, s=50 if m == "proposed" else 80, color=c, label=lab, zorder=3 if m == "proposed" else 2,
                   edgecolor="#fcfcfb", linewidth=1.5)
        if m == "proposed":
            offsets = {"etth1": (6, 4), "drift_ar1": (-8, -13), "syn_n1_Tinf": (6, 4), "syn_n3_Tinf": (6, -12)}
            for x, y, k in zip(xs, ys, config.DATASETS):
                ax.annotate(k, (x, y), xytext=offsets.get(k, (6, -10)), textcoords="offset points", fontsize=7,
                            color=INK2, ha="right" if offsets.get(k, (6,))[0] < 0 else "left")
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], color=MUTED, ls="--", lw=1, label="gain = switching potential")
    ax.axhline(0, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("switching potential: 1 - segment oracle / best expert (%)")
    ax.set_ylabel("gain over best single expert (%)")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("Mixtures gain where the best expert changes over time", loc="left")
    fig.tight_layout()
    savefig(fig, OUT / "figures" / "d_gain_vs_switching_potential")


def ladder_table(main: pd.DataFrame, log) -> None:
    piv = main.pivot_table(index=["dataset", "seed"], columns="method", values="cum_mse")
    rows = []
    for k in config.DATASETS:
        p = piv.loc[k]
        rec = dict(dataset=k)
        for m in LADDER_METHODS:
            rec[METHOD_LABELS[m]] = mean_std(p[m], 4)
        for a, b in zip(LADDER_METHODS[:-1], LADDER_METHODS[1:]):
            rec[f"step {a}->{b} (%)"] = fmt_num(100 * (p[b] / p[a] - 1).mean(), 3)
        rows.append(rec)
    tab = pd.DataFrame(rows)
    save_table(tab, OUT / "tables" / "b_learning_ladder",
               "(b) Learning ladder: cum. MSE (mean ± std over seeds) and what each step buys (%; negative = better)",
               note="The std column shows the cost in variance across seeds: Hedge, Algorithm 1 and FPE are "
                    "deterministic given the data, the learned levels also depend on the gate initialization.")
    log("(b) ladder:\n" + tab.to_string(index=False))


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="1 seed (smoke test)")
    args = ap.parse_args()
    log = Logger(config.OUTPUTS / "logs" / "06_ablation.log")
    setup_style()
    hps = load_hparams()
    seeds = (0,) if args.quick else config.SEEDS
    keys = config.ABLATION_KEYS
    lams = config.GRID["lam"]
    main_path = config.OUT_MAIN / "tables" / "main_results_long.csv"
    if not main_path.exists():
        raise FileNotFoundError("run experiments/05_main.py first")
    main_rows = pd.read_csv(main_path)

    # ---- (b) ladder and (d) characterization: from the main run ----
    ladder_table(main_rows, log)
    fig_ladder(main_rows)
    characterization(main_rows, log)

    # ---- precompute the expert streams needed below (avoids duplicated work in parallel jobs) ----
    specs = {(k, s if config.DATASETS[k]["kind"] in ("regime", "drift") else 0, L, lam)
             for k in keys for s in seeds for (L, lam) in
             [(L, hps[k]["lam"]) for L in L_VALUES] + [(config.L_DEFAULT, lam) for lam in lams]}
    log(f"precomputing {len(specs)} expert streams")
    parallel_map(stream_job, [dict(key=k, seed=s, L=L, lam=lam) for k, s, L, lam in sorted(specs)], args.jobs, log)

    base4 = ["proposed", "alg1", "fpe", "rls"]
    jobs = []
    for k in keys:
        hp = hps[k]
        for s in seeds:
            common = dict(key=k, seed=s, hp=hp)
            for L in L_VALUES:  # (a) number of experts
                jobs.append(dict(common, ablation="L", setting=f"L={L}", methods=base4, L=L,
                                 eval_start=SYN_EVAL_START))
            for lam in lams:  # (a) forgetting factor
                jobs.append(dict(common, ablation="lam", setting=f"lam={lam:g}", methods=base4, lam=lam))
            jobs.append(dict(common, ablation="K", setting="", methods=["alg1"] + [f"proposed@K={K}" for K in K_VALUES],
                             method_gate_kw={f"proposed@K={K}": dict(K=K) for K in K_VALUES}))
            jobs.append(dict(common, ablation="features", setting="",
                             methods=["alg1"] + [f"proposed@{f}" for f in FEATURE_SETS],
                             method_gate_kw={f"proposed@{f}": dict(feature_set=f) for f in FEATURE_SETS}))
            jobs.append(dict(common, ablation="replay", setting="",
                             methods=["alg1"] + [f"proposed@B={b}" for b in REPLAY_VALUES],
                             method_gate_kw={f"proposed@B={b}": dict(replay=b) for b in REPLAY_VALUES}))
            for lr in SWEEP_LR:  # (b) learning rate x weight decay
                names = [f"proposed@lr={lr:g},wd={wd:g}" for wd in SWEEP_WD]
                jobs.append(dict(common, ablation="lrwd", setting="", methods=["alg1"] + names,
                                 method_hp={nm: dict(lr=lr, wd=wd) for nm, wd in zip(names, SWEEP_WD)}))
            names = [f"{m}@top{k_}" for k_ in TOPK_VALUES for m in ("alg1", "proposed")]
            jobs.append(dict(common, ablation="topk", setting="", methods=["alg1", "proposed"] + names,
                             method_gate_kw={nm: dict(top_k=int(nm.split("@top")[1])) for nm in names}))
    for k in config.REAL_KEYS:  # (c) heterogeneous experts
        extras = [("ses", hps[k]["ses_alpha"])] + ([("snaive", 24)] if k == "etth1" else [])
        for s in seeds:
            for setting, ex in (("AR only", None), ("AR + non-AR", extras)):
                jobs.append(dict(key=k, seed=s, hp=hps[k], ablation="nonar", setting=setting,
                                 methods=["proposed", "alg1", "rls"], extras=ex, nonar_weight=ex is not None))
    jobs.sort(key=lambda j: -config.DATASETS[j["key"]]["n"])
    log(f"running {len(jobs)} ablation jobs")
    rows = pd.DataFrame([r for rr in parallel_map(abl_job, jobs, args.jobs, log) for r in rr])
    rows.to_csv(OUT / "tables" / "ablation_results_long.csv", index=False)
    d = with_rel_to_alg1(rows)

    refs = ["best_expert", "segment_oracle"]
    a = d[d.ablation == "L"]
    save_pair(a, "a_num_experts_L", "(a) Number of AR experts L", base4 + refs, [f"L={L}" for L in L_VALUES],
              note=f"Synthetic data evaluated from m = {SYN_EVAL_START} for every L. With L = 4 the order-6 and "
                   "order-8 regimes have no correctly specified expert.")
    a = d[d.ablation == "lam"]
    save_pair(a, "a_forgetting_lambda", "(a) Forgetting factor lam", base4 + refs, [f"lam={lam:g}" for lam in lams])
    fig_lambda(a)
    for abl, title, order in (("K", "(a) Error window K", [f"K={K}" for K in K_VALUES]),
                              ("features", "(a) Gate input features", list(FEATURE_SETS)),
                              ("replay", "(a) Replay buffer size B", [f"B={b}" for b in REPLAY_VALUES])):
        a = d[(d.ablation == abl) & (d.method == "proposed")]
        save_table(pivot_table(a, ["proposed"], "rel_vs_alg1_pct", 3, order).rename(
            columns={METHOD_LABELS["proposed"]: "proposed vs Alg. 1 (%)"}), OUT / "tables" / f"a_{abl}",
            f"{title}: proposed relative to Algorithm 1 (%, mean ± std over seeds; negative = better)")
    a = d[d.ablation == "lrwd"]
    fig_sweep(a)
    g = a[a.method == "proposed"].groupby(["dataset", "setting"]).rel_vs_alg1_pct
    sweep = g.agg(["mean", "std", "max"]).reset_index().rename(
        columns={"mean": "mean_rel_pct", "std": "std_rel_pct", "max": "worst_seed_rel_pct"})
    sweep["dataset"] = pd.Categorical(sweep.dataset, [k for k in config.DATASETS if k in set(sweep.dataset)], True)
    save_table(sweep.sort_values(["dataset", "setting"]), OUT / "tables" / "b_lr_wd_sweep",
               "(b) Residual learning rate x weight decay: proposed vs Algorithm 1 (%)", digits=3,
               note="worst_seed_rel_pct > 0 means at least one seed was worse than Algorithm 1.")
    a = d[d.ablation == "topk"].copy()
    a["variant"] = np.where(a.setting == "", a.method + " (full softmax)", a.method + " " + a.setting)
    tk = []
    for k in keys:
        for v in dict.fromkeys(a.variant):
            gg = a[(a.dataset == k) & (a.variant == v)]
            if len(gg) and gg.method.iloc[0] in ("alg1", "proposed"):
                tk.append(dict(dataset=k, variant=v, cum_mse=mean_std(gg.cum_mse), rel_vs_alg1_pct=mean_std(gg.rel_vs_alg1_pct, 3)))
    save_table(pd.DataFrame(tk), OUT / "tables" / "b_topk_gating",
               "(b) Top-k sparse gating vs full softmax (relative to Algorithm 1 with full softmax)")
    a = d[d.ablation == "nonar"]
    nr = []
    for k in config.REAL_KEYS:
        for setting in ("AR only", "AR + non-AR"):
            gg = a[(a.dataset == k) & (a.setting == setting)]
            rec = dict(dataset=k, experts=setting, best_expert_in_hindsight=gg.best_expert_name.iloc[0])
            for m in ("proposed", "alg1", "rls", "best_expert"):
                rec[METHOD_LABELS[m]] = mean_std(gg[gg.method == m].cum_mse)
            if "mean_weight_nonar" in gg:
                w = gg[gg.method == "proposed"].mean_weight_nonar
                rec["proposed: mean weight on non-AR"] = fmt_num(w.mean(), 3) if w.notna().any() else "-"
            nr.append(rec)
    save_table(pd.DataFrame(nr), OUT / "tables" / "c_heterogeneous_experts",
               "(c) Adding non-AR experts (SES; + seasonal naive lag 24 on ETTh1): cumulative MSE")
    log(f"done -> {OUT}")


if __name__ == "__main__":
    main()
