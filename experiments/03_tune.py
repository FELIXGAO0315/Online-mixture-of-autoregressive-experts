"""Step 03 - hyperparameter tuning (protocol Sec. 1.5).

Tuned: the experts' forgetting factor lam, Algorithm 1's (eta, alpha), the residual gate's AdamW
learning rate and weight decay, and (for the ladder / ablations) the plain gate's lr and wd, the
Adam lr of the learned-(eta, alpha) variant and the SES smoothing constant.
Tuning data: a separate synthetic stream (seed 1000) / the first 20 % of each real series.

Outputs -> outputs/03_tuning/
  hparams.json                      selected hyperparameters (read by every later step)
  tables/selected_hparams.{csv,md}
  tables/alg1_grid.csv              stage A: Algorithm 1 tuning loss for every (lam, eta, alpha)
  tables/lambda_selection.{csv,md}  best Algorithm 1 tuning loss per lam
  tables/gate_grid.{csv,md}         stage B: tuning loss of every gate configuration
  figures/alg1_grid_heatmaps.png    Algorithm 1 tuning loss over (eta, alpha) at the selected lam
"""
import _bootstrap  # noqa: F401

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from omar import config
from omar.pipeline import HPARAM_FILE, load_hparams, parallel_map, save_hparams
from omar.plotting import BLUE_RAMP, INK, savefig, setup_style
from omar.reporting import Logger, save_table
from omar.tuning import stage_a_job, stage_b_job, tune_ses

OUT = config.OUT_TUNING


def cost_rank(key: str) -> int:  # schedule expensive datasets first (better load balancing)
    return -config.DATASETS[key]["n"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=None, help="parallel worker processes")
    ap.add_argument("--quick", action="store_true", help="reduced grids (smoke test)")
    ap.add_argument("--datasets", nargs="*", default=None, help="subset of dataset keys")
    args = ap.parse_args()
    log = Logger(config.OUTPUTS / "logs" / "03_tune.log")
    setup_style()
    G = config.GRID_QUICK if args.quick else config.GRID
    keys = sorted(args.datasets or list(config.DATASETS), key=cost_rank)

    # ---------------- stage A: lam x eta x alpha for Algorithm 1 ----------------
    log(f"stage A: {len(keys)} datasets x {len(G['lam'])} lam (+ vectorized {len(G['eta'])}x{len(G['alpha'])} grid)")
    jobs = [dict(key=k, lam=lam, etas=G["eta"], alphas=G["alpha"]) for k in keys for lam in G["lam"]]
    res_a = parallel_map(stage_a_job, jobs, args.jobs, log)
    rows_a = []
    for r in res_a:
        for i, eta in enumerate(G["eta"]):
            for j, alpha in enumerate(G["alpha"]):
                rows_a.append(dict(dataset=r["key"], lam=r["lam"], eta=eta, alpha=alpha, loss=r["grid"][i, j]))
    df_a = pd.DataFrame(rows_a)
    hp = {}
    for k in keys:
        d = df_a[df_a.dataset == k]
        best = d.loc[d.loss.idxmin()]
        hed = d[(d.lam == best.lam) & (d.alpha == 0.0)]
        hp[k] = dict(lam=float(best.lam), eta=float(best.eta), alpha=float(best.alpha),
                     hedge_eta=float(hed.loc[hed.loss.idxmin()].eta), alg1_tune_loss=float(best.loss))
        log(f"  {k:13s} lam={best.lam:<6g} eta={best.eta:<5g} alpha={best.alpha:<7g} loss={best.loss:.4f}")

    # ---------------- stage B: gates at the selected lam ----------------
    jobs = []
    for k in keys:
        base = dict(key=k, lam=hp[k]["lam"], eta=hp[k]["eta"], alpha=hp[k]["alpha"])
        jobs += [dict(base, method="proposed", params=dict(lr=lr, wd=wd)) for lr in G["lr"] for wd in G["wd"]]
        jobs += [dict(base, method="plain_mlp", params=dict(plain_lr=lr, plain_wd=wd))
                 for lr in G["plain_lr"] for wd in G["plain_wd"]]
        jobs += [dict(base, method="alg1_learned", params=dict(learned_lr=lr)) for lr in G["learned_lr"]]
    log(f"stage B: {len(jobs)} gate runs")
    df_b = pd.DataFrame(parallel_map(stage_b_job, jobs, args.jobs, log))
    for k in keys:
        d = df_b[df_b.key == k]
        p = d[d.method == "proposed"].loc[lambda x: x.loss.idxmin()]
        q = d[d.method == "plain_mlp"].loc[lambda x: x.loss.idxmin()]
        r = d[d.method == "alg1_learned"].loc[lambda x: x.loss.idxmin()]
        hp[k].update(lr=float(p.lr), wd=float(p.wd), proposed_tune_loss=float(p.loss),
                     plain_lr=float(q.plain_lr), plain_wd=float(q.plain_wd), learned_lr=float(r.learned_lr))
        log(f"  {k:13s} residual lr={p.lr:g} wd={p.wd:g} loss={p.loss:.4f} (Alg.1 {hp[k]['alg1_tune_loss']:.4f}) | "
            f"plain lr={q.plain_lr:g} wd={q.plain_wd:g} | learned lr={r.learned_lr:g}")
    ses_rows = []
    for k in keys:
        if k in config.REAL_KEYS:
            hp[k]["ses_alpha"], rows = tune_ses(k, G["ses_alpha"])
            ses_rows += rows

    # merge with an existing file when tuning a subset of datasets
    if args.datasets and HPARAM_FILE.exists():
        old = load_hparams()
        old.update(hp)
        hp = old
    save_hparams(hp)

    # ---------------- tables ----------------
    sel = pd.DataFrame([dict(dataset=k, **v) for k, v in hp.items()])
    sel = sel[["dataset", "lam", "eta", "alpha", "hedge_eta", "lr", "wd", "plain_lr", "plain_wd", "learned_lr",
               "alg1_tune_loss", "proposed_tune_loss"] + (["ses_alpha"] if "ses_alpha" in sel else [])]
    save_table(sel, OUT / "tables" / "selected_hparams", "Selected hyperparameters (tuning data only)")
    df_a.to_csv(OUT / "tables" / "alg1_grid.csv", index=False)
    lam_tab = df_a.groupby(["dataset", "lam"]).loss.min().unstack("lam")
    lam_tab.columns = [f"lam={c:g}" for c in lam_tab.columns]
    save_table(lam_tab.reset_index(), OUT / "tables" / "lambda_selection",
               "Best Algorithm 1 tuning loss (over eta, alpha) for every forgetting factor")
    gate_tab = df_b.rename(columns={"key": "dataset"})
    save_table(gate_tab, OUT / "tables" / "gate_grid", "Stage B: tuning loss of every gate configuration")
    if ses_rows:
        save_table(pd.DataFrame(ses_rows), OUT / "tables" / "ses_grid", "SES expert: tuning loss")

    # ---------------- figure: Algorithm 1 grid at the selected lam ----------------
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f0efec"] + BLUE_RAMP)
    ncol = 3
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.7 * nrow), squeeze=False, layout="constrained")
    for ax, k in zip(axes.ravel(), [k for k in config.DATASETS if k in keys]):
        d = df_a[(df_a.dataset == k) & (df_a.lam == hp[k]["lam"])]
        Z = d.pivot(index="eta", columns="alpha", values="loss").sort_index()
        rel = 100 * (Z.values / Z.values.min() - 1)
        im = ax.imshow(np.minimum(rel, 10), cmap=cmap, vmin=0, vmax=10, aspect="auto", origin="lower")
        ax.set_xticks(range(Z.shape[1]), [f"{a:g}" for a in Z.columns], rotation=45, fontsize=7)
        ax.set_yticks(range(Z.shape[0]), [f"{e:g}" for e in Z.index], fontsize=7)
        i, j = np.unravel_index(np.argmin(Z.values), Z.shape)
        ax.scatter([j], [i], marker="o", s=60, facecolor="none", edgecolor=INK, lw=1.5)
        ax.set_title(f"{config.DATASETS[k]['label']}, lam={hp[k]['lam']:g}", loc="left", fontsize=9)
        ax.set_xlabel("alpha")
        ax.set_ylabel("eta")
        ax.grid(False)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="excess tuning loss over best cell (%, clipped at 10)")
    savefig(fig, OUT / "figures" / "alg1_grid_heatmaps")
    log(f"done -> {OUT}")


if __name__ == "__main__":
    main()
