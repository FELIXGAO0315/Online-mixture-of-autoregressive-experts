"""Step 01 - data: download the real series and describe every dataset.

Outputs -> outputs/01_data/
  tables/datasets.{csv,md}          sizes, warm-up, tuning / evaluation windows, regimes
  figures/synthetic_overview.png    regime-switching VAR streams (seed 0), shaded by true order
  figures/drift_overview.png        drifting AR(1) of [1, Ex. 4] and its true coefficient
  figures/etth1_overview.png        the 7 ETTh1 channels with the 20 % / 80 % split
  figures/sunspots_overview.png     monthly sunspots, raw and causally standardized
"""
import _bootstrap  # noqa: F401

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omar import config
from omar.data.datasets import load_dataset
from omar.data.real import ensure_real_data
from omar.plotting import INK2, MUTED, order_colors, savefig, setup_style
from omar.reporting import Logger, save_table

OUT = config.OUT_DATA


def dataset_table() -> pd.DataFrame:
    rows = []
    for key, spec in config.DATASETS.items():
        ds = load_dataset(key, 0)
        rows.append(dict(
            dataset=key, description=spec["label"], N=ds.N, n=ds.n, warmup_T0=ds.T0,
            tuning_window=("separate stream (seed %d)" % config.TUNE_SEED) if ds.is_synthetic
            else f"[{ds.tune_start}, {ds.tune_end})",
            eval_window=f"[{ds.eval_start}, {ds.eval_end})", eval_steps=ds.eval_end - ds.eval_start,
            n_segments=len(np.unique(ds.segments[ds.eval_start:ds.eval_end])),
            n_switches=len(ds.switch_points) if ds.is_synthetic else "-",
            segment_def="true regimes" if ds.is_synthetic else f"blocks of {config.REAL_BLOCK}",
        ))
    return pd.DataFrame(rows)


def shade_orders(ax, ds, colors):
    bounds = [0] + ds.switch_points + [ds.N]
    for a, b in zip(bounds[:-1], bounds[1:]):
        p = int(ds.true_order[a])
        ax.axvspan(a, b, color=colors[p - 1], alpha=0.35, lw=0)
        if len(bounds) <= 8:
            ax.text((a + b) / 2, 0.97, f"p0={p}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7, color=INK2)


def fig_synthetic():
    colors = order_colors(config.L_DEFAULT)
    fig, axes = plt.subplots(3, 2, figsize=(11, 6.5), sharex=True)
    for j, n in enumerate((1, 3)):
        for i, T in enumerate(("T500", "T2000", "Tinf")):
            ds = load_dataset(f"syn_n{n}_{T}", 0)
            ax = axes[i, j]
            shade_orders(ax, ds, colors)
            ax.plot(ds.X[:, 0], color=INK2, lw=0.4)
            ax.set_title(f"{ds.label} (seed 0), channel 1", loc="left")
            ax.set_ylabel("standardized x")
    for ax in axes[-1]:
        ax.set_xlabel("time step m")
    handles = [plt.Rectangle((0, 0), 1, 1, color=order_colors(config.L_DEFAULT)[p - 1], alpha=0.5)
               for p in sorted(set(config.ORDER_CYCLE) | {config.STATIONARY_ORDER})]
    labels = [f"true order {p}" for p in sorted(set(config.ORDER_CYCLE) | {config.STATIONARY_ORDER})]
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    savefig(fig, OUT / "figures" / "synthetic_overview")


def fig_drift():
    ds = load_dataset("drift_ar1", 0)
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    axes[0].plot(ds.X[:, 0], color=INK2, lw=0.4)
    axes[0].set_ylabel("standardized x")
    axes[0].set_title("Drifting AR(1) [1, Ex. 4] (seed 0)", loc="left")
    axes[1].plot(ds.true_coef, color=INK2, lw=1.5)
    axes[1].set_ylabel("true a_t")
    axes[1].set_xlabel("time step m")
    fig.tight_layout()
    savefig(fig, OUT / "figures" / "drift_overview")


def fig_real():
    ds = load_dataset("etth1", 0)
    fig, axes = plt.subplots(ds.n, 1, figsize=(10, 9), sharex=True)
    for c in range(ds.n):
        axes[c].plot(ds.X_raw[:, c], color=INK2, lw=0.3)
        axes[c].axvline(ds.tune_end, color=MUTED, ls="--", lw=1)
        axes[c].set_ylabel(ds.channels[c], rotation=0, ha="right", va="center")
    axes[0].set_title("ETTh1 raw channels; dashed line = end of the 20 % tuning window", loc="left")
    axes[-1].set_xlabel("time step m (hours)")
    fig.tight_layout()
    savefig(fig, OUT / "figures" / "etth1_overview")

    ds = load_dataset("sunspots", 0)
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    axes[0].plot(ds.X_raw[:, 0], color=INK2, lw=0.5)
    axes[0].set_ylabel("sunspot number")
    axes[0].set_title("Monthly mean total sunspot number (SILSO)", loc="left")
    axes[1].plot(ds.X[:, 0], color=INK2, lw=0.5)
    axes[1].set_ylabel("standardized\n(expanding stats)")
    axes[1].set_xlabel("time step m (months since 1749)")
    for ax in axes:
        ax.axvline(ds.tune_end, color=MUTED, ls="--", lw=1)
    fig.tight_layout()
    savefig(fig, OUT / "figures" / "sunspots_overview")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    log = Logger(config.OUTPUTS / "logs" / "01_prepare_data.log")
    setup_style()
    log("downloading real data (skipped if present) ...")
    ensure_real_data()
    tab = dataset_table()
    save_table(tab, OUT / "tables" / "datasets", "Datasets and evaluation windows")
    log("\n" + tab.to_string(index=False))
    fig_synthetic()
    fig_drift()
    fig_real()
    log(f"done -> {OUT}")


if __name__ == "__main__":
    main()
