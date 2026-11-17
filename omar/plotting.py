"""Shared figure style and plotting helpers (matplotlib, static PNG + PDF for the report).

Colors follow the *method*, never its rank, so every figure uses the same color for the same
method. Experts AR(1)..AR(L) are ordered, so they get a sequential (light -> dark) blue ramp.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

# reference palette (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
             "#104281", "#0d366b"]
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec835a",
                                                                 "#b02a2a"])

METHOD_STYLE = {
    # name: (color, linestyle, label)
    "proposed": (CATEGORICAL[0], "-", "Alg. 1 + residual MLP (proposed)"),
    "alg1": (CATEGORICAL[1], "-", "Algorithm 1 (fixed share)"),
    "fpe": (CATEGORICAL[2], "-", "FPE order selection"),
    "rls": (CATEGORICAL[3], "-", "RLS stacking"),
    "best_expert": (CATEGORICAL[4], "-", "Best single expert (hindsight)"),
    "hedge": (CATEGORICAL[5], "-", "Hedge (alpha=0)"),
    "alg1_learned": (CATEGORICAL[6], "-", "Alg. 1, learned (eta, alpha)"),
    "plain_mlp": (CATEGORICAL[7], "-", "Plain MLP gate"),
    "uniform": (MUTED, "--", "Uniform average (ref.)"),
    "segment_oracle": (INK2, "--", "Segment oracle (ref.)"),
    "step_oracle": (MUTED, ":", "Per-step oracle (ref.)"),
}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "figure.dpi": 100,
    })


def order_colors(L: int) -> list:
    """Sequential blue ramp for experts AR(1)..AR(L) (light = low order, dark = high order)."""
    ramp = np.array([to_rgb(c) for c in BLUE_RAMP])
    pos = np.linspace(0, len(ramp) - 1, L)
    return [tuple(np.array([np.interp(p, np.arange(len(ramp)), ramp[:, k]) for k in range(3)])) for p in pos]


def expert_colors(names: list) -> list:
    """AR experts on the blue ramp; non-AR experts get categorical slots 2, 3, ..."""
    n_ar = sum(nm.startswith("AR(") for nm in names)
    cols = order_colors(n_ar)
    extra = iter(CATEGORICAL[1:])
    return [cols[i] if nm.startswith("AR(") else next(extra) for i, nm in enumerate(names)]


def style(name: str) -> tuple:
    return METHOD_STYLE.get(name, (MUTED, "-", name))


def savefig(fig, path: Path, pdf: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def stacked_gamma(ax, t: np.ndarray, gamma: np.ndarray, names: list, legend: bool = True) -> None:
    """Stacked area of the gate weights over time with a thin surface gap between bands."""
    cols = expert_colors(names)
    polys = ax.stackplot(t, gamma.T, colors=cols, labels=names, linewidth=0.3, edgecolor=SURFACE)
    for p in polys:  # 10^4-step areas: rasterize inside PDFs (axes and text stay vector) to keep files small
        p.set_rasterized(True)
    ax.set_ylim(0, 1)
    ax.set_xlim(t[0], t[-1])
    ax.grid(False)
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7, reverse=True)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")
