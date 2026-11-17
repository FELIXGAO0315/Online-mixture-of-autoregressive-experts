"""Diagnostics on regime-switching data: how fast does the gate re-adapt after a switch?"""
from __future__ import annotations

import numpy as np


def mean_order(gamma: np.ndarray, n_ar: int) -> np.ndarray:
    """Gate-weighted AR order sum_p gamma_p p (renormalized over the AR experts)."""
    g = gamma[:, :n_ar]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (g @ np.arange(1, n_ar + 1)) / g.sum(1)


def switch_recovery(gamma: np.ndarray, true_order: np.ndarray, switch_points: list, start: int, end: int,
                    n_ar: int, hold: int = 25) -> list[dict]:
    """Re-adaptation time after every regime switch.

    tau = number of steps after the switch until the gate's mean order crosses the midpoint between
    the old and the new true order *and stays* on the new side for ``hold`` consecutive steps.
    NaN if this never happens before the next switch.
    """
    mo = mean_order(gamma, n_ar)
    sp = [s for s in switch_points if start <= s < end]
    out = []
    for j, s in enumerate(sp):
        nxt = sp[j + 1] if j + 1 < len(sp) else end
        p_old, p_new = int(true_order[s - 1]), int(true_order[s])
        mid = 0.5 * (p_old + p_new)
        seg = mo[s:nxt]
        ok = (seg > mid) if p_new > p_old else (seg < mid)
        tau = np.nan
        if len(ok) >= hold:
            runs = np.convolve(ok.astype(float), np.ones(hold), mode="valid")
            hit = np.flatnonzero(runs >= hold)
            if len(hit):
                tau = float(hit[0])
        out.append(dict(switch=s, p_old=p_old, p_new=p_new, tau=tau))
    return out


def switch_aligned(loss: np.ndarray, switch_points: list, start: int, end: int, before: int = 50,
                   after: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Matrix of losses aligned at every switch: rows = switches, columns = offsets -before..after-1."""
    offs = np.arange(-before, after)
    rows = [loss[s + offs] for s in switch_points if s - before >= start and s + after <= end]
    return offs, (np.array(rows) if rows else np.empty((0, len(offs))))
