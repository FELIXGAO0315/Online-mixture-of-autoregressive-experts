"""Evaluation metrics (project description, Sec. 1.5).

All metrics are accumulated over an evaluation window [start, end) only:
  * per-step loss         l_m = ||X_m - X_hat_m||^2
  * decayed running avg   lbar_m = beta lbar_{m-1} + (1 - beta) l_m  (beta = 0.99, bias-corrected)
  * cumulative MSE        (1/M) sum_m l_m
  * regret to the best fixed expert   sum_m l_m - min_p sum_m e_m(p)
  * regret to the segment oracle      sum_m l_m - sum_segments min_p sum_{m in seg} e_m(p)
  * regret to the per-step oracle     sum_m l_m - sum_m min_p e_m(p)     (unreachable; reference)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

METRICS = ("cum_mse", "regret_best", "regret_segment", "regret_step")


def decayed_average(loss: np.ndarray, start: int, end: int, beta: float = 0.99) -> np.ndarray:
    """Bias-corrected exponentially decayed running average of ``loss`` on [start, end) (NaN elsewhere)."""
    out = np.full(loss.shape[0], np.nan)
    seg = loss[start:end]
    avg = lfilter([1.0 - beta], [1.0, -beta], seg)
    out[start:end] = avg / (1.0 - beta ** np.arange(1, len(seg) + 1))
    return out


def segment_oracle_choice(E: np.ndarray, segments: np.ndarray, start: int, end: int) -> np.ndarray:
    """Index of the best fixed expert inside each segment, for every m in [start, end)."""
    seg = segments[start:end]
    Ew = E[start:end]
    choice = np.empty(end - start, dtype=int)
    for s in np.unique(seg):
        mask = seg == s
        choice[mask] = int(np.argmin(Ew[mask].sum(0)))
    return choice


def reference_losses(E: np.ndarray, segments: np.ndarray, start: int, end: int) -> dict:
    """Loss sequences of the hindsight / oracle references (NaN outside the window)."""
    N = E.shape[0]
    out = {}
    best = int(np.argmin(E[start:end].sum(0)))
    for name in ("best_expert", "segment_oracle", "step_oracle"):
        out[name] = np.full(N, np.nan)
    out["best_expert"][start:end] = E[start:end, best]
    choice = segment_oracle_choice(E, segments, start, end)
    out["segment_oracle"][start:end] = E[np.arange(start, end), choice]
    out["step_oracle"][start:end] = E[start:end].min(1)
    out["_best_index"] = best
    return out


def evaluate(loss: np.ndarray, E: np.ndarray, segments: np.ndarray, start: int, end: int) -> dict:
    """Scalar metrics of one loss sequence on [start, end)."""
    lw = loss[start:end]
    Ew = E[start:end]
    total = float(lw.sum())
    best_total = float(Ew.sum(0).min())
    choice = segment_oracle_choice(E, segments, start, end)
    seg_total = float(Ew[np.arange(end - start), choice].sum())
    step_total = float(Ew.min(1).sum())
    return {
        "cum_mse": total / len(lw),
        "regret_best": total - best_total,
        "regret_segment": total - seg_total,
        "regret_step": total - step_total,
    }
