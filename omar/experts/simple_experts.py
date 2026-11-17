"""Non-AR experts for ablation (c): seasonal naive and simple exponential smoothing.

Both return an (N, n) array whose row m is the forecast of X[m] computed from X[:m] only.
"""
from __future__ import annotations

import numpy as np


def seasonal_naive(X: np.ndarray, lag: int) -> np.ndarray:
    """Y[m] = X[m - lag] (zero before enough history is available)."""
    X = np.asarray(X, dtype=float)
    Y = np.zeros_like(X)
    Y[lag:] = X[:-lag]
    return Y


def exponential_smoothing(X: np.ndarray, a: float) -> np.ndarray:
    """Simple exponential smoothing: level <- a x_m + (1-a) level; Y[m] = level after x_{m-1}."""
    X = np.asarray(X, dtype=float)
    Y = np.zeros_like(X)
    level = np.zeros(X.shape[1])
    for m in range(X.shape[0]):
        Y[m] = level
        level = X[m] if m == 0 else a * X[m] + (1.0 - a) * level
    return Y
