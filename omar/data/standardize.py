"""Causal (no look-ahead) per-channel standardization."""
from __future__ import annotations

import numpy as np


def standardize_online(X: np.ndarray, T0: int, mode: str = "expanding", min_std: float = 1e-8) -> np.ndarray:
    """Standardize each channel using only past information.

    mode="warmup":    z_m = (x_m - mu_W) / sd_W with the statistics of the warm-up block x_0..x_{T0-1}.
    mode="expanding": z_m = (x_m - mu_{m-1}) / sd_{m-1} with the expanding-window statistics of
                      x_0..x_{m-1} for m >= T0 (warm-up statistics inside the warm-up block).

    In both cases the transform applied to x_m is a function of x_0..x_{T0-1} or x_0..x_{m-1}
    only, so the standardized value z_m never uses the future. (Warm-up points are only used to
    initialize the system and are never evaluated.)
    """
    # C order makes every reduction below independent of the caller's memory layout (bit-reproducible)
    X = np.ascontiguousarray(X, dtype=float)
    N = X.shape[0]
    mu_w = X[:T0].mean(0)
    sd_w = np.maximum(X[:T0].std(0), min_std)
    if mode == "warmup":
        return (X - mu_w) / sd_w
    if mode != "expanding":
        raise ValueError(mode)
    cnt = np.arange(1, N + 1)[:, None]
    mean = np.cumsum(X, 0) / cnt  # mean of x_0..x_m
    var = np.maximum(np.cumsum(X**2, 0) / cnt - mean**2, 0.0)
    sd = np.maximum(np.sqrt(var), min_std)
    Z = (X - mu_w) / sd_w
    # statistics up to m-1 are mean[m-1], sd[m-1]
    Z[T0:] = (X[T0:] - mean[T0 - 1 : N - 1]) / sd[T0 - 1 : N - 1]
    return Z
