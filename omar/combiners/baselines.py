"""Classical baselines: uniform average, online FPE order selection, RLS linear stacking."""
from __future__ import annotations

import numpy as np

from ..stream import ExpertStream, RunResult, empty_result


def run_uniform(stream: ExpertStream, start: int, end: int, name: str = "uniform") -> RunResult:
    """Reference line: gamma_i = 1/M."""
    res = empty_result(name, stream.N, stream.M)
    w = np.full(stream.M, 1.0 / stream.M)
    xhat = np.einsum("i,mij->mj", w, stream.Y[start:end])
    r = stream.X[start:end] - xhat
    res.loss[start:end] = (r * r).sum(1)
    res.gamma[start:end] = w
    return res


def fpe_penalty(L: int, n: int, lam: float) -> np.ndarray:
    """n log((N_eff + p n + 1) / (N_eff - p n - 1)) for p = 1..L (inf where N_eff <= p n + 1)."""
    neff = 1.0 / (1.0 - lam)
    p = np.arange(1, L + 1)
    denom = neff - p * n - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        pen = n * np.log((neff + p * n + 1.0) / denom)
    return np.where(denom > 0, pen, np.inf)


def run_fpe(stream: ExpertStream, start: int, end: int, name: str = "fpe") -> RunResult:
    """Akaike's Final Prediction Error, online: p_hat_m = argmin_p FPE(p), gamma one-hot.

    log FPE(p) = log det Sigma_p + n log((N_eff + p n + 1) / (N_eff - p n - 1)),  N_eff = 1/(1-lam),
    with Sigma_p the prediction-error covariances of the very fit that produced Y_m (available
    before X_m arrives, a by-product of Levinson-Durbin).
    """
    res = empty_result(name, stream.N, stream.M)
    L = stream.n_ar
    crit = stream.logdet[start:end, :L] + fpe_penalty(L, stream.n, stream.lam)[None, :]
    choice = np.argmin(crit, axis=1)
    idx = np.arange(start, end)
    res.loss[start:end] = stream.E[idx, choice]
    g = np.zeros((end - start, stream.M))
    g[np.arange(end - start), choice] = 1.0
    res.gamma[start:end] = g
    res.extra["order"] = np.full(stream.N, np.nan)
    res.extra["order"][start:end] = choice + 1
    return res


def run_rls(stream: ExpertStream, start: int, end: int, lam_rls: float = 1.0, name: str = "rls") -> RunResult:
    """Linear stacking with recursive least squares, weights shared across channels.

    Forecast X_hat_{m,c} = sum_i w_i Y_m(i)_c with unconstrained w in R^M. For each channel c at
    step m (regressor phi = (Y_m(1)_c, ..., Y_m(M)_c)), with P_0 = I and w_0 = 1/M:
        k = P phi / (lam_rls + phi^T P phi),  w <- w + k (X_{m,c} - phi^T w),
        P <- (P - k phi^T P) / lam_rls.
    """
    res = empty_result(name, stream.N, stream.M)
    M = stream.M
    w = np.full(M, 1.0 / M)
    P = np.eye(M)
    Y, X = stream.Y, stream.X
    for m in range(start, end):
        Ym = Y[m]  # (M, n)
        r = X[m] - w @ Ym
        res.loss[m] = r @ r
        res.gamma[m] = w
        for c in range(stream.n):
            phi = Ym[:, c]
            Pphi = P @ phi
            k = Pphi / (lam_rls + phi @ Pphi)
            w = w + k * (X[m, c] - phi @ w)
            P = (P - np.outer(k, Pphi)) / lam_rls
        P = 0.5 * (P + P.T)
    return res
