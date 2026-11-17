"""Algorithm 1 of [1]: exponentially weighted averaging with fixed share (and Hedge, alpha = 0).

After observing X_m:
    v_i = w_i exp(-eta e_m(i)),      w_i <- (1 - alpha) v_i / sum_j v_j + alpha / M.
The forecast is the weighted average sum_i w_i Y_m(i). Everything is done in log space.
"""
from __future__ import annotations

import numpy as np

from ..stream import ExpertStream, RunResult, empty_result


def _logsumexp(s: np.ndarray, axis: int = -1) -> np.ndarray:
    mx = s.max(axis=axis, keepdims=True)
    return mx + np.log(np.exp(s - mx).sum(axis=axis, keepdims=True))


class FixedShare:
    """Streaming Algorithm 1 (backbone of the proposed gate)."""

    def __init__(self, M: int, eta: float, alpha: float):
        self.M, self.eta, self.alpha = M, float(eta), float(alpha)
        self.logw = np.full(M, -np.log(M))  # uniform initialization

    @property
    def w(self) -> np.ndarray:
        return np.exp(self.logw)

    def update(self, e: np.ndarray) -> None:
        s = self.logw - self.eta * e
        s = s - _logsumexp(s)
        if self.alpha > 0:
            s = np.logaddexp(np.log1p(-self.alpha) + s, np.log(self.alpha / self.M))
        self.logw = s


def top_k_renormalize(w: np.ndarray, k: int | None) -> np.ndarray:
    """Keep the k largest weights and renormalize (top-k sparse gating)."""
    if k is None or k >= w.shape[-1]:
        return w
    out = np.zeros_like(w)
    idx = np.argpartition(-w, k - 1)[:k]
    out[idx] = w[idx] / w[idx].sum()
    return out


def run_fixed_share(stream: ExpertStream, start: int, end: int, eta: float, alpha: float,
                    top_k: int | None = None, name: str = "alg1") -> RunResult:
    res = empty_result(name, stream.N, stream.M)
    fs = FixedShare(stream.M, eta, alpha)
    Y, X, E = stream.Y, stream.X, stream.E
    for m in range(start, end):
        w = top_k_renormalize(fs.w, top_k)
        r = X[m] - w @ Y[m]
        res.loss[m] = r @ r
        res.gamma[m] = w
        fs.update(E[m])
    return res


def fixed_share_grid(stream: ExpertStream, start: int, end: int, etas, alphas) -> np.ndarray:
    """Vectorized Algorithm 1 over a grid; returns mean loss on [start, end), shape (len(etas), len(alphas))."""
    eta_g, alpha_g = np.meshgrid(np.asarray(etas, float), np.asarray(alphas, float), indexing="ij")
    eta = eta_g.ravel()[:, None]
    alpha = alpha_g.ravel()[:, None]
    G, M = eta.shape[0], stream.M
    with np.errstate(divide="ignore"):
        log_keep = np.log1p(-alpha)
        log_share = np.log(alpha / M)  # -inf for alpha = 0 -> logaddexp is exact Hedge
    logw = np.full((G, M), -np.log(M))
    total = np.zeros(G)
    Y, X, E = stream.Y, stream.X, stream.E
    for m in range(start, end):
        w = np.exp(logw)
        r = X[m][None, :] - w @ Y[m]
        total += (r * r).sum(1)
        s = logw - eta * E[m][None, :]
        s = s - _logsumexp(s, axis=1)
        logw = np.logaddexp(log_keep + s, log_share)
    return (total / (end - start)).reshape(eta_g.shape)
