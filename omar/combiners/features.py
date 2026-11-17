"""Scale-free input features u_m of the gate, built from information available before X_m.

Default feature set ``window_ewma`` (dimension (K+3) M):
  * the last K error vectors e_{m-1}, ..., e_{m-K},
  * their exponential moving averages at two time scales (beta in {0.9, 0.99}),
  * center(log w_m), the backbone log-weights (only when a backbone is used).
Error vectors are made scale-free by taking logs and centering over experts:
  slog(e) = center(log(e + 0.01 mean(e))),   center(v) = v - mean(v).
(The small additive floor keeps log ratios bounded when one error is ~0; it is itself scale-free.)

Ablation feature sets: ``last`` (only e_{m-1}), ``window`` (no EWMAs), ``window_ewma_x``
(+ the last few X vectors), ``window_ewma_logdet`` (+ center(log det Sigma_p)).
"""
from __future__ import annotations

import numpy as np

FEATURE_SETS = ("last", "window", "window_ewma", "window_ewma_x", "window_ewma_logdet")


def center(v: np.ndarray) -> np.ndarray:
    return v - v.mean(-1, keepdims=True)


def slog(e: np.ndarray) -> np.ndarray:
    return center(np.log(e + 1e-2 * e.mean(-1, keepdims=True) + 1e-12))


class GateFeatures:
    def __init__(self, M: int, n: int, K: int = 5, feature_set: str = "window_ewma", include_logw: bool = True,
                 betas=(0.9, 0.99), x_lags: int = 4, n_ar: int | None = None):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {feature_set!r}; choose from {FEATURE_SETS}")
        self.M, self.n = M, n
        self.K = 1 if feature_set == "last" else K
        self.use_ewma = feature_set.startswith("window_ewma")
        self.use_x = feature_set == "window_ewma_x"
        self.use_logdet = feature_set == "window_ewma_logdet"
        self.include_logw = include_logw
        self.betas = np.asarray(betas, float)[:, None]
        self.x_lags = x_lags
        self.n_ar = M if n_ar is None else n_ar
        self.win = np.zeros((self.K, M))  # most recent first
        self.ema = np.zeros((len(betas), M))
        self.xbuf = np.zeros((x_lags, n))

    @property
    def dim(self) -> int:
        d = self.K * self.M
        d += self.betas.shape[0] * self.M if self.use_ewma else 0
        d += self.M if self.include_logw else 0
        d += self.x_lags * self.n if self.use_x else 0
        d += self.M if self.use_logdet else 0
        return d

    def init(self, E_warm: np.ndarray, X_warm: np.ndarray) -> None:
        """Fill the error window / EWMAs / X buffer from the warm-up period (rows in time order)."""
        k = min(self.K, len(E_warm))
        if k > 0:
            self.win[:k] = E_warm[::-1][:k]
            if k < self.K:
                self.win[k:] = E_warm.mean(0)
            self.ema[:] = E_warm[-max(self.K, 10):].mean(0)
        else:
            self.win[:] = 1.0
            self.ema[:] = 1.0
        q = min(self.x_lags, len(X_warm))
        if q > 0:
            self.xbuf[:q] = X_warm[::-1][:q]

    def vector(self, logw: np.ndarray | None = None, logdet: np.ndarray | None = None) -> np.ndarray:
        parts = [slog(self.win).ravel()]
        if self.use_ewma:
            parts.append(slog(self.ema).ravel())
        if self.include_logw:
            parts.append(center(logw))
        if self.use_x:
            parts.append(self.xbuf.ravel())
        if self.use_logdet:
            ld = np.zeros(self.M)
            ld[: self.n_ar] = center(logdet[: self.n_ar])
            parts.append(ld)
        return np.concatenate(parts)

    def update(self, e: np.ndarray, x: np.ndarray) -> None:
        self.win[1:] = self.win[:-1]
        self.win[0] = e
        self.ema = self.betas * self.ema + (1.0 - self.betas) * e[None, :]
        if self.x_lags > 0:
            self.xbuf[1:] = self.xbuf[:-1]
            self.xbuf[0] = x
