"""Containers shared by all combiners."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExpertStream:
    """Everything a combiner may use, indexed by time.

    Row m of ``Y`` / ``logdet`` is computed from X[:m] only (the experts are causal), so a
    combiner that at step m reads ``Y[m]``, ``logdet[m]`` and errors/observations of steps < m
    is strictly online. The experts do not depend on the combiner, which is why their forecasts
    can be computed once and shared by every method (identical to interleaving the two loops).
    """

    X: np.ndarray  # (N, n)
    Y: np.ndarray  # (N, M, n) expert forecasts (AR experts first, then optional extra experts)
    logdet: np.ndarray  # (N, L) log det Sigma_p of the AR experts
    names: list  # expert names, length M
    n_ar: int  # number of AR experts (= L)
    lam: float  # forgetting factor used by the AR experts
    valid: np.ndarray | None = None  # (N,) largest valid LD order
    coef1: np.ndarray | None = None  # (N, n, n) AR(1) coefficient
    _E: np.ndarray | None = field(default=None, repr=False)

    @property
    def N(self) -> int:
        return self.X.shape[0]

    @property
    def n(self) -> int:
        return self.X.shape[1]

    @property
    def M(self) -> int:
        return self.Y.shape[1]

    @property
    def E(self) -> np.ndarray:
        """(N, M) squared errors e_m(i) = ||X_m - Y_m(i)||^2 (known only after X_m is observed)."""
        if self._E is None:
            self._E = ((self.X[:, None, :] - self.Y) ** 2).sum(-1)
        return self._E


@dataclass
class RunResult:
    """Output of one combiner run. Arrays are NaN outside the run window [start, end)."""

    name: str
    loss: np.ndarray  # (N,) per-step loss ||X_m - Xhat_m||^2
    gamma: np.ndarray | None = None  # (N, M) combination weights used at step m
    extra: dict = field(default_factory=dict)


def empty_result(name: str, N: int, M: int | None = None) -> RunResult:
    loss = np.full(N, np.nan)
    gamma = None if M is None else np.full((N, M), np.nan)
    return RunResult(name, loss, gamma)
