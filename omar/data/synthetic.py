"""Synthetic data: regime-switching vector AR processes and the drifting AR(1) of [1, Ex. 4]."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SyntheticSeries:
    X: np.ndarray  # (N, n) observations
    true_order: np.ndarray  # (N,) true AR order at every step
    regime_id: np.ndarray  # (N,) index of the regime each step belongs to
    true_coef: np.ndarray | None = None  # (N,) true AR(1) coefficient (drifting process only)
    regime_coefs: list = field(default_factory=list)  # list of (p, n, n) coefficient arrays


# --------------------------------------------------------------------------------------
# stable VAR coefficients
# --------------------------------------------------------------------------------------
def companion_matrix(A: np.ndarray) -> np.ndarray:
    """Companion matrix of x_t = sum_k A_k x_{t-k}; A has shape (p, n, n)."""
    p, n, _ = A.shape
    C = np.zeros((n * p, n * p))
    C[:n] = np.concatenate(list(A), axis=1)
    if p > 1:
        C[n:, :-n] = np.eye(n * (p - 1))
    return C


def spectral_radius(A: np.ndarray) -> float:
    return float(np.abs(np.linalg.eigvals(companion_matrix(A))).max())


def draw_stable_var(p: int, n: int, rng: np.random.Generator, radius: float = 0.9,
                    init_scale: float = 0.1) -> np.ndarray:
    """Draw A_1..A_p i.i.d. Gaussian, then rescale so the companion spectral radius equals ``radius``.

    The rescaling ``A_k <- c^k A_k`` with ``c = radius / rho`` multiplies every eigenvalue of the
    companion matrix by ``c`` (the roots of det(z^p I - sum_k A_k z^{p-k}) scale by c), so the
    result has spectral radius exactly ``radius`` (<= 0.9) -- each regime is stable. The small
    initial scale keeps ``c`` close to 1 so that the highest lag is not geometrically suppressed
    and the true order stays identifiable.
    """
    A = rng.standard_normal((p, n, n)) * init_scale / np.sqrt(n)
    rho = spectral_radius(A)
    c = radius / rho
    return A * (c ** np.arange(1, p + 1))[:, None, None]


def population_autocov(A: np.ndarray, max_lag: int, noise_cov: np.ndarray | None = None) -> np.ndarray:
    """Exact autocovariances R(0..max_lag) of a stable VAR(p) with N(0, noise_cov) noise.

    The companion state s_t = [x_t; ...; x_{t-p+1}] has Cov(s) = Gamma solving the discrete
    Lyapunov equation Gamma = C Gamma C^T + Q, and E[s_t s_{t-k}^T] = C^k Gamma, whose top-left
    n x n block is R(k).
    """
    from scipy.linalg import solve_discrete_lyapunov

    p, n, _ = A.shape
    C = companion_matrix(A)
    Q = np.zeros((n * p, n * p))
    Q[:n, :n] = np.eye(n) if noise_cov is None else noise_cov
    G = solve_discrete_lyapunov(C, Q)
    R = np.empty((max_lag + 1, n, n))
    Ck = np.eye(n * p)
    for k in range(max_lag + 1):
        R[k] = (Ck @ G)[:n, :n]
        Ck = C @ Ck
    return R


def simulate_var(A: np.ndarray, length: int, rng: np.random.Generator, burn_in: int = 200) -> np.ndarray:
    """Simulate a VAR(p) with N(0, I) noise; the first ``burn_in`` samples are discarded."""
    p, n, _ = A.shape
    total = length + burn_in
    eps = rng.standard_normal((total, n))
    A_flat = np.concatenate(list(A), axis=1)  # n x np, acting on [x_{t-1}; ...; x_{t-p}]
    state = np.zeros(n * p)
    out = np.empty((total, n))
    for t in range(total):
        x = A_flat @ state + eps[t]
        out[t] = x
        state = np.concatenate([x, state[:-n]]) if p > 1 else x
    return out[burn_in:]


# --------------------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------------------
def regime_switching_var(n: int, length: int, T: int | None, seed: int,
                         orders=(2, 6, 3, 8), p_stationary: int = 4, radius: float = 0.9,
                         burn_in: int = 200) -> SyntheticSeries:
    """Piecewise-stationary VAR with regimes of equal length T.

    The true order follows the cycle ``orders`` (2 -> 6 -> 3 -> 8 -> 2 -> ...); coefficient
    matrices are redrawn at every regime boundary and each regime is simulated independently
    with its own burn-in. ``T=None`` is the stationary case (one regime of order ``p_stationary``).
    """
    rng = np.random.default_rng(seed)
    if T is None:
        plan = [(p_stationary, length)]
    else:
        n_reg = int(np.ceil(length / T))
        plan = [(orders[i % len(orders)], min(T, length - i * T)) for i in range(n_reg)]
    xs, order, rid, coefs = [], [], [], []
    for r, (p, Tr) in enumerate(plan):
        A = draw_stable_var(p, n, rng, radius)
        xs.append(simulate_var(A, Tr, rng, burn_in))
        order.append(np.full(Tr, p))
        rid.append(np.full(Tr, r))
        coefs.append(A)
    return SyntheticSeries(np.concatenate(xs), np.concatenate(order), np.concatenate(rid),
                           None, coefs)


def drifting_ar1(length: int, seed: int, burn_in: int = 200) -> SyntheticSeries:
    """AR(1) of [1, Example 4]: coefficient drifts linearly 0.9 -> -0.9 over the first half,
    then jumps to a constant 0.8 (smooth variation followed by an abrupt change)."""
    rng = np.random.default_rng(seed)
    half = length // 2
    a = np.concatenate([np.linspace(0.9, -0.9, half), np.full(length - half, 0.8)])
    eps = rng.standard_normal(length + burn_in)
    x = 0.0
    for t in range(burn_in):  # burn-in at the initial coefficient
        x = a[0] * x + eps[t]
    out = np.empty(length)
    for t in range(length):
        x = a[t] * x + eps[burn_in + t]
        out[t] = x
    rid = (np.arange(length) >= half).astype(int)
    return SyntheticSeries(out[:, None], np.ones(length, dtype=int), rid, a,
                           [np.array([[[0.9]]]), np.array([[[0.8]]])])
