"""Levinson-Durbin order recursions for scalar and vector autoregressive models.

Conventions
-----------
Autocovariances are stored lag-first: ``R[k] = E[x_t x_{t-k}^T]`` for ``k = 0..L``
(shape ``(L+1, n, n)``), so ``R(-k) = R(k)^T``.

A forward VAR(p) model is ``x_t = sum_{k=1}^p A_k x_{t-k} + e_t`` with prediction-error
covariance ``Sigma_p``. One run of the recursion returns the coefficients and
``Sigma_p`` for *every* order ``p = 1..L`` at once.

Multichannel (Whittle / Wiggins-Robinson) recursion, order ``p-1 -> p``::

    Delta   = R(p) - sum_{k=1}^{p-1} A_k R(p-k)          (forward/backward error cross-cov.)
    K_f     = Delta Sigma_b^{-1},   K_b = Delta^T Sigma_f^{-1}
    A_k    <- A_k - K_f B_{p-k},    B_k <- B_k - K_b A_{p-k}      (k < p)
    A_p     = K_f,                  B_p = K_b
    Sigma_f <- Sigma_f - K_f Delta^T,   Sigma_b <- Sigma_b - K_b Delta

Stability guard: the step is valid only while the updated forward and backward error
covariances stay positive definite, i.e. while the *normalized* reflection matrix
``N = Sigma_f^{-1/2} Delta Sigma_b^{-T/2}`` has all singular values < 1 (we require
``sigma_max(N) < 1 - tol``). The recursion stops at the first order that fails.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LDResult:
    """Output of one Levinson-Durbin run.

    Attributes
    ----------
    coefs : (L, L, n, n) array; ``coefs[p-1, k-1]`` is ``A_k`` of the AR(p) model (zero for k > p).
    sigma : (L, n, n) array; forward prediction-error covariance ``Sigma_p`` (NaN where invalid).
    refl : (L,) array; largest singular value of the normalized reflection matrix at each order.
    valid : int; largest order whose recursion step passed the stability guard (0..L).
    """

    coefs: np.ndarray
    sigma: np.ndarray
    refl: np.ndarray
    valid: int


# --------------------------------------------------------------------------------------
# scalar recursion
# --------------------------------------------------------------------------------------
def levinson_durbin_scalar(r: np.ndarray, order: int, tol: float = 1e-4) -> LDResult:
    """Classical scalar Levinson-Durbin recursion.

    Parameters
    ----------
    r : (order+1,) autocovariances r(0), ..., r(order).
    order : maximal AR order L.
    tol : stability margin; the recursion stops when |kappa_p| >= 1 - tol.

    Returns an :class:`LDResult` with ``n = 1``.
    """
    r = np.asarray(r, dtype=float).ravel()
    coefs = np.zeros((order, order))
    sigma = np.full(order, np.nan)
    refl = np.full(order, np.nan)
    a = np.zeros(order)
    s = float(r[0])
    valid = 0
    for p in range(1, order + 1):
        delta = r[p] - np.dot(a[: p - 1], r[p - 1 : 0 : -1]) if p > 1 else r[1]
        if not s > 0:
            break
        kappa = delta / s
        refl[p - 1] = abs(kappa)
        if not abs(kappa) < 1.0 - tol:
            break
        if p > 1:
            a[: p - 1] = a[: p - 1] - kappa * a[: p - 1][::-1]
        a[p - 1] = kappa
        s = s * (1.0 - kappa * kappa)
        coefs[p - 1, :p] = a[:p]
        sigma[p - 1] = s
        valid = p
    return LDResult(coefs[:, :, None, None], sigma[:, None, None], refl, valid)


# --------------------------------------------------------------------------------------
# multichannel (Whittle) recursion
# --------------------------------------------------------------------------------------
def whittle_levinson(R: np.ndarray, order: int, tol: float = 1e-4) -> LDResult:
    """Multichannel Levinson-Durbin (Whittle / Wiggins-Robinson) recursion.

    Parameters
    ----------
    R : (order+1, n, n) autocovariances, ``R[k] = E[x_t x_{t-k}^T]``.
    order : maximal AR order L.
    tol : stability margin on the normalized reflection matrix.
    """
    R = np.asarray(R, dtype=float)
    n = R.shape[1]
    if n == 1:
        return levinson_durbin_scalar(R[:, 0, 0], order, tol)

    coefs = np.zeros((order, order, n, n))
    sigma = np.full((order, n, n), np.nan)
    refl = np.full(order, np.nan)
    Af = np.zeros((order, n, n))  # forward coefficients of the current order
    Ab = np.zeros((order, n, n))  # backward coefficients of the current order
    Sf = R[0].copy()
    Sb = R[0].copy()
    valid = 0
    for p in range(1, order + 1):
        if p == 1:
            delta = R[1].copy()
        else:
            delta = R[p] - np.einsum("kij,kjl->il", Af[: p - 1], R[p - 1 : 0 : -1])
        try:
            Lf = np.linalg.cholesky(Sf)
            Lb = np.linalg.cholesky(Sb)
        except np.linalg.LinAlgError:
            break
        # normalized reflection matrix N = Lf^{-1} Delta Lb^{-T}
        N = np.linalg.solve(Lb, np.linalg.solve(Lf, delta).T).T
        smax = np.linalg.norm(N, 2)
        refl[p - 1] = smax
        if not smax < 1.0 - tol:
            break
        Kf = np.linalg.solve(Sb, delta.T).T  # Delta Sigma_b^{-1}
        Kb = np.linalg.solve(Sf, delta).T  # Delta^T Sigma_f^{-1}
        if p > 1:
            Af_old = Af[: p - 1].copy()
            Af[: p - 1] = Af_old - np.einsum("ij,kjl->kil", Kf, Ab[: p - 1][::-1])
            Ab[: p - 1] = Ab[: p - 1] - np.einsum("ij,kjl->kil", Kb, Af_old[::-1])
        Af[p - 1] = Kf
        Ab[p - 1] = Kb
        Sf = Sf - Kf @ delta.T
        Sb = Sb - Kb @ delta
        Sf = 0.5 * (Sf + Sf.T)
        Sb = 0.5 * (Sb + Sb.T)
        coefs[p - 1, :p] = Af[:p]
        sigma[p - 1] = Sf
        valid = p
    return LDResult(coefs, sigma, refl, valid)


# --------------------------------------------------------------------------------------
# reference implementations used for testing
# --------------------------------------------------------------------------------------
def block_toeplitz(R: np.ndarray, p: int) -> np.ndarray:
    """(np x np) block-Toeplitz matrix Gamma_p with block (k, j) = R(j-k), R(-h) = R(h)^T."""
    R = np.asarray(R, dtype=float)
    n = R.shape[1]
    G = np.zeros((n * p, n * p))
    for k in range(p):
        for j in range(p):
            h = j - k
            G[k * n : (k + 1) * n, j * n : (j + 1) * n] = R[h] if h >= 0 else R[-h].T
    return G


def yule_walker_direct(R: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Solve the block-Toeplitz Yule-Walker system of order p with ``np.linalg.solve``.

    ``[R(1) ... R(p)] = [A_1 ... A_p] Gamma_p`` and ``Sigma_p = R(0) - sum_k A_k R(k)^T``.

    Returns ``A`` of shape (p, n, n) and ``Sigma`` of shape (n, n).
    """
    R = np.asarray(R, dtype=float)
    n = R.shape[1]
    G = block_toeplitz(R, p)
    rhs = np.concatenate([R[k] for k in range(1, p + 1)], axis=1)  # n x np
    A_flat = np.linalg.solve(G.T, rhs.T).T  # n x np
    A = A_flat.reshape(n, p, n).transpose(1, 0, 2)
    Sigma = R[0] - sum(A[k] @ R[k + 1].T for k in range(p))
    return A, Sigma


def sample_autocov(X: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased sample autocovariances (guaranteed positive-definite block Toeplitz).

    ``R[k] = (1/N) sum_{t=k}^{N-1} x_t x_{t-k}^T`` (no mean removal).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    N = X.shape[0]
    return np.stack([X[k:].T @ X[: N - k] / N for k in range(max_lag + 1)])
