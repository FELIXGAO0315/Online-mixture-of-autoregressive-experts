"""Online bank of VAR(p) experts, p = 1..L, re-fitted at every step by Levinson-Durbin."""
from __future__ import annotations

import numpy as np

from .levinson import whittle_levinson


class ARExpertBank:
    """L vector-AR experts sharing one exponentially-weighted autocovariance estimate.

    After observing x_m:
        R(k) <- lam R(k) + (1 - lam) x_m x_{m-k}^T,   k = 0..L
    then (bias-corrected by 1/(1 - lam^t)) and regularized,
        R(0) <- R(0) + eps tr(R(0))/n I,
    one run of the Whittle/Levinson-Durbin recursion yields A^{(p)} and Sigma_p for every order.
    If the recursion stops at order p (stability guard), experts of order >= p keep their
    coefficients (and Sigma) from the previous step; an expert that has never been fitted
    falls back to the highest valid order of the current fit.

    ``predict()`` returns the one-step forecasts Y(p) = sum_k A_k^{(p)} x_{m+1-k} for all p and
    only uses observations already passed to ``update`` (no look-ahead by construction).
    """

    def __init__(self, n: int, L: int = 8, lam: float = 0.99, eps: float = 1e-3, tol: float = 1e-4):
        self.n, self.L, self.lam, self.eps, self.tol = n, L, lam, eps, tol
        self.R = np.zeros((L + 1, n, n))
        self.hist = np.zeros((L + 1, n))  # hist[k] = x_{t-k}, t = most recent observation
        self.coefs = np.zeros((L, L, n, n))
        self.sigma = np.tile(np.eye(n), (L, 1, 1))
        self.logdet = np.zeros(L)
        self.has_fit = np.zeros(L, dtype=bool)
        self.t = 0
        self.valid = 0
        self.refl = np.full(L, np.nan)

    def predict(self) -> np.ndarray:
        """(L, n) forecasts of the next observation, one per expert order."""
        return np.einsum("pkij,kj->pi", self.coefs, self.hist[: self.L])

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        self.hist[1:] = self.hist[:-1]
        self.hist[0] = x
        self.R *= self.lam
        self.R += (1.0 - self.lam) * x[None, :, None] * self.hist[:, None, :]
        self.t += 1
        Rc = self.R / (1.0 - self.lam**self.t)
        Rc[0] = Rc[0] + self.eps * np.trace(Rc[0]) / self.n * np.eye(self.n)
        res = whittle_levinson(Rc, self.L, self.tol)
        v = res.valid
        self.valid, self.refl = v, res.refl
        if v > 0:
            self.coefs[:v] = res.coefs[:v]
            self.sigma[:v] = res.sigma[:v]
            self.logdet[:v] = np.linalg.slogdet(res.sigma[:v])[1]
            self.has_fit[:v] = True
            fresh = ~self.has_fit[v:]
            if fresh.any():  # never-fitted higher orders fall back to the highest valid order
                idx = np.flatnonzero(fresh) + v
                self.coefs[idx] = res.coefs[v - 1]
                self.sigma[idx] = res.sigma[v - 1]
                self.logdet[idx] = self.logdet[v - 1]


def run_expert_bank(X: np.ndarray, L: int = 8, lam: float = 0.99, eps: float = 1e-3, tol: float = 1e-4,
                    record_orders: tuple = ()) -> dict:
    """Run the expert bank over a whole stream, strictly online.

    Returns a dict with
      Y      (N, L, n): Y[m, p-1] = forecast of X[m] by AR(p), computed from X[:m] only;
      logdet (N, L):    log det Sigma_p of the fit that produced Y[m];
      valid  (N,):      largest valid order of that fit;
      refl   (N, L):    normalized reflection norms of that fit (stability diagnostic);
      coef1  (N, n, n): AR(1) coefficient used for Y[m];
      coefs_p (N, p, n, n) for every p in ``record_orders``.
    """
    X = np.asarray(X, dtype=float)
    N, n = X.shape
    bank = ARExpertBank(n, L, lam, eps, tol)
    Y = np.zeros((N, L, n))
    logdet = np.zeros((N, L))
    valid = np.zeros(N, dtype=int)
    refl = np.full((N, L), np.nan)
    coef1 = np.zeros((N, n, n))
    traces = {p: np.zeros((N, p, n, n)) for p in record_orders}
    for m in range(N):
        Y[m] = bank.predict()
        logdet[m] = bank.logdet
        valid[m] = bank.valid
        refl[m] = bank.refl
        coef1[m] = bank.coefs[0, 0]
        for p, tr in traces.items():
            tr[m] = bank.coefs[p - 1, :p]
        bank.update(X[m])
    out = dict(Y=Y, logdet=logdet, valid=valid, refl=refl, coef1=coef1)
    for p, tr in traces.items():
        out[f"coefs_{p}"] = tr
    return out
