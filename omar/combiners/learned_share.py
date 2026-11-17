"""Learning-ladder level 3: Algorithm 1 whose (eta, alpha) are learned online by gradient descent.

Parametrization eta = exp(a), alpha = sigmoid(b). The weights w_m depend on (a, b) through the
whole history, so we use real-time recurrent learning (forward-mode differentiation): the
sensitivity D_m = dw_m / d(a, b) (shape M x 2) is propagated through each update
    s = log w - eta e,   p = softmax(s),   w' = (1 - alpha) p + alpha / M,
    ds/dtheta = D / w  - [eta e, 0],      dp/dtheta = (diag(p) - p p^T) ds/dtheta,
    D' = (1 - alpha) dp/dtheta + [0, alpha (1 - alpha) (1/M - p)].
The gradient of the one-step loss ||X_m - sum_i w_i Y_m(i)||^2 is (-2 Y_m r_m)^T D_m and (a, b)
take one Adam step per time step. This is exact for the current parameters under the usual
RTRL "slowly varying parameters" approximation.
"""
from __future__ import annotations

import numpy as np

from ..stream import ExpertStream, RunResult, empty_result


def run_learned_share(stream: ExpertStream, start: int, end: int, eta0: float, alpha0: float,
                      lr: float = 3e-3, name: str = "alg1_learned") -> RunResult:
    res = empty_result(name, stream.N, stream.M)
    M = stream.M
    alpha0 = float(np.clip(alpha0, 1e-5, 0.5))
    theta = np.array([np.log(eta0), np.log(alpha0 / (1.0 - alpha0))])
    m1 = np.zeros(2)
    m2 = np.zeros(2)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    logw = np.full(M, -np.log(M))
    D = np.zeros((M, 2))
    eta_tr = np.full(stream.N, np.nan)
    alpha_tr = np.full(stream.N, np.nan)
    grad_tr = np.full((stream.N, 2), np.nan)  # d loss_m / d(a, b), for the gradient unit test
    Y, X, E = stream.Y, stream.X, stream.E
    t = 0
    for m in range(start, end):
        w = np.exp(logw)
        r = X[m] - w @ Y[m]
        res.loss[m] = r @ r
        res.gamma[m] = w
        # gradient of the loss of step m w.r.t. theta (through w_m)
        g = (-2.0 * (Y[m] @ r)) @ D
        grad_tr[m] = g
        t += 1
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b2 * m2 + (1 - b2) * g * g
        theta = theta - lr * (m1 / (1 - b1**t)) / (np.sqrt(m2 / (1 - b2**t)) + eps_adam)
        theta = np.clip(theta, [-10.0, -12.0], [5.0, 0.0])
        eta = np.exp(theta[0])
        alpha = 1.0 / (1.0 + np.exp(-theta[1]))
        eta_tr[m], alpha_tr[m] = eta, alpha
        # weight update + sensitivity propagation
        e = E[m]
        s = logw - eta * e
        s = s - (s.max() + np.log(np.exp(s - s.max()).sum()))
        p = np.exp(s)
        dS = D / w[:, None]
        dS[:, 0] -= eta * e
        dP = p[:, None] * (dS - (p @ dS)[None, :])
        w_new = (1.0 - alpha) * p + alpha / M
        D = (1.0 - alpha) * dP
        D[:, 1] += alpha * (1.0 - alpha) * (1.0 / M - p)
        logw = np.log(w_new)
    res.extra["eta"] = eta_tr
    res.extra["alpha"] = alpha_tr
    res.extra["grad"] = grad_tr
    return res
