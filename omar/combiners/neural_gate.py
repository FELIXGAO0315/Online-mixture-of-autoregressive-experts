"""The proposed learned residual gate (and the plain-MLP gate of the learning ladder).

    z = center(log w) + g_theta(u),     gamma = softmax(z),     X_hat = sum_i gamma_i Y_m(i)

g_theta(u) = W2 ReLU(W1 u + b1) + b2 with W2 = b2 = 0 at initialization, so the gate starts
*exactly* as Algorithm 1 (gamma = w). After X_m is observed, the backbone w takes its fixed-share
step and theta takes one AdamW step on ||X_m - X_hat_m||^2, with the expert forecasts and the
backbone log-weights treated as constants (detached). Weight decay pulls theta back toward 0,
i.e. back toward Algorithm 1, whenever the gradient signal is weak.

``backbone=None`` gives the plain MLP gate (logits = g_theta(u) only, no fixed-share prior).
``top_k`` replaces the softmax by top-k sparse gating; ``replay`` > 1 averages the loss over a
buffer of the most recent steps (the current step included).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import nn

from ..stream import ExpertStream, RunResult, empty_result
from .features import GateFeatures, center
from .fixed_share import FixedShare


class ResidualMLP(nn.Module):
    """Two-layer MLP whose output layer is zero-initialized."""

    def __init__(self, d_in: int, d_out: int, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, d_out)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(u)))


def gate_softmax(z: torch.Tensor, top_k: int | None = None) -> torch.Tensor:
    if top_k is None or top_k >= z.shape[-1]:
        return torch.softmax(z, dim=-1)
    vals, idx = torch.topk(z, top_k, dim=-1)
    return torch.zeros_like(z).scatter(-1, idx, torch.softmax(vals, dim=-1))


def run_neural_gate(stream: ExpertStream, start: int, end: int, *, backbone: tuple | None, lr: float,
                    wd: float, seed: int, K: int = 5, feature_set: str = "window_ewma", hidden: int = 32,
                    replay: int = 1, top_k: int | None = None, betas=(0.9, 0.99), x_lags: int = 4,
                    train: bool = True, name: str = "proposed") -> RunResult:
    """Run the residual gate (``backbone=(eta, alpha)``) or the plain gate (``backbone=None``)."""
    torch.manual_seed(seed)
    M, X, Y, E = stream.M, stream.X, stream.Y, stream.E
    res = empty_result(name, stream.N, M)
    fs = FixedShare(M, *backbone) if backbone is not None else None
    feats = GateFeatures(M, stream.n, K, feature_set, include_logw=fs is not None, betas=betas,
                         x_lags=x_lags, n_ar=stream.n_ar)
    lo = max(1, start - 50)
    feats.init(E[lo:start], X[:start])
    net = ResidualMLP(feats.dim, M, hidden).double()
    try:  # the fused CPU kernel is ~25 % faster for this tiny model; same update rule
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd, fused=True)
    except (RuntimeError, TypeError):
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    train = train and lr > 0
    Yt = torch.from_numpy(np.ascontiguousarray(Y))
    Xt = torch.from_numpy(np.ascontiguousarray(X))
    zeros = np.zeros(M)
    buf: deque = deque(maxlen=max(1, replay))
    for m in range(start, end):
        logw = fs.logw if fs is not None else None
        u = torch.from_numpy(feats.vector(logw, stream.logdet[m]))
        c = torch.from_numpy(center(logw) if fs is not None else zeros)
        g = net(u)
        gamma = gate_softmax(c + g, top_k)
        r = Xt[m] - gamma @ Yt[m]
        loss = r @ r
        res.loss[m] = loss.item()
        res.gamma[m] = gamma.detach().numpy()
        if train:
            if replay > 1:
                buf.append((u, c, m))
                U = torch.stack([b[0] for b in buf])
                C = torch.stack([b[1] for b in buf])
                idx = [b[2] for b in buf]
                G = gate_softmax(C + net(U), top_k)
                R = Xt[idx] - torch.einsum("bm,bmn->bn", G, Yt[idx])
                batch_loss = (R * R).sum(1).mean()
            else:
                batch_loss = loss
            opt.zero_grad(set_to_none=True)
            batch_loss.backward()
            opt.step()
        if fs is not None:
            fs.update(E[m])
        feats.update(E[m], X[m])
    return res
