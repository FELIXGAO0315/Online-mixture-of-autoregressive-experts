"""Tests of the online machinery: causality, sanity check (a), and the combiners' update rules."""
import numpy as np
import pytest

from omar.combiners.baselines import run_rls
from omar.combiners.fixed_share import FixedShare, fixed_share_grid, run_fixed_share
from omar.combiners.learned_share import run_learned_share
from omar.combiners.neural_gate import run_neural_gate
from omar.data.standardize import standardize_online
from omar.data.synthetic import regime_switching_var
from omar.experts.ar_bank import run_expert_bank
from omar.stream import ExpertStream

L = 6


def make_stream(n=2, N=1500, seed=0, X=None):
    if X is None:
        X = standardize_online(regime_switching_var(n, N, 500, seed).X, 5 * L, "expanding")
    arr = run_expert_bank(X, L, 0.99)
    return ExpertStream(X, arr["Y"], arr["logdet"], [f"AR({p})" for p in range(1, L + 1)], L, 0.99,
                        arr["valid"], arr["coef1"])


@pytest.fixture(scope="module")
def stream():
    return make_stream()


def test_experts_and_standardization_are_causal():
    raw = regime_switching_var(2, 1200, 500, 1).X
    m_star = 700
    raw2 = raw.copy()
    raw2[m_star:] += 10.0
    Z1, Z2 = standardize_online(raw, 30, "expanding"), standardize_online(raw2, 30, "expanding")
    assert np.array_equal(Z1[:m_star], Z2[:m_star])
    s1, s2 = make_stream(X=Z1), make_stream(X=Z2)
    assert np.array_equal(s1.Y[: m_star + 1], s2.Y[: m_star + 1])  # Y[m] only uses X[:m]
    assert not np.array_equal(s1.Y[m_star + 1:], s2.Y[m_star + 1:])
    for kw in (dict(backbone=(1.0, 1e-2), lr=1e-3, wd=0.1), dict(backbone=None, lr=1e-3, wd=0.1)):
        r1 = run_neural_gate(s1, 30, 1200, seed=0, **kw)
        r2 = run_neural_gate(s2, 30, 1200, seed=0, **kw)
        assert np.array_equal(r1.gamma[30: m_star + 1], r2.gamma[30: m_star + 1])


def test_frozen_residual_reproduces_algorithm1(stream):
    a = run_fixed_share(stream, 30, stream.N, 1.0, 1e-2)
    f = run_neural_gate(stream, 30, stream.N, backbone=(1.0, 1e-2), lr=0.0, wd=0.0, seed=0, train=False)
    np.testing.assert_allclose(f.gamma[30:], a.gamma[30:], atol=1e-12)
    np.testing.assert_allclose(f.loss[30:], a.loss[30:], atol=1e-10)


def test_training_moves_away_from_algorithm1(stream):
    a = run_fixed_share(stream, 30, stream.N, 1.0, 1e-2)
    g = run_neural_gate(stream, 30, stream.N, backbone=(1.0, 1e-2), lr=1e-3, wd=0.1, seed=0)
    np.testing.assert_allclose(g.gamma[30], a.gamma[30], atol=1e-12)  # identical at the start
    assert np.abs(g.gamma[30:] - a.gamma[30:]).max() > 1e-4


def test_fixed_share_update_rule():
    fs = FixedShare(4, eta=2.0, alpha=0.1)
    e = np.array([0.1, 0.5, 1.0, 2.0])
    w0 = fs.w.copy()
    fs.update(e)
    v = w0 * np.exp(-2.0 * e)
    np.testing.assert_allclose(fs.w, 0.9 * v / v.sum() + 0.1 / 4)
    assert fs.w.min() >= 0.1 / 4 - 1e-15


def test_vectorized_grid_matches_streaming(stream):
    etas, alphas = [0.3, 3.0], [0.0, 1e-2]
    grid = fixed_share_grid(stream, 30, stream.N, etas, alphas)
    for i, eta in enumerate(etas):
        for j, alpha in enumerate(alphas):
            r = run_fixed_share(stream, 30, stream.N, eta, alpha)
            assert grid[i, j] == pytest.approx(np.mean(r.loss[30:]), rel=1e-12)


def test_rls_equals_ridge_solution(stream):
    """With lam_rls = 1, P0 = I, w0 = 1/M, RLS is the exact minimizer of
    sum (x - phi^T w)^2 + ||w - w0||^2 over all (step, channel) pairs seen so far."""
    end = 400
    r = run_rls(stream, 30, end + 1)
    w_rls = r.gamma[end]  # weights used at step `end` = fitted on steps 30..end-1
    Phi = stream.Y[30:end].transpose(0, 2, 1).reshape(-1, stream.M)
    x = stream.X[30:end].reshape(-1)
    w0 = np.full(stream.M, 1.0 / stream.M)
    w_ridge = np.linalg.solve(Phi.T @ Phi + np.eye(stream.M), Phi.T @ x + w0)
    np.testing.assert_allclose(w_rls, w_ridge, rtol=1e-6, atol=1e-8)


def test_learned_share_rtrl_gradient(stream):
    """The forward-mode (RTRL) gradient equals a finite difference of the loss w.r.t. (log eta, logit alpha)."""
    eta0, alpha0, end = 1.5, 0.05, 600
    r = run_learned_share(stream, 30, end, eta0, alpha0, lr=0.0)
    h = 1e-6
    a0, b0 = np.log(eta0), np.log(alpha0 / (1 - alpha0))
    sig = lambda b: 1 / (1 + np.exp(-b))  # noqa: E731
    lp = run_fixed_share(stream, 30, end, np.exp(a0 + h), alpha0).loss
    lm = run_fixed_share(stream, 30, end, np.exp(a0 - h), alpha0).loss
    np.testing.assert_allclose(r.extra["grad"][30:end, 0], (lp - lm)[30:end] / (2 * h), rtol=1e-4, atol=1e-7)
    lp = run_fixed_share(stream, 30, end, eta0, sig(b0 + h)).loss
    lm = run_fixed_share(stream, 30, end, eta0, sig(b0 - h)).loss
    np.testing.assert_allclose(r.extra["grad"][30:end, 1], (lp - lm)[30:end] / (2 * h), rtol=1e-4, atol=1e-7)
