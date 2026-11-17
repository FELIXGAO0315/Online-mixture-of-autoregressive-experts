"""Unit tests of the Levinson-Durbin recursions against a direct solve of the Yule-Walker system."""
import numpy as np
import pytest

from omar.data.synthetic import draw_stable_var, population_autocov, simulate_var, spectral_radius
from omar.experts.levinson import (block_toeplitz, levinson_durbin_scalar, sample_autocov, whittle_levinson,
                                   yule_walker_direct)


@pytest.mark.parametrize("n", [1, 2, 3, 7])
@pytest.mark.parametrize("p_true", [1, 3, 6])
def test_levinson_matches_direct_solve(n, p_true):
    rng = np.random.default_rng(10 * n + p_true)
    A = draw_stable_var(p_true, n, rng)
    X = simulate_var(A, 2000, rng)
    L = 8
    R = sample_autocov(X, L)  # biased estimator -> positive-definite block Toeplitz
    res = whittle_levinson(R, L)
    assert res.valid == L
    for p in range(1, L + 1):
        A_dir, S_dir = yule_walker_direct(R, p)
        np.testing.assert_allclose(res.coefs[p - 1, :p], A_dir, atol=1e-10)
        np.testing.assert_allclose(res.sigma[p - 1], S_dir, atol=1e-10)
        assert np.all(res.coefs[p - 1, p:] == 0)


def test_scalar_and_multichannel_agree_for_n1():
    rng = np.random.default_rng(0)
    X = simulate_var(draw_stable_var(4, 1, rng), 3000, rng)
    R = sample_autocov(X, 8)
    s = levinson_durbin_scalar(R[:, 0, 0], 8)
    for p in range(1, 9):
        A_dir, S_dir = yule_walker_direct(R, p)
        np.testing.assert_allclose(s.coefs[p - 1, :p, 0, 0], A_dir[:, 0, 0], atol=1e-12)
        np.testing.assert_allclose(s.sigma[p - 1, 0, 0], S_dir[0, 0], atol=1e-12)


@pytest.mark.parametrize("n", [1, 3])
def test_population_recovers_true_model(n):
    """On exact autocovariances of a VAR(p0), Sigma_p = I for p >= p0 and A^{(p0)} = A."""
    rng = np.random.default_rng(3)
    p0 = 4
    A = draw_stable_var(p0, n, rng)
    res = whittle_levinson(population_autocov(A, 8), 8)
    np.testing.assert_allclose(res.coefs[p0 - 1, :p0], A, atol=1e-8)
    for p in range(p0, 9):
        np.testing.assert_allclose(res.sigma[p - 1], np.eye(n), atol=1e-8)
        np.testing.assert_allclose(res.coefs[p - 1, p0:], 0.0, atol=1e-8)


def test_guard_stops_on_invalid_autocovariance():
    # r(1) > r(0) is not a valid autocovariance: the very first reflection coefficient exceeds 1
    res = levinson_durbin_scalar(np.array([1.0, 1.2, 0.5, 0.1]), 3)
    assert res.valid == 0
    # multichannel: a lag-1 matrix that makes the 2-block Toeplitz matrix indefinite
    R = np.zeros((3, 2, 2))
    R[0] = np.eye(2)
    R[1] = np.array([[1.5, 0.0], [0.0, 0.2]])
    assert np.linalg.eigvalsh(block_toeplitz(R, 2)).min() < 0
    assert whittle_levinson(R, 2).valid == 0


def test_generator_is_stable():
    rng = np.random.default_rng(1)
    for p in (2, 3, 6, 8):
        for n in (1, 3):
            assert spectral_radius(draw_stable_var(p, n, rng)) == pytest.approx(0.9, abs=1e-9)
