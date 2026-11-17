"""Step 02 - the AR experts (project Step 2).

  (a) Levinson-Durbin (scalar and Whittle multichannel) vs a direct np.linalg.solve of the
      block-Toeplitz Yule-Walker system, all orders, n in {1, 2, 3, 7}; plus timing.
  (b) Stationary case (T = inf, p0 = 4), raw data with N(0, I) noise: the online estimates of the
      AR(p0) coefficients converge, and Sigma_p converges and flattens at p = p0 (log det -> 0).
  (c) Stability guard: how often the multichannel recursion stops before order L, per dataset
      and forgetting factor (expected to trigger often when p n^2 approaches N_eff = 1/(1-lam)).
  (d) Order identifiability of the synthetic generator: population gain in log det Sigma from
      order p0-1 to p0 (how much the true last lag matters).

Outputs -> outputs/02_experts/{tables,figures}
"""
import _bootstrap  # noqa: F401

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omar import config
from omar.data.datasets import load_dataset
from omar.data.synthetic import draw_stable_var, population_autocov, simulate_var
from omar.experts.ar_bank import run_expert_bank
from omar.experts.levinson import sample_autocov, whittle_levinson, yule_walker_direct
from omar.pipeline import get_stream, parallel_map
from omar.plotting import CATEGORICAL, INK, INK2, order_colors, savefig, setup_style
from omar.reporting import Logger, save_table

OUT = config.OUT_EXPERTS
LAMS_CONV = (0.99, 0.999, 0.9999)


# --------------------------------------------------------------------------------------
# (a) unit check against a direct solve
# --------------------------------------------------------------------------------------
def check_vs_direct(log) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    L = config.L_DEFAULT
    for n in (1, 2, 3, 7):
        for p_true in (2, 5):
            A = draw_stable_var(p_true, n, rng)
            X = simulate_var(A, 3000, rng)
            R = sample_autocov(X, L)
            t = time.perf_counter()
            for _ in range(20):
                res = whittle_levinson(R, L)
            t_ld = (time.perf_counter() - t) / 20
            t = time.perf_counter()
            direct = [yule_walker_direct(R, p) for p in range(1, L + 1)]
            t_direct = time.perf_counter() - t
            err_a = max(np.abs(res.coefs[p - 1, :p] - direct[p - 1][0]).max() for p in range(1, L + 1))
            err_s = max(np.abs(res.sigma[p - 1] - direct[p - 1][1]).max() for p in range(1, L + 1))
            rows.append(dict(n=n, true_order=p_true, orders_checked=f"1..{L}", valid_orders=res.valid,
                             max_abs_err_coefs=err_a, max_abs_err_sigma=err_s,
                             levinson_ms=1e3 * t_ld, direct_all_orders_ms=1e3 * t_direct))
    df = pd.DataFrame(rows)
    log("(a) LD vs direct solve:\n" + df.to_string(index=False))
    return df


def timing_scaling(log) -> pd.DataFrame:
    """All orders 1..L at once: O(L^2 n^3) recursion vs L separate O((p n)^3) direct solves."""
    rng = np.random.default_rng(1)
    n = 3
    A = draw_stable_var(3, n, rng)
    X = simulate_var(A, 20000, rng)
    rows = []
    for L in (8, 16, 32, 64):
        R = sample_autocov(X, L)
        reps = 5
        t = time.perf_counter()
        for _ in range(reps):
            whittle_levinson(R, L)
        t_ld = (time.perf_counter() - t) / reps
        t = time.perf_counter()
        for _ in range(reps):
            for p in range(1, L + 1):
                yule_walker_direct(R, p)
        t_dir = (time.perf_counter() - t) / reps
        rows.append(dict(n=n, L=L, levinson_all_orders_ms=1e3 * t_ld, direct_all_orders_ms=1e3 * t_dir,
                         speedup=t_dir / t_ld))
    df = pd.DataFrame(rows)
    log("(a') timing vs L:\n" + df.to_string(index=False))
    return df


# --------------------------------------------------------------------------------------
# (b) convergence on the stationary case
# --------------------------------------------------------------------------------------
def _conv_job(n: int, lam: float) -> dict:
    ds = load_dataset(f"syn_n{n}_Tinf", 0)
    out = run_expert_bank(ds.X_raw, config.L_DEFAULT, lam, config.LD_EPS, config.LD_TOL,
                          record_orders=(config.STATIONARY_ORDER,))
    A = ds.regime_coefs[0]
    est = out[f"coefs_{config.STATIONARY_ORDER}"]
    err = np.sqrt(((est - A[None]) ** 2).sum((1, 2, 3))) / np.sqrt((A**2).sum())
    return dict(n=n, lam=lam, err=err, logdet=out["logdet"] / n, A=A)


def convergence(log, n_jobs):
    jobs = [dict(n=n, lam=lam) for n in (1, 3) for lam in LAMS_CONV]
    res = parallel_map(_conv_job, jobs, n_jobs)
    L, p0 = config.L_DEFAULT, config.STATIONARY_ORDER
    m = np.arange(config.SYN_LENGTH)
    lam_colors = dict(zip(LAMS_CONV, CATEGORICAL[:3]))

    # coefficient error vs time
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, n in zip(axes, (1, 3)):
        for r in (r for r in res if r["n"] == n):
            ax.loglog(m[50:], r["err"][50:], color=lam_colors[r["lam"]], lw=1.2,
                      label=f"lam={r['lam']} (N_eff={1 / (1 - r['lam']):.0f})")
        ref = m[50:].astype(float)
        ax.loglog(ref, 3.0 / np.sqrt(ref), color=INK2, ls=":", lw=1, label="~ 1/sqrt(m)")
        ax.set_title(f"n={n}: relative error of the AR({p0}) coefficients", loc="left")
        ax.set_xlabel("time step m")
    axes[0].set_ylabel("||A_hat - A||_F / ||A||_F")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    savefig(fig, OUT / "figures" / "coef_convergence")

    # Sigma_p vs order (end of stream) + population values
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    rows = []
    for ax, n in zip(axes, (1, 3)):
        A = next(r["A"] for r in res if r["n"] == n)
        pop = whittle_levinson(population_autocov(A, L), L)
        pop_ld = np.array([np.linalg.slogdet(pop.sigma[p])[1] / n for p in range(L)])
        ax.plot(np.arange(1, L + 1), pop_ld, color=INK, ls="--", marker="o", ms=5, label="population (exact)")
        for r in (r for r in res if r["n"] == n):
            final = r["logdet"][-2000:].mean(0)
            ax.plot(np.arange(1, L + 1), final, color=lam_colors[r["lam"]], marker="o", ms=4,
                    label=f"online, lam={r['lam']} (mean of last 2000 steps)")
            for p in range(L):
                rows.append(dict(n=n, lam=r["lam"], order=p + 1, logdet_per_channel_online=final[p],
                                 logdet_per_channel_population=pop_ld[p]))
        ax.axvline(p0, color=INK2, lw=0.8, ls=":")
        ax.set_title(f"n={n}: (1/n) log det Sigma_p vs order (true p0={p0}, noise cov = I)", loc="left")
        ax.set_xlabel("AR order p")
    axes[0].set_ylabel("(1/n) log det Sigma_p")
    axes[1].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    savefig(fig, OUT / "figures" / "sigma_vs_order")
    save_table(pd.DataFrame(rows), OUT / "tables" / "sigma_vs_order",
               "Stationary case: (1/n) log det Sigma_p, online (end of stream) vs population", digits=4)

    # Sigma_p over time (lam = 0.999)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    cols = order_colors(L)
    for ax, n in zip(axes, (1, 3)):
        r = next(r for r in res if r["n"] == n and r["lam"] == 0.999)
        for p in range(L):
            ax.plot(m[100:], r["logdet"][100:, p], color=cols[p], lw=1, label=f"p={p + 1}")
        ax.axhline(0.0, color=INK, ls="--", lw=0.8)
        ax.set_title(f"n={n}: (1/n) log det Sigma_p over time, lam=0.999", loc="left")
        ax.set_xlabel("time step m")
    axes[0].set_ylabel("(1/n) log det Sigma_p")
    axes[1].legend(ncol=2, loc="upper right", fontsize=7)
    fig.tight_layout()
    savefig(fig, OUT / "figures" / "sigma_over_time")

    conv = pd.DataFrame([dict(n=r["n"], lam=r["lam"], rel_coef_err_m1000=r["err"][1000],
                              rel_coef_err_m5000=r["err"][5000], rel_coef_err_final=r["err"][-1])
                         for r in res])
    save_table(conv, OUT / "tables" / "coef_convergence", "Relative error of the AR(p0) coefficient estimates")
    log("(b) convergence:\n" + conv.to_string(index=False))


# --------------------------------------------------------------------------------------
# (c) stability guard statistics
# --------------------------------------------------------------------------------------
def _guard_job(key: str, lam: float) -> dict:
    ds, st = get_stream(key, 0, config.L_DEFAULT, lam)
    v = st.valid[ds.T0:]
    L, n = config.L_DEFAULT, ds.n
    neff = 1.0 / (1.0 - lam)
    return dict(dataset=key, n=n, lam=lam, N_eff=neff, L_n2_over_Neff=L * n * n / neff,
                frac_steps_stopped=float(np.mean(v < L)), mean_valid_order=float(v.mean()),
                min_valid_order=int(v.min()))


def guard_stats(log, n_jobs):
    jobs = [dict(key=k, lam=lam) for k in config.DATASETS for lam in config.GRID["lam"]]
    df = pd.DataFrame(parallel_map(_guard_job, jobs, n_jobs, log))
    save_table(df, OUT / "tables" / "guard_stats", "Stability guard: fraction of steps where the LD recursion "
               "stopped before order L (seed 0, after warm-up)", digits=4)
    log("(c) guard statistics:\n" + df.to_string(index=False))


# --------------------------------------------------------------------------------------
# (d) identifiability of the true order
# --------------------------------------------------------------------------------------
def identifiability(log):
    rng = np.random.default_rng(123)
    rows = []
    for n in (1, 3):
        for p0 in sorted(set(config.ORDER_CYCLE) | {config.STATIONARY_ORDER}):
            gains, total = [], []
            for _ in range(50):
                A = draw_stable_var(p0, n, rng, config.SPECTRAL_RADIUS)
                R = population_autocov(A, p0)
                res = whittle_levinson(R, p0)
                ld = [np.linalg.slogdet(R[0])[1]] + [np.linalg.slogdet(res.sigma[q])[1] for q in range(p0)]
                gains.append((ld[p0 - 1] - ld[p0]) / n)
                total.append((ld[0] - ld[p0]) / n)
            g = np.array(gains)
            rows.append(dict(n=n, true_order=p0, gain_last_lag_median=np.median(g), gain_last_lag_q10=np.quantile(g, 0.1),
                             gain_last_lag_q90=np.quantile(g, 0.9), gain_vs_white_noise_median=np.median(total)))
    df = pd.DataFrame(rows)
    save_table(df, OUT / "tables" / "order_identifiability",
               "Population gain (1/n)[log det Sigma_{p0-1} - log det Sigma_{p0}] of the true last lag "
               "(50 draws of the generator)", digits=3)
    log("(d) identifiability:\n" + df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=None, help="parallel worker processes")
    args = ap.parse_args()
    log = Logger(config.OUTPUTS / "logs" / "02_check_experts.log")
    setup_style()
    df = check_vs_direct(log)
    save_table(df, OUT / "tables" / "levinson_vs_direct", "Levinson-Durbin vs direct block-Toeplitz solve "
               "(biased sample autocovariances of a simulated VAR)", digits=3)
    assert df["max_abs_err_coefs"].max() < 1e-8 and df["max_abs_err_sigma"].max() < 1e-8
    save_table(timing_scaling(log), OUT / "tables" / "levinson_timing_vs_L",
               "Wall-clock time to obtain all orders 1..L (n=3): Levinson-Durbin vs L direct solves. At L=8 "
               "Python overhead dominates; the O(L^2 n^3) vs O(L^4 n^3) gap shows as L grows.", digits=3)
    convergence(log, args.jobs)
    guard_stats(log, args.jobs)
    identifiability(log)
    log(f"done -> {OUT}")


if __name__ == "__main__":
    main()
