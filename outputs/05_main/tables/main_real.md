### Main results - real (mean ± std over 3 seeds)

| dataset | method | cum_mse | regret_best | regret_segment | regret_step |
|---|---|---|---|---|---|
| etth1 | Alg. 1 + residual MLP (proposed) | 0.9793 ± 4.235e-05 | -141.4 ± 0.5901 | -68.4165 ± 0.5901 | 2614.6 ± 0.5901 |
| etth1 | Algorithm 1 (fixed share) | 0.9826 ± 0.0000 | -94.6437 ± 0.0000 | -21.6754 ± 0.0000 | 2661.4 ± 0.0000 |
| etth1 | FPE order selection | 1.0063 ± 0.0000 | 234.9 ± 0.0000 | 307.8 ± 0.0000 | 2990.9 ± 0.0000 |
| etth1 | RLS stacking | 0.9808 ± 0.0000 | -119.5 ± 0.0000 | -46.5366 ± 0.0000 | 2636.5 ± 0.0000 |
| etth1 | Best single expert (hindsight) | 0.9894 ± 0.0000 | 0.0000 ± 0.0000 | 72.9683 ± 0.0000 | 2756.0 ± 0.0000 |
| etth1 | Uniform average (ref.) | 0.9972 ± 0.0000 | 108.4 ± 0.0000 | 181.4 ± 0.0000 | 2864.4 ± 0.0000 |
| etth1 | Segment oracle (ref.) | 0.9842 ± 0.0000 | -72.9683 ± 0.0000 | 0.0000 ± 0.0000 | 2683.1 ± 0.0000 |
| etth1 | Per-step oracle (ref.) | 0.7917 ± 0.0000 | -2756.0 ± 0.0000 | -2683.1 ± 0.0000 | 0.0000 ± 0.0000 |
| sunspots | Alg. 1 + residual MLP (proposed) | 0.1479 ± 6.308e-05 | 3.7387 ± 0.1680 | 6.9911 ± 0.1680 | 129.7 ± 0.1680 |
| sunspots | Algorithm 1 (fixed share) | 0.1480 ± 0.0000 | 3.9811 ± 0.0000 | 7.2335 ± 0.0000 | 130.0 ± 0.0000 |
| sunspots | FPE order selection | 0.1476 ± 0.0000 | 2.8166 ± 0.0000 | 6.0691 ± 0.0000 | 128.8 ± 0.0000 |
| sunspots | RLS stacking | 0.1456 ± 0.0000 | -2.3895 ± 0.0000 | 0.8629 ± 0.0000 | 123.6 ± 0.0000 |
| sunspots | Best single expert (hindsight) | 0.1465 ± 0.0000 | 0.0000 ± 0.0000 | 3.2524 ± 0.0000 | 126.0 ± 0.0000 |
| sunspots | Uniform average (ref.) | 0.1459 ± 0.0000 | -1.5071 ± 0.0000 | 1.7453 ± 0.0000 | 124.5 ± 0.0000 |
| sunspots | Segment oracle (ref.) | 0.1453 ± 0.0000 | -3.2524 ± 0.0000 | 0.0000 ± 0.0000 | 122.7 ± 0.0000 |
| sunspots | Per-step oracle (ref.) | 0.0992 ± 0.0000 | -126.0 ± 0.0000 | -122.7 ± 0.0000 | 0.0000 ± 0.0000 |

Cum. MSE = mean per-step loss ||X_m - X_hat_m||^2 on the evaluation window (standardized units). Regrets are cumulative sums over the window; negative = better than the reference. Rows marked (ref.) are reference lines, not causal baselines.
