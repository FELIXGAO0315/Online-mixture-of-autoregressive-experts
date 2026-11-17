### (d) When does the learned mixture beat the best single expert?

| dataset | n | switching_potential_pct | uniform_vs_best_pct | alg1_vs_best_pct | proposed_vs_best_pct | proposed_vs_alg1_pct | rls_vs_best_pct | winner |
|---|---|---|---|---|---|---|---|---|
| syn_n1_T500 | 1 | 3.10 ± 0.51 | 1.31 ± 0.24 | -1.88 ± 0.44 | -1.94 ± 0.43 | -0.060 ± 0.010 | -0.82 ± 0.43 | proposed |
| syn_n1_T2000 | 1 | 1.65 ± 0.23 | 1.79 ± 0.59 | -1.38 ± 0.29 | -1.39 ± 0.30 | -0.012 ± 0.016 | -0.56 ± 0.24 | proposed |
| syn_n1_Tinf | 1 | 0.00 ± 0.00 | 6.98 ± 2.63 | 0.05 ± 0.02 | 0.05 ± 0.02 | -0.002 ± 0.005 | 0.27 ± 0.02 | best_expert |
| syn_n3_T500 | 3 | 4.27 ± 0.71 | 0.05 ± 0.56 | -4.25 ± 0.81 | -4.32 ± 0.76 | -0.074 ± 0.057 | -1.65 ± 0.51 | proposed |
| syn_n3_T2000 | 3 | 3.35 ± 0.20 | 2.06 ± 1.26 | -3.28 ± 0.09 | -3.39 ± 0.05 | -0.117 ± 0.079 | -1.41 ± 0.27 | proposed |
| syn_n3_Tinf | 3 | 0.00 ± 0.00 | 5.72 ± 2.60 | 0.03 ± 0.01 | 0.03 ± 0.01 | -9.71e-04 ± 0.002 | 0.16 ± 0.03 | best_expert |
| drift_ar1 | 1 | 0.41 ± 0.24 | 1.52 ± 0.53 | -0.72 ± 0.21 | -0.72 ± 0.20 | -0.006 ± 0.007 | -0.09 ± 0.04 | proposed |
| etth1 | 7 | 0.53 ± 0.00 | 0.79 ± 0.00 | -0.69 ± 0.00 | -1.03 ± 0.00 | -0.341 ± 0.004 | -0.87 ± 0.00 | proposed |
| sunspots | 1 | 0.83 ± 0.00 | -0.39 ± 0.00 | 1.02 ± 0.00 | 0.96 ± 0.04 | -0.061 ± 0.043 | -0.61 ± 0.00 | rls |

switching_potential = 1 - (segment oracle / best fixed expert): how much a method that tracks the best expert per segment could gain. *_vs_best = cum. MSE relative to the best single expert (%). winner = lowest mean cum. MSE among the causal methods and the best expert in hindsight.
