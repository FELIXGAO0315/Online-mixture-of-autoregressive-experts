### Re-adaptation after regime switches

| dataset | method | switches | frac_recovered | mean_tau | median_tau | mean_tau_up | mean_tau_down | switch_cost_200 |
|---|---|---|---|---|---|---|---|---|
| syn_n3_T500 | Alg. 1 + residual MLP (proposed) | 57 | 1.0 | 79.9 ± 9.0 | 52.0 | 122.8 | 32.3 | -1.52 ± 1.47 |
| syn_n3_T500 | Algorithm 1 (fixed share) | 57 | 1.0 | 76.6 ± 12.2 | 55.0 | 116.7 | 32.1 | -1.29 ± 1.37 |
| syn_n3_T500 | FPE order selection | 57 | 0.7 | 85.2 ± 24.5 | 1.5 | 281.4 | 5.1 | 24.18 ± 1.53 |
| syn_n3_T2000 | Alg. 1 + residual MLP (proposed) | 12 | 1.0 | 154.2 ± 37.0 | 133.5 | 175.5 | 132.8 | -4.46 ± 7.33 |
| syn_n3_T2000 | Algorithm 1 (fixed share) | 12 | 1.0 | 162.9 ± 48.3 | 138.0 | 181.2 | 144.7 | -5.27 ± 6.84 |
| syn_n3_T2000 | FPE order selection | 12 | 1.0 | 305.1 ± 149.6 | 153.5 | 494.5 | 115.7 | 24.40 ± 8.90 |
| syn_n1_T500 | Alg. 1 + residual MLP (proposed) | 57 | 1.0 | 103.5 ± 11.4 | 90.0 | 112.3 | 93.7 | 2.44 ± 0.44 |
| syn_n1_T500 | Algorithm 1 (fixed share) | 57 | 1.0 | 104.7 ± 16.0 | 87.0 | 114.7 | 93.6 | 2.39 ± 0.45 |
| syn_n1_T500 | FPE order selection | 57 | 0.9 | 75.2 ± 21.0 | 49.5 | 128.1 | 29.3 | 6.56 ± 0.80 |
| syn_n1_T2000 | Alg. 1 + residual MLP (proposed) | 12 | 1.0 | 167.9 ± 56.6 | 142.0 | 136.5 | 199.3 | 0.50 ± 1.42 |
| syn_n1_T2000 | Algorithm 1 (fixed share) | 12 | 1.0 | 165.6 ± 61.9 | 147.5 | 147.5 | 183.7 | 0.20 ± 1.40 |
| syn_n1_T2000 | FPE order selection | 12 | 1.0 | 108.6 ± 42.4 | 87.0 | 119.0 | 98.2 | 1.62 ± 0.74 |

tau = steps until the gate's mean order sum_p gamma_p p crosses the midpoint between the old and the new true order and stays there for 25 steps (NaN = never before the next switch); mean_tau: mean ± std over seeds of the per-seed average; up/down = switches to a higher / lower order. switch_cost_200 = excess loss over the segment oracle in the 200 steps after a switch.
