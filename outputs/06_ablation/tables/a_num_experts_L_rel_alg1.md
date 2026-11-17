### (a) Number of AR experts L: cum. MSE relative to Algorithm 1 at the same setting (%, negative = better)

| dataset | setting | Alg. 1 + residual MLP (proposed) | Algorithm 1 (fixed share) | FPE order selection | RLS stacking |
|---|---|---|---|---|---|
| syn_n1_T500 | L=4 | -0.002 ± 0.033 | 0.000 ± 0.000 | 1.322 ± 0.541 | 0.657 ± 0.327 |
| syn_n1_T500 | L=8 | -0.060 ± 0.010 | 0.000 ± 0.000 | 3.025 ± 0.642 | 1.067 ± 0.019 |
| syn_n1_T500 | L=12 | -0.038 ± 0.052 | 0.000 ± 0.000 | 2.900 ± 0.600 | 1.127 ± 0.075 |
| syn_n3_T500 | L=4 | -0.092 ± 0.037 | 0.000 ± 0.000 | 2.703 ± 0.049 | 1.628 ± 0.335 |
| syn_n3_T500 | L=8 | -0.074 ± 0.058 | 0.000 ± 0.000 | 5.402 ± 0.509 | 2.682 ± 0.591 |
| syn_n3_T500 | L=12 | -0.083 ± 0.070 | 0.000 ± 0.000 | 5.390 ± 0.542 | 2.660 ± 0.617 |
| etth1 | L=4 | -0.090 ± 0.005 | 0.000 ± 0.000 | -0.009 ± 0.000 | 0.312 ± 0.000 |
| etth1 | L=8 | -0.341 ± 0.004 | 0.000 ± 0.000 | 2.406 ± 0.000 | -0.182 ± 0.000 |
| etth1 | L=12 | -0.628 ± 0.024 | 0.000 ± 0.000 | 4.184 ± 0.000 | -1.494 ± 0.000 |
| sunspots | L=4 | -0.003 ± 0.004 | 0.000 ± 0.000 | 0.107 ± 0.000 | -1.226 ± 0.000 |
| sunspots | L=8 | -0.061 ± 0.043 | 0.000 ± 0.000 | -0.295 ± 0.000 | -1.616 ± 0.000 |
| sunspots | L=12 | 0.054 ± 0.157 | 0.000 ± 0.000 | 0.203 ± 0.000 | -1.391 ± 0.000 |

Synthetic data evaluated from m = 60 for every L. With L = 4 the order-6 and order-8 regimes have no correctly specified expert.
