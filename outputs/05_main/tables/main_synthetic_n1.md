### Main results - synthetic n1 (mean ± std over 3 seeds)

| dataset | method | cum_mse | regret_best | regret_segment | regret_step |
|---|---|---|---|---|---|
| syn_n1_T500 | Alg. 1 + residual MLP (proposed) | 0.5260 ± 0.0772 | -101.4 ± 8.4404 | 61.7830 ± 8.0971 | 1840.0 ± 256.3 |
| syn_n1_T500 | Algorithm 1 (fixed share) | 0.5263 ± 0.0772 | -98.2167 ± 9.2606 | 64.9445 ± 8.8949 | 1843.2 ± 257.0 |
| syn_n1_T500 | FPE order selection | 0.5419 ± 0.0768 | 57.2675 ± 8.6946 | 220.4 ± 8.0300 | 1998.6 ± 253.0 |
| syn_n1_T500 | RLS stacking | 0.5320 ± 0.0780 | -41.8211 ± 15.5142 | 121.3 ± 14.7563 | 1899.6 ± 264.3 |
| syn_n1_T500 | Best single expert (hindsight) | 0.5362 ± 0.0766 | 0.0000 ± 0.0000 | 163.2 ± 1.2342 | 1941.4 ± 250.7 |
| syn_n1_T500 | Uniform average (ref.) | 0.5431 ± 0.0765 | 68.8680 ± 3.7881 | 232.0 ± 4.7316 | 2010.2 ± 249.2 |
| syn_n1_T500 | Segment oracle (ref.) | 0.5198 ± 0.0767 | -163.2 ± 1.2342 | 0.0000 ± 0.0000 | 1778.2 ± 251.8 |
| syn_n1_T500 | Per-step oracle (ref.) | 0.3412 ± 0.0514 | -1941.4 ± 250.7 | -1778.2 ± 251.8 | 0.0000 ± 0.0000 |
| syn_n1_T2000 | Alg. 1 + residual MLP (proposed) | 0.4799 ± 0.0213 | -67.6799 ± 17.2585 | 12.4395 ± 5.5958 | 1579.9 ± 53.9163 |
| syn_n1_T2000 | Algorithm 1 (fixed share) | 0.4800 ± 0.0214 | -67.0810 ± 16.7842 | 13.0384 ± 5.8415 | 1580.5 ± 54.1375 |
| syn_n1_T2000 | FPE order selection | 0.4838 ± 0.0221 | -29.3617 ± 7.2819 | 50.7577 ± 7.9871 | 1618.2 ± 63.2551 |
| syn_n1_T2000 | RLS stacking | 0.4840 ± 0.0220 | -27.3455 ± 12.3899 | 52.7739 ± 8.6267 | 1620.2 ± 57.7934 |
| syn_n1_T2000 | Best single expert (hindsight) | 0.4867 ± 0.0227 | 0.0000 ± 0.0000 | 80.1194 ± 14.6032 | 1647.5 ± 70.1798 |
| syn_n1_T2000 | Uniform average (ref.) | 0.4954 ± 0.0217 | 86.3159 ± 26.0618 | 166.4 ± 28.7784 | 1733.9 ± 86.2310 |
| syn_n1_T2000 | Segment oracle (ref.) | 0.4787 ± 0.0213 | -80.1194 ± 14.6032 | 0.0000 ± 0.0000 | 1567.4 ± 58.7972 |
| syn_n1_T2000 | Per-step oracle (ref.) | 0.3213 ± 0.0195 | -1647.5 ± 70.1798 | -1567.4 ± 58.7972 | 0.0000 ± 0.0000 |
| syn_n1_Tinf | Alg. 1 + residual MLP (proposed) | 0.6172 ± 0.0038 | 2.8370 ± 1.3720 | 2.8370 ± 1.3720 | 2001.3 ± 77.7536 |
| syn_n1_Tinf | Algorithm 1 (fixed share) | 0.6172 ± 0.0038 | 2.9559 ± 1.1578 | 2.9559 ± 1.1578 | 2001.4 ± 77.4911 |
| syn_n1_Tinf | FPE order selection | 0.6178 ± 0.0038 | 8.7326 ± 1.4496 | 8.7326 ± 1.4496 | 2007.2 ± 77.8741 |
| syn_n1_Tinf | RLS stacking | 0.6185 ± 0.0038 | 16.4838 ± 1.4028 | 16.4838 ± 1.4028 | 2014.9 ± 77.5388 |
| syn_n1_Tinf | Best single expert (hindsight) | 0.6169 ± 0.0037 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1998.5 ± 77.1279 |
| syn_n1_Tinf | Uniform average (ref.) | 0.6600 ± 0.0184 | 429.3 ± 162.6 | 429.3 ± 162.6 | 2427.7 ± 133.2 |
| syn_n1_Tinf | Segment oracle (ref.) | 0.6169 ± 0.0037 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1998.5 ± 77.1279 |
| syn_n1_Tinf | Per-step oracle (ref.) | 0.4162 ± 0.0071 | -1998.5 ± 77.1279 | -1998.5 ± 77.1279 | 0.0000 ± 0.0000 |
| drift_ar1 | Alg. 1 + residual MLP (proposed) | 0.5475 ± 0.0027 | -39.6404 ± 11.3180 | -16.8449 ± 4.6191 | 1036.3 ± 70.8051 |
| drift_ar1 | Algorithm 1 (fixed share) | 0.5475 ± 0.0026 | -39.3341 ± 11.5603 | -16.5385 ± 4.2685 | 1036.6 ± 70.4222 |
| drift_ar1 | FPE order selection | 0.5490 ± 0.0027 | -24.7440 ± 4.5344 | -1.9485 ± 8.8892 | 1051.1 ± 73.9169 |
| drift_ar1 | RLS stacking | 0.5510 ± 0.0029 | -4.9984 ± 2.1369 | 17.7972 ± 11.2843 | 1070.9 ± 76.6453 |
| drift_ar1 | Best single expert (hindsight) | 0.5515 ± 0.0031 | 0.0000 ± 0.0000 | 22.7956 ± 13.3355 | 1075.9 ± 78.2403 |
| drift_ar1 | Uniform average (ref.) | 0.5599 ± 6.499e-04 | 83.6046 ± 28.7193 | 106.4 ± 18.0559 | 1159.5 ± 49.5408 |
| drift_ar1 | Segment oracle (ref.) | 0.5492 ± 0.0022 | -22.7956 ± 13.3355 | 0.0000 ± 0.0000 | 1053.1 ± 66.5428 |
| drift_ar1 | Per-step oracle (ref.) | 0.4435 ± 0.0049 | -1075.9 ± 78.2403 | -1053.1 ± 66.5428 | 0.0000 ± 0.0000 |

Cum. MSE = mean per-step loss ||X_m - X_hat_m||^2 on the evaluation window (standardized units). Regrets are cumulative sums over the window; negative = better than the reference. Rows marked (ref.) are reference lines, not causal baselines.
