### (b) Stationary AR(4) data: mixture vs best single expert vs FPE (mean ± std over 3 seeds)

| dataset | method | cum_mse | rel_diff_vs_best_expert_pct | rel_diff_vs_fpe_pct | within_1pct_of_best |
|---|---|---|---|---|---|
| syn_n1_Tinf | proposed | 0.6172 ± 0.0038 | 0.046 ± 0.022 | -0.096 ± 0.002 | True |
| syn_n1_Tinf | alg1 | 0.6172 ± 0.0038 | 0.048 ± 0.018 | -0.094 ± 0.006 | True |
| syn_n1_Tinf | fpe | 0.6178 ± 0.0038 | 0.142 ± 0.023 | 0.000 ± 0.000 | True |
| syn_n1_Tinf | best_expert | 0.6169 ± 0.0037 | 0.000 ± 0.000 | -0.142 ± 0.023 | True |
| syn_n3_Tinf | proposed | 1.8480 ± 0.2649 | 0.030 ± 0.009 | -0.293 ± 0.035 | True |
| syn_n3_Tinf | alg1 | 1.8480 ± 0.2649 | 0.031 ± 0.008 | -0.292 ± 0.036 | True |
| syn_n3_Tinf | fpe | 1.8535 ± 0.2664 | 0.324 ± 0.028 | 0.000 ± 0.000 | True |
| syn_n3_Tinf | best_expert | 1.8474 ± 0.2649 | 0.000 ± 0.000 | -0.323 ± 0.028 | True |

Criterion: mean relative difference to the best expert in hindsight below 1 %.
