### (c) Adding non-AR experts (SES; + seasonal naive lag 24 on ETTh1): cumulative MSE

| dataset | experts | best_expert_in_hindsight | Alg. 1 + residual MLP (proposed) | Algorithm 1 (fixed share) | RLS stacking | Best single expert (hindsight) | proposed: mean weight on non-AR |
|---|---|---|---|---|---|---|---|
| etth1 | AR only | AR(8) | 0.9793 ± 4.235e-05 | 0.9826 ± 0.0000 | 0.9808 ± 0.0000 | 0.9894 ± 0.0000 | - |
| etth1 | AR + non-AR | AR(8) | 0.9013 ± 4.205e-04 | 0.9705 ± 0.0000 | 0.9250 ± 0.0000 | 0.9894 ± 0.0000 | 0.228 |
| sunspots | AR only | AR(4) | 0.1479 ± 6.308e-05 | 0.1480 ± 0.0000 | 0.1456 ± 0.0000 | 0.1465 ± 0.0000 | - |
| sunspots | AR + non-AR | SES(0.5) | 0.1433 ± 0.0000 | 0.1433 ± 0.0000 | 0.1408 ± 0.0000 | 0.1433 ± 0.0000 | 1.000 |
