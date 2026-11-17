### (c) No look-ahead: the raw series is perturbed from m* on; everything used up to m* must be identical

| dataset | method | max_abs_diff_before | changed_after | m_star | pass |
|---|---|---|---|---|---|
| syn_n3_T500 | experts (Y) | 0.0000 | True | 5000 | True |
| syn_n3_T500 | proposed | 0.0000 | True | 5000 | True |
| syn_n3_T500 | alg1 | 0.0000 | True | 5000 | True |
| syn_n3_T500 | fpe | 0.0000 | True | 5000 | True |
| syn_n3_T500 | rls | 0.0000 | True | 5000 | True |
| syn_n3_T500 | uniform | 0.0000 | True | 5000 | True |
| syn_n3_T500 | hedge | 0.0000 | True | 5000 | True |
| syn_n3_T500 | alg1_learned | 0.0000 | True | 5000 | True |
| syn_n3_T500 | plain_mlp | 0.0000 | True | 5000 | True |
| etth1 | experts (Y) | 0.0000 | True | 8710 | True |
| etth1 | proposed | 0.0000 | True | 8710 | True |
| etth1 | alg1 | 0.0000 | True | 8710 | True |
| etth1 | fpe | 0.0000 | True | 8710 | True |
| etth1 | rls | 0.0000 | True | 8710 | True |
| etth1 | uniform | 0.0000 | True | 8710 | True |
| etth1 | hedge | 0.0000 | True | 8710 | True |
| etth1 | alg1_learned | 0.0000 | True | 8710 | True |
| etth1 | plain_mlp | 0.0000 | True | 8710 | True |

max_abs_diff_before = max |difference| of expert forecasts / weights up to m* and losses before m*.
