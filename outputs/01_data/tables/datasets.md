### Datasets and evaluation windows

| dataset | description | N | n | warmup_T0 | tuning_window | eval_window | eval_steps | n_segments | n_switches | segment_def |
|---|---|---|---|---|---|---|---|---|---|---|
| syn_n1_T500 | Synthetic n=1, T=500 | 10000 | 1 | 40 | separate stream (seed 1000) | [40, 10000) | 9960 | 20 | 19 | true regimes |
| syn_n1_T2000 | Synthetic n=1, T=2000 | 10000 | 1 | 40 | separate stream (seed 1000) | [40, 10000) | 9960 | 5 | 4 | true regimes |
| syn_n1_Tinf | Synthetic n=1, T=inf | 10000 | 1 | 40 | separate stream (seed 1000) | [40, 10000) | 9960 | 1 | 0 | true regimes |
| syn_n3_T500 | Synthetic n=3, T=500 | 10000 | 3 | 40 | separate stream (seed 1000) | [40, 10000) | 9960 | 20 | 19 | true regimes |
| syn_n3_T2000 | Synthetic n=3, T=2000 | 10000 | 3 | 40 | separate stream (seed 1000) | [40, 10000) | 9960 | 5 | 4 | true regimes |
| syn_n3_Tinf | Synthetic n=3, T=inf | 10000 | 3 | 40 | separate stream (seed 1000) | [40, 10000) | 9960 | 1 | 0 | true regimes |
| drift_ar1 | Drifting AR(1) [1, Ex. 4] | 10000 | 1 | 40 | separate stream (seed 1000) | [40, 10000) | 9960 | 2 | 1 | true regimes |
| etth1 | ETTh1 (n=7) | 17420 | 7 | 40 | [40, 3484) | [3484, 17420) | 13936 | 28 | - | blocks of 500 |
| sunspots | Monthly sunspots (n=1) | 3330 | 1 | 40 | [40, 666) | [666, 3330) | 2664 | 6 | - | blocks of 500 |
