### (b) Top-k sparse gating vs full softmax (relative to Algorithm 1 with full softmax)

| dataset | variant | cum_mse | rel_vs_alg1_pct |
|---|---|---|---|
| syn_n1_T500 | alg1 (full softmax) | 0.5263 ± 0.0643 | 0.000 ± 0.000 |
| syn_n1_T500 | proposed (full softmax) | 0.5260 ± 0.0642 | -0.060 ± 0.008 |
| syn_n1_T500 | alg1 top1 | 0.5290 ± 0.0642 | 0.528 ± 0.110 |
| syn_n1_T500 | proposed top1 | 0.5290 ± 0.0642 | 0.528 ± 0.110 |
| syn_n1_T500 | alg1 top2 | 0.5270 ± 0.0641 | 0.138 ± 0.043 |
| syn_n1_T500 | proposed top2 | 0.5267 ± 0.0640 | 0.087 ± 0.136 |
| syn_n1_T500 | alg1 top3 | 0.5265 ± 0.0643 | 0.033 ± 0.010 |
| syn_n1_T500 | proposed top3 | 0.5260 ± 0.0639 | -0.054 ± 0.090 |
| syn_n3_T500 | alg1 (full softmax) | 1.8259 ± 0.2657 | 0.000 ± 0.000 |
| syn_n3_T500 | proposed (full softmax) | 1.8244 ± 0.2647 | -0.074 ± 0.048 |
| syn_n3_T500 | alg1 top1 | 1.8328 ± 0.2683 | 0.370 ± 0.080 |
| syn_n3_T500 | proposed top1 | 1.8328 ± 0.2683 | 0.370 ± 0.080 |
| syn_n3_T500 | alg1 top2 | 1.8275 ± 0.2664 | 0.086 ± 0.023 |
| syn_n3_T500 | proposed top2 | 1.8279 ± 0.2662 | 0.110 ± 0.034 |
| syn_n3_T500 | alg1 top3 | 1.8266 ± 0.2661 | 0.037 ± 0.021 |
| syn_n3_T500 | proposed top3 | 1.8260 ± 0.2659 | 0.004 ± 0.022 |
| etth1 | alg1 (full softmax) | 0.9826 ± 0.0000 | 0.000 ± 0.000 |
| etth1 | proposed (full softmax) | 0.9793 ± 3.523e-05 | -0.341 ± 0.004 |
| etth1 | alg1 top1 | 0.9887 ± 0.0000 | 0.616 ± 0.000 |
| etth1 | proposed top1 | 0.9887 ± 0.0000 | 0.616 ± 0.000 |
| etth1 | alg1 top2 | 0.9859 ± 0.0000 | 0.335 ± 0.000 |
| etth1 | proposed top2 | 0.9844 ± 4.029e-04 | 0.179 ± 0.041 |
| etth1 | alg1 top3 | 0.9844 ± 0.0000 | 0.176 ± 0.000 |
| etth1 | proposed top3 | 0.9813 ± 2.425e-04 | -0.136 ± 0.025 |
| sunspots | alg1 (full softmax) | 0.1480 ± 0.0000 | 0.000 ± 0.000 |
| sunspots | proposed (full softmax) | 0.1479 ± 5.249e-05 | -0.061 ± 0.035 |
| sunspots | alg1 top1 | 0.1481 ± 0.0000 | 0.033 ± 0.000 |
| sunspots | proposed top1 | 0.1481 ± 0.0000 | 0.033 ± 0.000 |
| sunspots | alg1 top2 | 0.1480 ± 0.0000 | 0.000 ± 0.000 |
| sunspots | proposed top2 | 0.1479 ± 2.715e-05 | -0.085 ± 0.018 |
| sunspots | alg1 top3 | 0.1480 ± 0.0000 | 0.000 ± 0.000 |
| sunspots | proposed top3 | 0.1479 ± 4.584e-05 | -0.064 ± 0.031 |
