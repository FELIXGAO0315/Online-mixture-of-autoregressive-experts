### Levinson-Durbin vs direct block-Toeplitz solve (biased sample autocovariances of a simulated VAR)

| n | true_order | orders_checked | valid_orders | max_abs_err_coefs | max_abs_err_sigma | levinson_ms | direct_all_orders_ms |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 1..8 | 8 | 4.44e-16 | 6.66e-16 | 0.027 | 0.282 |
| 1 | 5 | 1..8 | 8 | 1.11e-16 | 1.11e-16 | 0.025 | 0.235 |
| 2 | 2 | 1..8 | 8 | 4.54e-16 | 4.44e-16 | 0.309 | 0.222 |
| 2 | 5 | 1..8 | 8 | 8.33e-17 | 2.22e-16 | 0.314 | 0.235 |
| 3 | 2 | 1..8 | 8 | 6.66e-16 | 6.66e-16 | 0.379 | 0.275 |
| 3 | 5 | 1..8 | 8 | 1.67e-16 | 4.44e-16 | 0.353 | 0.262 |
| 7 | 2 | 1..8 | 8 | 9.61e-16 | 5.11e-16 | 0.385 | 0.511 |
| 7 | 5 | 1..8 | 8 | 1.14e-16 | 4.44e-16 | 0.380 | 0.304 |
