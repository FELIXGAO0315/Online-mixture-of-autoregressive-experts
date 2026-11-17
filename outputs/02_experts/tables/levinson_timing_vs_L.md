### Wall-clock time to obtain all orders 1..L (n=3): Levinson-Durbin vs L direct solves. At L=8 Python overhead dominates; the O(L^2 n^3) vs O(L^4 n^3) gap shows as L grows.

| n | L | levinson_all_orders_ms | direct_all_orders_ms | speedup |
|---|---|---|---|---|
| 3 | 8 | 0.574 | 0.231 | 0.403 |
| 3 | 16 | 0.676 | 1.055 | 1.560 |
| 3 | 32 | 1.430 | 6.523 | 4.562 |
| 3 | 64 | 3.422 | 47.804 | 13.970 |
