# Method notes

Derivations and implementation details behind each component, written so they can be lifted into the
report's method section. File references point to where each formula is implemented.

---

## 1. Notation and the online loop

Observations $X_1, X_2, \dots \in \mathbb{R}^n$; $L$ AR experts of orders $p = 1..L$; forecasts
$Y_m(p)$ of $X_m$ made **before** $X_m$ arrives. One time step of the system
([`omar/combiners/neural_gate.py`](../omar/combiners/neural_gate.py)):

1. experts predict $Y_m(1..L)$ (from coefficients fitted on $X_{1..m-1}$),
2. the gate predicts $\gamma_m = \mathrm{softmax}(\mathrm{center}(\log w_m) + g_\theta(u_m))$, forecast $\tilde X_m = \sum_i \gamma_{m,i} Y_m(i)$,
3. $X_m$ is observed, loss $\ell_m = \|X_m - \tilde X_m\|^2$, expert errors $e_m(i) = \|X_m - Y_m(i)\|^2$,
4. autocovariances are updated and Levinson–Durbin is re-run,
5. the backbone weights take the fixed-share step,
6. $\theta$ takes one AdamW step on $\ell_m$ with $Y_m$ and $\log w_m$ detached.

**Why the experts can be pre-computed.** Steps 1 and 4 never read anything produced by steps 2, 5 or 6:
the experts do not depend on the combiner. Computing the expert forecasts for the whole stream first
(strictly causally, row $m$ from $X_{1..m-1}$) and then running each combiner over them is therefore
*identical* to interleaving the two loops; it only lets every method share the same experts and the same
cache. Causality is verified end to end in `experiments/04_sanity.py` (c) and `tests/test_online.py`:
perturbing the raw series from $m^\star$ on leaves every forecast and weight used up to $m^\star$ bit-identical.

## 2. Experts: exponentially weighted Yule–Walker + Levinson–Durbin

**Autocovariances** ([`omar/experts/ar_bank.py`](../omar/experts/ar_bank.py)), $k = 0..L$:
$$\hat R_m(k) = \lambda \hat R_{m-1}(k) + (1-\lambda) X_m X_{m-k}^\top,$$
with $X_j = 0$ for $j \le 0$. Before every fit we divide by $1-\lambda^m$ (Adam-style bias correction, so
$\Sigma_p$ is in data units early on; coefficients are invariant to this scale) and regularize
$\hat R(0) \leftarrow \hat R(0) + \epsilon\,\mathrm{tr}(\hat R(0))/n\, I$ with $\epsilon = 10^{-3}$.

**Yule–Walker.** For $X_t = \sum_{k=1}^p A_k X_{t-k} + e_t$, multiplying by $X_{t-j}^\top$ and taking
expectations gives $R(j) = \sum_k A_k R(j-k)$, $j = 1..p$, with $R(-h) = R(h)^\top$; in block form
$[R(1) \cdots R(p)] = [A_1 \cdots A_p]\,\Gamma_p$ with the block-Toeplitz $\Gamma_p[k,j] = R(j-k)$, and
$\Sigma_p = R(0) - \sum_k A_k R(k)^\top$. A direct solve costs $O((pn)^3)$ per order, $O(L^4 n^3)$ for all orders.

**Whittle / Wiggins–Robinson recursion** ([`omar/experts/levinson.py`](../omar/experts/levinson.py)).
Alongside the forward model keep the *backward* model $X_{t-p} = \sum_k B_k X_{t-p+k} + b_t$ with error
covariance $\Sigma_b$. With $\Sigma_f^{(0)} = \Sigma_b^{(0)} = R(0)$, the order update $p-1 \to p$ is
$$\Delta_p = R(p) - \sum_{k=1}^{p-1} A_k R(p-k), \qquad K_f = \Delta_p \Sigma_b^{-1},\quad K_b = \Delta_p^\top \Sigma_f^{-1},$$
$$A_k \leftarrow A_k - K_f B_{p-k},\quad B_k \leftarrow B_k - K_b A_{p-k}\ (k<p), \qquad A_p = K_f,\ B_p = K_b,$$
$$\Sigma_f \leftarrow \Sigma_f - K_f \Delta_p^\top, \qquad \Sigma_b \leftarrow \Sigma_b - K_b \Delta_p.$$
$\Delta_p = E[e^f_{p-1}(t)\, e^b_{p-1}(t-p)^\top]$ is the cross-covariance of the order-$(p-1)$ forward and
backward errors; the new forward error is $e^f_p = e^f_{p-1} - K_f e^b_{p-1}$ (projection of the forward error on
the backward error). Each order costs $O(p n^3)$ (the sum in $\Delta_p$ and the coefficient updates), so all
orders cost $O(L^2 n^3)$ — $O(L^2)$ in the scalar case, where $\Sigma_f = \Sigma_b$, $B = A$ and $\kappa_p = \Delta_p/\sigma^2_{p-1}$
is the reflection (partial autocorrelation) coefficient. One run yields $(A^{(p)}, \Sigma_p)$ for *every*
order, which is what both the expert bank and FPE need. Verified against `np.linalg.solve` to $10^{-15}$
(`tests/test_levinson.py`, `outputs/02_experts/tables/levinson_vs_direct.md`); timing vs $L$ in
`levinson_timing_vs_L.md`.

**Stability guard.** Write $N = \Sigma_f^{-1/2}\Delta_p\Sigma_b^{-\top/2}$ (normalized reflection matrix). Then
$$\Sigma_f^{(p)} = \Sigma_f^{1/2}(I - NN^\top)\Sigma_f^{\top/2}, \qquad \Sigma_b^{(p)} = \Sigma_b^{1/2}(I - N^\top N)\Sigma_b^{\top/2},$$
so both updated covariances are positive definite **iff** $\sigma_{\max}(N) < 1$. The raw reflection
$\Delta_p\Sigma_b^{-1}$ can exceed norm 1 legitimately when $n > 1$; the normalized one cannot. The exponentially
weighted $\hat R(k)$ are *not* guaranteed to form a positive-definite block-Toeplitz matrix (unlike the biased
windowed estimator), so the test can fail — increasingly so when $p n^2$ approaches $N_\text{eff} = 1/(1-\lambda)$.
We require $\sigma_{\max}(N) < 1 - 10^{-4}$ (computed through Cholesky factors). When order $p$ fails, the
recursion stops and experts of order $\ge p$ keep their coefficients and $\Sigma$ from the previous step (an
expert that has never been fitted falls back to the highest valid order). Frequency per dataset/λ:
`outputs/02_experts/tables/guard_stats.md` (ETTh1, $n=7$: 96 % of steps at λ = 0.98, ≈0 % for λ ≥ 0.998).

## 3. Backbone: Algorithm 1 of [1] (EWA with fixed share)

[`omar/combiners/fixed_share.py`](../omar/combiners/fixed_share.py):
$$v_i = w_i e^{-\eta e_m(i)}, \qquad w_i \leftarrow (1-\alpha)\frac{v_i}{\sum_j v_j} + \frac{\alpha}{L}.$$
The first step is Hedge / exponentially weighted averaging (multiplicative weights on the experts' losses);
the second mixes toward uniform (fixed share), keeping every $w_i \ge \alpha/L$ so that an expert that becomes
good after a regime change needs only roughly $\log(L/\alpha)/(\eta\,\Delta e)$ steps to regain the lead (with
$\Delta e$ its per-step error advantage) instead of having to undo an exponentially small weight. With $\alpha = 0$ it is Hedge, which converges to the best fixed expert and
cannot track switches. A natural choice is $\alpha \approx (M_T - 1)/(T - 1)$, the switching rate; on our
streams this is $1.9\cdot10^{-3}$ for $T = 500$ and $4\cdot10^{-4}$ for $T = 2000$, close to the tuned values.

**Why a weighted average of the experts.** The squared error $\ell(x, \hat x) = \|x - \hat x\|^2$ is convex in
$\hat x$, so by Jensen
$$\ell\Big(X_m, \sum_i \gamma_i Y_m(i)\Big) \le \sum_i \gamma_i\, \ell(X_m, Y_m(i))$$
for any $\gamma$ in the simplex: predicting with the average is never worse than drawing an expert at random
from $\gamma$. This is the property used to turn a bound on the weighted-average loss of the experts into a
bound on the loss of the averaged forecast.

**Vectorized tuning.** `fixed_share_grid` runs every $(\eta, \alpha)$ pair of the grid simultaneously
(the update only depends on the errors), and is tested to match the streaming implementation exactly.

## 4. The learned residual gate (proposed method)

[`omar/combiners/neural_gate.py`](../omar/combiners/neural_gate.py), features in
[`omar/combiners/features.py`](../omar/combiners/features.py).
$$z = \mathrm{center}(\log w) + g_\theta(u), \qquad \gamma = \mathrm{softmax}(z), \qquad g_\theta(u) = W_2\,\mathrm{ReLU}(W_1 u + b_1) + b_2.$$
* $W_2 = b_2 = 0$ at initialization, so $g_\theta \equiv 0$ and $\mathrm{softmax}(\mathrm{center}(\log w)) = w$:
  the gate starts **exactly** as Algorithm 1 (sanity check (a): $\max|\gamma - w| < 10^{-13}$, i.e. float64
  round-off, on every dataset).
* Input $u \in \mathbb{R}^{(K+3)L}$: the last $K$ error vectors, their EWMAs with $\beta \in \{0.9, 0.99\}$,
  and $\mathrm{center}(\log w)$. Errors are made scale-free by
  $\mathrm{slog}(e) = \mathrm{center}(\log(e + 0.01\,\overline{e}))$ (log-ratios to the geometric mean; the small
  floor, itself scale-free, bounds the log when an error is ≈ 0).
* Hidden width 32, $K = 5$, $L = 8$: 64 inputs, 2,344 parameters.
* One `torch.optim.AdamW` step per time step on $\ell_m$ with $Y_m$ and $\log w$ as constants; lr ≤ $10^{-3}$
  and weight decay ≈ 0.1 (tuned): decoupled weight decay shrinks $\theta$ toward 0, i.e. back toward
  Algorithm 1, whenever the gradient signal is weak. float64 throughout (so (a) holds to round-off).
* Options (ablations): feature sets `last`, `window`, `window_ewma` (default), `window_ewma_x` (+ last 4 $X$
  vectors), `window_ewma_logdet` (+ $\mathrm{center}(\log\det\Sigma_p)$); replay buffer $B$ (loss averaged over the
  last $B$ steps); top-$k$ sparse gating (softmax over the $k$ largest logits).

**Plain MLP gate** (top of the learning ladder): logits $= g_\theta(u)$ only; $u$ then excludes $\log w$
(there is no backbone), the zero-initialized output layer starts it at the uniform average.

## 5. Learning ladder level 3: (η, α) learned online

[`omar/combiners/learned_share.py`](../omar/combiners/learned_share.py). Parametrize $\eta = e^a$,
$\alpha = \sigma(b)$. $w_m$ depends on $(a, b)$ through the entire history, so we propagate the sensitivity
$D_m = \partial w_m / \partial(a, b) \in \mathbb{R}^{L\times 2}$ forward (real-time recurrent learning):
$$s = \log w - \eta e,\ \ p = \mathrm{softmax}(s),\ \ w' = (1-\alpha)p + \alpha/L,$$
$$\frac{\partial s}{\partial(a,b)} = \frac{D}{w} - [\eta e,\ 0], \qquad \frac{\partial p}{\partial(a,b)} = (\mathrm{diag}(p) - pp^\top)\frac{\partial s}{\partial(a,b)},$$
$$D' = (1-\alpha)\frac{\partial p}{\partial(a,b)} + \big[0,\ \alpha(1-\alpha)(1/L - p)\big].$$
The gradient of $\ell_m$ is $(-2\,Y_m r_m)^\top D_m$ with $r_m = X_m - Y_m^\top w_m$; $(a, b)$ take one Adam step per
time step, starting from the tuned $(\eta, \alpha)$. With a frozen learning rate the gradient is exact, which
`tests/test_online.py::test_learned_share_rtrl_gradient` checks against finite differences.

## 6. Baselines

**FPE** ([`omar/combiners/baselines.py`](../omar/combiners/baselines.py)). In-sample $\Sigma_p$ decreases
monotonically in $p$, so it cannot select the order. Akaike's FPE estimates the out-of-sample one-step error
by inflating it for the $pn^2$ estimated coefficients:
$$\log\mathrm{FPE}(p) = \log\det\Sigma_p + n\log\frac{N_\text{eff} + pn + 1}{N_\text{eff} - pn - 1}, \qquad N_\text{eff} = \frac{1}{1-\lambda}$$
(for $n = 1$: $\hat\sigma_p^2 (N+p+1)/(N-p-1)$). At each step $\hat p_m = \arg\min_p \mathrm{FPE}(p)$ using the
$\Sigma_p$ of the very fit that produced $Y_m$ (a free by-product of Levinson–Durbin), $\gamma$ one-hot;
orders with $N_\text{eff} \le pn + 1$ are excluded.

**RLS stacking.** Unconstrained $w \in \mathbb{R}^L$ shared across channels, one RLS update per channel with
regressor $\phi = (Y_m(1)_c, \dots, Y_m(L)_c)$, $P_0 = I$, $w_0 = 1/L$, $\lambda_\text{rls} = 1$. This is exactly the
ridge solution $\arg\min_w \sum (X_{m,c} - \phi^\top w)^2 + \|w - w_0\|^2$ over all past (step, channel)
pairs (verified in `tests/test_online.py::test_rls_equals_ridge_solution`). It can extrapolate between experts
and correct a shared bias; it is the only method without the simplex constraint.

**Best single expert** (hindsight): $\arg\min_p \sum_m e_m(p)$ over the evaluation window.
**References** (plotted, not baselines): uniform average; segment oracle (best fixed expert inside each true
regime; 500-step blocks on real data; the two halves for the drifting AR(1)); per-step oracle $\min_i e_m(i)$.

## 7. Metrics and protocol

[`omar/metrics.py`](../omar/metrics.py), [`omar/data/datasets.py`](../omar/data/datasets.py),
[`omar/tuning.py`](../omar/tuning.py).

* Warm-up $T_0 = 5L = 40$: initializes $\hat R$, the first Levinson–Durbin passes and the gate's error window
  (window and EWMAs filled with warm-up errors); the backbone starts uniform at $T_0$; nothing is scored before.
* Standardization: $z_m = (x_m - \mu_{m-1})/\sigma_{m-1}$ with expanding-window statistics of $x_{1..m-1}$
  (warm-up statistics inside the warm-up block) — a function of the past only.
* Synthetic data: tuned on a separate stream (seed 1000); evaluated on seeds 0, 1, 2 over $m \in [T_0, 10^4)$.
  Real data: tuned on the first 20 %; the whole series is processed online once and the last 80 % is scored.
* Metrics on the evaluation window: cumulative MSE $\frac1M\sum_m \ell_m$; decayed average
  $\bar\ell_m = \beta\bar\ell_{m-1} + (1-\beta)\ell_m$, $\beta = 0.99$ (bias-corrected); regret to the best fixed
  expert, to the segment oracle and to the per-step oracle (cumulative sums; negative = better than the reference).
* Re-adaptation after a switch: steps until the gate's mean order $\sum_p \gamma_p p$ crosses the midpoint
  between the old and the new true order and stays there for 25 steps; plus the excess loss over the segment
  oracle in the 200 steps after the switch.
