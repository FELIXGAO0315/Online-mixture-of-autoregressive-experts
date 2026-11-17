# Online Mixture of Autoregressive Experts — Learning to Gate Levinson–Durbin Predictors

ECE 685D final project (Project 1). An online forecaster for a vector time series
$X_1, X_2, \dots \in \mathbb{R}^n$ that combines $L$ vector-AR experts of orders $1..L$ through a softmax gate:

$$\tilde X_m = \sum_{i=1}^{L} \gamma_i\, Y_m(i), \qquad \gamma = \mathrm{softmax}\big(\mathrm{center}(\log w) + g_\theta(u)\big).$$

* **Experts** – VAR($p$) models re-fitted at every step from exponentially weighted autocovariances with one
  run of the (multichannel, Whittle) **Levinson–Durbin** recursion, with a stability guard.
* **Backbone** – the fixed rule of Algorithm 1 of Ding, Zhou & Tarokh [1] (exponential weighting + fixed share), $w$.
* **Learned residual** – a small MLP $g_\theta$, zero at initialization (so the system starts *exactly* as
  Algorithm 1), trained online with one AdamW step per time step and weight decay pulling it back to Algorithm 1.
* **Baselines** – Algorithm 1 alone, online FPE order selection, RLS linear stacking, the best single expert in
  hindsight; references: uniform average, segment oracle, per-step oracle; plus the full "learning ladder"
  (Hedge → Algorithm 1 → learned $(\eta,\alpha)$ → + residual → plain MLP gate).

Everything is strictly online (no look-ahead, verified bit-for-bit), runs on a laptop CPU, and **one command
reproduces every table and figure** in [`outputs/`](outputs/).

---

## Contents

1. [Quick start](#1-quick-start)
2. [Results at a glance](#2-results-at-a-glance)
3. [Repository layout](#3-repository-layout)
4. [Pipeline / workflow](#4-pipeline--workflow)
5. [Method](#5-method)
6. [Evaluation protocol](#6-evaluation-protocol)
7. [Where every figure and table comes from](#7-where-every-figure-and-table-comes-from)
8. [Implementation decisions not fixed by the project description](#8-implementation-decisions-not-fixed-by-the-project-description)
9. [Tests](#9-tests)
10. [References](#10-references)

---

## 1. Quick start

```bash
# Python >= 3.10; a CPU build of PyTorch is enough
pip install -r requirements.txt          # numpy, scipy, pandas, matplotlib, torch, pytest

python experiments/run_all.py            # unit tests + steps 01..06, writes outputs/ and outputs/SUMMARY.md
```

| Command | What it does | Time (16-core laptop, 12 workers) |
|---|---|---|
| `python experiments/run_all.py` | full pipeline: tests, data, expert checks, tuning, sanity checks, main experiments, ablations | ≈ 14 min from scratch (816 s measured) |
| `python experiments/run_all.py --quick` | reduced tuning grids, 1 ablation seed (smoke test); writes to `outputs_quick/` so it never overwrites the real results | a few minutes |
| `python experiments/run_all.py --from 05` | resume from a step (earlier outputs are reused) | – |
| `python experiments/run_all.py --jobs 4` | limit the number of worker processes (default `min(8, cpus-1)`) | – |
| `python experiments/05_main.py --plots-only` | redraw main figures/tables from cached runs | ≈ 15 s |
| `python -m pytest -q` | unit tests only | ≈ 10 s |

Each step can also be run on its own (`python experiments/0X_name.py --help`), in the order 01 → 06.
The real data (ETTh1, SILSO sunspots) are downloaded automatically by step 01 into `data/raw/`.

> **Reproducibility.** All seeds are fixed (`omar/config.py`): synthetic data and gate initialization use
> seeds 0, 1, 2; tuning uses a separate synthetic stream with seed 1000. The SILSO file is updated monthly,
> so the loader keeps only *definitive* values up to `SUNSPOT_LAST_MONTH = 2026-06` (N = 3330).
> Computations are in float64; parallel workers are single-threaded, so results do not depend on `--jobs`.

---

## 2. Results at a glance

All numbers are produced by `python experiments/run_all.py`; every table is collected in
[`outputs/SUMMARY.md`](outputs/SUMMARY.md).

**Cumulative MSE relative to the best single AR expert in hindsight** (%, mean ± std over 3 seeds; negative =
better than the best expert; bold = best of the four causal methods when not tied).
Source: [`outputs/05_main/tables/relative_to_best_expert.md`](outputs/05_main/tables/relative_to_best_expert.md).

| Method | n=1, T=500 | n=1, T=2000 | n=1, T=∞ | n=3, T=500 | n=3, T=2000 | n=3, T=∞ | drift AR(1) | ETTh1 (n=7) | sunspots |
|---|---|---|---|---|---|---|---|---|---|
| **Alg. 1 + residual MLP (proposed)** | **−1.94 ± 0.43** | **−1.39 ± 0.30** | 0.05 ± 0.02 | **−4.32 ± 0.76** | **−3.39 ± 0.05** | 0.03 ± 0.01 | −0.72 ± 0.20 | **−1.03 ± 0.00** | 0.96 ± 0.04 |
| Algorithm 1 (fixed share) | −1.88 ± 0.44 | −1.38 ± 0.29 | 0.05 ± 0.02 | −4.25 ± 0.81 | −3.28 ± 0.09 | 0.03 ± 0.01 | −0.72 ± 0.21 | −0.69 ± 0.00 | 1.02 ± 0.00 |
| FPE order selection | 1.08 ± 0.19 | −0.60 ± 0.13 | 0.14 ± 0.02 | 0.95 ± 0.90 | −1.45 ± 0.75 | 0.32 ± 0.03 | −0.45 ± 0.08 | 1.70 ± 0.00 | 0.72 ± 0.00 |
| RLS stacking | −0.82 ± 0.43 | −0.56 ± 0.24 | 0.27 ± 0.02 | −1.65 ± 0.51 | −1.41 ± 0.27 | 0.16 ± 0.03 | −0.09 ± 0.04 | −0.87 ± 0.00 | **−0.61 ± 0.00** |
| *Uniform average (ref.)* | 1.31 ± 0.24 | 1.79 ± 0.59 | 6.98 ± 2.63 | 0.05 ± 0.56 | 2.06 ± 1.26 | 5.72 ± 2.60 | 1.52 ± 0.53 | 0.79 ± 0.00 | −0.39 ± 0.00 |
| *Segment oracle (ref.)* | −3.10 ± 0.51 | −1.65 ± 0.23 | 0.00 ± 0.00 | −4.27 ± 0.71 | −3.35 ± 0.20 | 0.00 ± 0.00 | −0.41 ± 0.24 | −0.53 ± 0.00 | −0.83 ± 0.00 |

(On real data the series is fixed, so only the neural gates vary across seeds.) Absolute cumulative MSE and
all three regrets: `outputs/05_main/tables/main_{synthetic_n1,synthetic_n3,real}.md`.

**Key findings**

1. **Piecewise-stationary synthetic data — fixed share is hard to beat, as expected.** Algorithm 1 essentially
   reaches the segment oracle (regret to it ≈ 0) and beats the best single expert by 1.4–4.3 %. The learned
   residual improves on it on every switching stream, consistently but modestly (0.01–0.12 %; better on 3/3 seeds
   for n=1, T=500 and both n=3 streams; [`proposed_vs_alg1.md`](outputs/05_main/tables/proposed_vs_alg1.md)).
   The gate recovers the true order 2 → 6 → 3 → 8 and re-adapts ≈ 80–170 steps after a switch, while FPE stays
   stuck at a low order after a third of the n=3, T=500 switches
   ([`gamma_zoom_syn_n3_T500.png`](outputs/05_main/figures/gamma_zoom_syn_n3_T500.png),
   [`readaptation.md`](outputs/05_main/tables/readaptation.md)).
2. **Stationary data (sanity check b).** Mixture, Algorithm 1 and the best expert agree within 0.05 %; FPE is
   0.14–0.32 % behind ([`04_sanity`](outputs/04_sanity/tables/)).
3. **Multichannel real data is where learning pays most.** On ETTh1 the proposed gate is the best method
   (−1.03 % vs the best expert, −0.34 % vs Algorithm 1 on every seed, ahead of RLS stacking). Giving the gate the
   recent X values doubles the gain (−0.81 % vs Algorithm 1), and adding non-AR experts (seasonal naive lag 24 +
   exponential smoothing) lowers its MSE from 0.979 to **0.901 (−8 %)**, while Algorithm 1 only reaches 0.971 and
   RLS 0.925 ([`c_heterogeneous_experts.md`](outputs/06_ablation/tables/c_heterogeneous_experts.md)).
4. **Sunspots (short, n=1): the convex mixtures lose.** RLS stacking (−0.61 %) and even the uniform average beat
   the best expert. Tuned on the first 666 months, Algorithm 1 picks η = 100, α = 0 (≈ follow-the-leader) and
   λ = 0.99; both are sub-optimal on the remaining 80 % (with η = 0.1 chosen in hindsight it would reach a cum. MSE
   of 0.1458 instead of 0.1480, and λ = 0.999 lowers every method to ≈ 0.141, `a_forgetting_lambda_cum_mse.md`).
   With this little tuning data, tuning and the convexity constraint are the bottleneck, not the gating rule.
5. **Learning ladder** ([`b_learning_ladder.md`](outputs/06_ablation/tables/b_learning_ladder.md),
   [`learning_ladder.png`](outputs/06_ablation/figures/learning_ladder.png)): Hedge → Algorithm 1 (the mixing
   step) is by far the largest gain on switching data (0.8–3.3 %); learning (η, α) online buys ≈ 0 and can hurt
   (+3.5 % on sunspots, +0.5 % on stationary n=3); the residual adds a small, consistent gain at a small variance
   cost; the plain MLP gate without the backbone is worse than Algorithm 1 on every synthetic dataset (+0.1 to
   +1.6 %) and, on real data (where only the gate initialization varies), has by far the largest seed-to-seed
   spread (std 0.48 % on sunspots vs 0.04 % for the residual).
6. **The residual is safe in a well-defined range** ([`lr_wd_sweep.png`](outputs/06_ablation/figures/lr_wd_sweep.png),
   [`b_lr_wd_sweep.md`](outputs/06_ablation/tables/b_lr_wd_sweep.md)): for lr ≤ 10⁻³ and weight decay ≤ 0.1 the
   proposed method is never worse than Algorithm 1 on average on any ablation dataset (single seeds at most
   +0.28 %); at lr = 3·10⁻³ it loses up to 0.37 %, at lr = 10⁻² up to 3.4 % — strong weight decay (1.0) damps
   this, as designed. A replay buffer (B = 8, 32) and top-k sparse gating do *not* help.
7. **When does the mixture beat the best single expert?** Its gain tracks the *switching potential*
   1 − (segment oracle / best expert): large on switching data, zero on stationary data
   ([`d_gain_vs_switching_potential.png`](outputs/06_ablation/figures/d_gain_vs_switching_potential.png),
   [`d_characterization.md`](outputs/06_ablation/tables/d_characterization.md)). With AR experts only, Algorithm 1
   already captures most of it, so the learned residual has little room left. **Proposed improvement:** give the
   gate what Algorithm 1 cannot use — heterogeneous experts and observable state (recent X) — which is exactly
   where the learned gate pulled ahead (finding 3).

---

## 3. Repository layout

```
.
├── README.md                     this file
├── requirements.txt, pyproject.toml
├── docs/method_notes.md          derivations of every component (for the report's method section)
├── omar/                         the library: every algorithm lives here
│   ├── config.py                 single source of truth: paths, protocol constants, datasets, tuning grids
│   ├── data/
│   │   ├── synthetic.py          regime-switching VAR, drifting AR(1), stable-coefficient sampler, exact autocovariances
│   │   ├── real.py               download + loaders for ETTh1 and SILSO sunspots
│   │   ├── standardize.py        causal per-channel standardization (expanding / warm-up statistics)
│   │   └── datasets.py           registry: dataset key + seed -> standardized stream, windows, segments, ground truth
│   ├── experts/
│   │   ├── levinson.py           scalar + Whittle (multichannel) Levinson-Durbin, stability guard, direct solve
│   │   ├── ar_bank.py            online bank of VAR(1..L) experts (EW autocovariances + LD at every step)
│   │   └── simple_experts.py     seasonal naive and exponential smoothing experts (ablation c)
│   ├── combiners/
│   │   ├── fixed_share.py        Algorithm 1 of [1] / Hedge (+ vectorized tuning grid, top-k)
│   │   ├── neural_gate.py        PROPOSED residual gate and the plain MLP gate (PyTorch, AdamW)
│   │   ├── features.py           scale-free gate inputs (error window, EWMAs, log w, X, log det Sigma)
│   │   ├── learned_share.py      Algorithm 1 with (eta, alpha) learned online (forward-mode / RTRL gradients)
│   │   └── baselines.py          uniform average, online FPE order selection, RLS stacking
│   ├── stream.py                 ExpertStream / RunResult containers
│   ├── metrics.py                cumulative MSE, decayed average, regrets, oracles
│   ├── analysis.py               re-adaptation after regime switches
│   ├── tuning.py                 tuning stages A (lam, eta, alpha) and B (gate hyperparameters)
│   ├── pipeline.py               cached expert streams, method registry, run_job, parallel_map
│   ├── plotting.py               figure style (fixed color per method) and helpers
│   └── reporting.py              logging, CSV + Markdown tables, "mean ± std" formatting
├── experiments/                  one script per pipeline step (run in this order)
│   ├── 01_prepare_data.py        download + describe datasets
│   ├── 02_check_experts.py       project Step 2: LD vs direct solve, convergence, Sigma_p flattening, guard
│   ├── 03_tune.py                Sec. 1.5: hyperparameter tuning -> outputs/03_tuning/hparams.json
│   ├── 04_sanity.py              project Step 3: sanity checks (a), (b) + end-to-end no-look-ahead test
│   ├── 05_main.py                project Step 4: all methods x all datasets x 3 seeds, figures
│   ├── 06_ablation.py            project Step 5: ablations (a)-(d)
│   ├── run_all.py                runs everything in order + writes outputs/SUMMARY.md
│   └── _bootstrap.py             import path / single-threaded BLAS (imported by every script)
├── tests/                        pytest unit tests (Levinson-Durbin vs direct solve, causality, update rules)
├── data/raw/                     downloaded data (git-ignored; created by step 01)
└── outputs/                      ALL results, one folder per step
    ├── SUMMARY.md                every key table in one file (auto-generated)
    ├── 01_data/  02_experts/  03_tuning/  04_sanity/  05_main/  06_ablation/
    │     └── tables/*.csv + *.md,  figures/*.png + *.pdf
    ├── logs/                     console log of every step (git-ignored)
    └── cache/                    expert forecasts and per-run arrays (git-ignored, re-creatable)
```

---

## 4. Pipeline / workflow

```
01 data ─► 02 experts ─► 03 tuning ─► 04 sanity ─► 05 main ─► 06 ablation ─► outputs/SUMMARY.md
   │                        │                         │            ▲
   ▼                        ▼                         └────────────┘
data/raw/          03_tuning/hparams.json      05_main/tables/main_results_long.csv
                   (read by 04, 05 and 06)     (06 reuses it for the ladder and (d))

shared cache: outputs/cache/experts/*.npz = expert forecasts, computed once, reused by every step and method
```

| Step | Script | Project description | What it produces |
|---|---|---|---|
| 01 | `01_prepare_data.py` | Sec. 1.2 Dataset | downloads ETTh1 + sunspots; dataset table; overview figures |
| 02 | `02_check_experts.py` | **Step 2** Implement the experts | LD = direct solve (all orders, $n \in \{1,2,3,7\}$); timing vs $L$; coefficient and $\Sigma_p$ convergence on $T=\infty$, flattening at $p_0 = 4$; guard statistics; order identifiability |
| 03 | `03_tune.py` | Sec. 1.5 Hyperparameters | stage A: $\lambda \times \eta \times \alpha$ for Algorithm 1; stage B: residual (lr, wd), plain gate (lr, wd), learned-$(\eta,\alpha)$ lr, SES constant → `hparams.json` |
| 04 | `04_sanity.py` | **Step 3** sanity checks | (a) frozen residual = Algorithm 1; (b) stationary data: mixture ≈ best expert ≈ FPE; (c) no look-ahead |
| 05 | `05_main.py` | **Step 4** Experiments | main tables (synthetic $n=1$, synthetic $n=3$, real), $\gamma$ trajectories vs true order, re-adaptation, error-vs-time and regret curves |
| 06 | `06_ablation.py` | **Step 5** Ablation and discussion | (a) $L$, $\lambda$, $K$, features, replay; (b) learning ladder, lr × wd sweep, top-$k$; (c) non-AR experts; (d) when does the mixture win |

**Why it is fast and still strictly online.** The experts never read anything the combiner produces, so
running the expert bank over the whole stream first (row $m$ computed from $X_{1..m-1}$ only) and then each
combiner over the cached forecasts is *identical* to interleaving the six sub-steps of the online update.
The expert forecasts are cached in `outputs/cache/experts/` and shared by every method (as required: "all
run on the same stream, with the same experts"), and independent (dataset, seed, method) jobs run in parallel
processes. The equivalence and causality are checked by perturbing the raw series from a time $m^\star$ on and
verifying that every forecast and gate weight used up to $m^\star$ is bit-identical (step 04 (c), unit tests).

---

## 5. Method

Full derivations: [`docs/method_notes.md`](docs/method_notes.md).

**Online update at every step $m$** (project Sec. 1.3): (1) experts predict $Y_m(1..L)$; (2) the gate predicts
$\gamma_m$; (3) $X_m$ is observed; (4) $\hat R(k) \leftarrow \lambda \hat R(k) + (1-\lambda) X_m X_{m-k}^\top$ and Levinson–Durbin is
re-run; (5) the backbone takes the exponential + fixed-share step; (6) one AdamW step on $\|X_m - \tilde X_m\|^2$ with
$Y_m$ and $\log w$ detached.

| Component | Implementation | Key choices |
|---|---|---|
| AR experts | `omar/experts/ar_bank.py`, `levinson.py` | $L = 8$; $\hat R(0) \mathrel{+}= \epsilon\,\mathrm{tr}(\hat R(0))/n\,I$, $\epsilon = 10^{-3}$; guard $\sigma_{\max}(\Sigma_f^{-1/2}\Delta\Sigma_b^{-\top/2}) < 1 - 10^{-4}$; failed orders keep last step's coefficients |
| Algorithm 1 | `omar/combiners/fixed_share.py` | log-space updates; $(\eta, \alpha)$ tuned |
| Residual gate | `omar/combiners/neural_gate.py`, `features.py` | $u \in \mathbb{R}^{(K+3)L}$ ($K = 5$ error vectors, EWMAs $\beta \in \{0.9, 0.99\}$, $\mathrm{center}(\log w)$) made scale-free by centered logs; hidden 32 (2,344 parameters); $W_2 = b_2 = 0$; `torch.optim.AdamW`, lr $\le 10^{-3}$, wd tuned; float64 |
| FPE | `omar/combiners/baselines.py` | $\log\det\Sigma_p + n\log\frac{N_\text{eff}+pn+1}{N_\text{eff}-pn-1}$, $N_\text{eff} = 1/(1-\lambda)$, one-hot |
| RLS stacking | `omar/combiners/baselines.py` | unconstrained $w \in \mathbb{R}^L$ shared across channels, $P_0 = I$, $w_0 = 1/L$, $\lambda_\text{rls} = 1$ |
| Learned $(\eta,\alpha)$ | `omar/combiners/learned_share.py` | $\eta = e^a$, $\alpha = \sigma(b)$, exact forward-mode gradient through the weight recursion, Adam |
| Plain MLP gate | `omar/combiners/neural_gate.py` (`backbone=None`) | logits $= g_\theta(u)$ only (no $\log w$ input) |

---

## 6. Evaluation protocol

| Item | Setting |
|---|---|
| Datasets | synthetic regime-switching VAR, $n \in \{1, 3\}$, length $10^4$, $T \in \{500, 2000, \infty\}$ (19 / 4 / 0 switches), order cycle 2→6→3→8, coefficients redrawn per regime (spectral radius 0.9), $N(0, I)$ noise, 200-step burn-in per regime; drifting AR(1) of [1, Ex. 4] (0.9 → −0.9 over the first half, then 0.8); ETTh1 (7 channels, 17,420 steps, `date` dropped); SILSO monthly sunspots (3,330 steps) |
| Warm-up | $T_0 = 5L = 40$ steps: initialize $\hat R$, LD, the gate's error window; backbone uniform at $T_0$; nothing scored before |
| Standardization | per channel, $z_m = (x_m - \mu_{m-1})/\sigma_{m-1}$ with expanding-window statistics of the past (warm-up statistics inside the warm-up) |
| Tuning | synthetic: a separate stream (seed 1000); real: the first 20 % of the series. Tuned: $\lambda$, Algorithm 1's $(\eta, \alpha)$, the residual's lr and weight decay (+ ladder / ablation-only parameters) |
| Evaluation | synthetic: seeds 0, 1, 2 (data + gate init), $m \in [T_0, 10^4)$; real: the remaining 80 %, processed exactly once (the series is run online from $T_0$, only the last 80 % is scored), gate seeds 0, 1, 2 |
| Metrics | cumulative MSE $\frac1M\sum\ell_m$ with $\ell_m = \lVert X_m - \tilde X_m\rVert^2$; decayed average ($\beta = 0.99$); regret to the best fixed expert, to the segment oracle (true regimes; 500-step blocks on real data), to the per-step oracle; mean ± std over seeds |

---

## 7. Where every figure and table comes from

All paths are relative to `outputs/`. Every table exists as `.csv` and `.md`, every figure as `.png` and `.pdf`.
To regenerate one item, run its step (or `run_all.py --from <step>`); steps 05/06 also accept cached results
(`05_main.py --plots-only`).

| Output | Content | Project item |
|---|---|---|
| **Step 01 – `experiments/01_prepare_data.py`** | | |
| `01_data/tables/datasets` | sizes, warm-up, tuning / evaluation windows, segments | Sec. 1.2, 1.5 |
| `01_data/figures/synthetic_overview` | the six regime-switching streams (seed 0) shaded by true order | Sec. 1.2 |
| `01_data/figures/drift_overview` | drifting AR(1) and its true coefficient | Sec. 1.2 |
| `01_data/figures/etth1_overview`, `sunspots_overview` | real series with the 20 % / 80 % split | Sec. 1.2 |
| **Step 02 – `experiments/02_check_experts.py`** | | |
| `02_experts/tables/levinson_vs_direct` | LD vs `np.linalg.solve` (all orders, n = 1, 2, 3, 7): max error ~1e-16 | Step 2, Sec. 1.4 |
| `02_experts/tables/levinson_timing_vs_L` | all orders at once: LD vs L direct solves, L = 8…64 | Step 1 (ii) |
| `02_experts/figures/coef_convergence` (+ table) | relative error of the AR(4) coefficients vs time, λ ∈ {0.99, 0.999, 0.9999} | Step 2 |
| `02_experts/figures/sigma_vs_order` (+ table) | $(1/n)\log\det\Sigma_p$ vs p: online vs exact population, flattening at $p_0 = 4$ | Step 2 |
| `02_experts/figures/sigma_over_time` | $\Sigma_p$ trajectories, p = 1..8 | Step 2 |
| `02_experts/tables/guard_stats` | how often the stability guard stops the recursion, per dataset and λ | Sec. 1.3 |
| `02_experts/tables/order_identifiability` | population gain of the true last lag under our generator | Sec. 1.2 |
| **Step 03 – `experiments/03_tune.py`** | | |
| `03_tuning/hparams.json`, `tables/selected_hparams` | selected λ, η, α, lr, wd (+ ladder / ablation parameters) | Sec. 1.5 |
| `03_tuning/tables/alg1_grid.csv`, `lambda_selection`, `gate_grid`, `ses_grid` | full tuning grids | Sec. 1.5 |
| `03_tuning/figures/alg1_grid_heatmaps` | Algorithm 1 tuning loss over (η, α) at the selected λ | Sec. 1.5 |
| **Step 04 – `experiments/04_sanity.py`** | | |
| `04_sanity/tables/a_frozen_residual_equals_alg1` | sanity check (a): max \|γ − w\| < 1e-13 on every dataset | Step 3 (a) |
| `04_sanity/tables/b_stationary_matches_best_and_fpe` | sanity check (b) on T = ∞, 3 seeds | Step 3 (b) |
| `04_sanity/tables/c_no_lookahead` | perturbed future → everything before m* bit-identical | Sec. 1.5 |
| **Step 05 – `experiments/05_main.py`** | | |
| `05_main/tables/main_synthetic_n1`, `main_synthetic_n3`, `main_real` | cum. MSE + regrets (best expert / segment oracle / per-step oracle) for every method and reference, mean ± std | Step 4, Sec. 1.7 |
| `05_main/tables/relative_to_best_expert`, `proposed_vs_alg1` | compact comparisons (table above) | Step 4 |
| `05_main/tables/readaptation` (+ `readaptation_per_switch.csv`) | re-adaptation time and switch cost | Step 4 |
| `05_main/tables/main_results_long.csv` | every (dataset, seed, method) row | – |
| `05_main/figures/gamma_<dataset>`, `gamma_zoom_<dataset>` | $\gamma_i$ trajectories (proposed, Algorithm 1) and mean order vs true order | Step 4, Sec. 1.7 |
| `05_main/figures/drift_tracking` | AR(1) coefficient tracking, gate weights, decayed loss | Step 4 |
| `05_main/figures/error_curves_<group>` | decayed running average of the loss, and its excess over the best expert | Step 4, Sec. 1.7 |
| `05_main/figures/regret_curves_<group>` | cumulative regret to the best expert over time | Step 4 |
| `05_main/figures/switch_response` | loss aligned at regime switches | Step 4 |
| **Step 06 – `experiments/06_ablation.py`** | | |
| `06_ablation/tables/a_num_experts_L_*` | L ∈ {4, 8, 12} | Step 5 (a) |
| `06_ablation/tables/a_forgetting_lambda_*`, `figures/lambda_ablation` | λ ∈ {0.97 … 0.9995} | Step 5 (a) |
| `06_ablation/tables/a_K`, `a_features`, `a_replay` | error window K, gate input features, replay buffer | Step 5 (a), Sec. 1.3 |
| `06_ablation/tables/b_learning_ladder`, `figures/learning_ladder` | Hedge → Alg. 1 → learned (η, α) → + residual → plain MLP, with seed variance | Step 5 (b) |
| `06_ablation/tables/b_lr_wd_sweep`, `figures/lr_wd_sweep` | residual lr × weight decay | Step 5 (b) |
| `06_ablation/tables/b_topk_gating` | top-k sparse gating, k ∈ {1, 2, 3} | Step 5 (b) |
| `06_ablation/tables/c_heterogeneous_experts` | + SES (+ seasonal naive 24 on ETTh1) | Step 5 (c) |
| `06_ablation/tables/d_characterization`, `figures/d_gain_vs_switching_potential` | when the mixture beats the best single expert | Step 5 (d) |
| `06_ablation/tables/ablation_results_long.csv` | every ablation row | – |

Ablations (a)–(c) use the datasets `syn_n1_T500`, `syn_n3_T500`, `etth1`, `sunspots` × 3 seeds
(`config.ABLATION_KEYS`); the ladder and (d) use all nine datasets from step 05.

---

## 8. Implementation decisions not fixed by the project description

| Decision | Choice and reason |
|---|---|
| Scale of the Gaussian coefficient draws | $A_k \sim N(0, 0.1^2/n)$, then $A_k \leftarrow c^k A_k$ with $c = 0.9/\rho$, which multiplies every companion eigenvalue by $c$ (spectral radius exactly 0.9). A small initial scale keeps $c$ near 1 so that high lags are not geometrically suppressed; with unit-variance draws the order-8 lag improves $\log\det\Sigma$ by < 0.01 and the true order is unidentifiable. See `02_experts/tables/order_identifiability.md`. |
| Experts with no valid fit yet | an order that has never passed the guard copies the highest valid order (zero-padded) instead of predicting 0 |
| Bias correction of $\hat R$ | divide by $1 - \lambda^m$ before LD, so $\Sigma_p$ is in data units from the first steps (coefficients are scale-invariant) |
| Standardization | expanding-window statistics (causal) for every dataset, which also handles the level shifts of ETTh1 |
| Tuning data for synthetic sets | a separate stream (seed 1000) so the evaluation seeds are never seen during tuning |
| Selecting λ | jointly with $(\eta, \alpha)$ by Algorithm 1's tuning loss; every method then uses the same experts |
| Segment oracle | true regimes on synthetic data (two halves for the drifting AR(1)), 500-step blocks aligned with the evaluation window on real data |
| Gate warm-up | error window / EWMAs filled with the last warm-up errors; the gate is not trained during warm-up |
| Feature floor | $\mathrm{slog}(e) = \mathrm{center}(\log(e + 0.01\,\bar e))$: scale-free, bounded when an error is ≈ 0 |
| Learned $(\eta, \alpha)$ | starts from the tuned values (so, like the residual, it can only move away from tuned Algorithm 1 when the data reward it) |
| Plain MLP gate | its own (wider) lr grid; inputs exclude $\log w$ |
| Sunspots | definitive values only, up to 2026-06 (the file is updated monthly) |
| Precision / speed | float64 everywhere (sanity check (a) holds to $10^{-13}$); fused AdamW kernel; single-threaded workers, process-level parallelism |
| Re-adaptation time | steps until the gate's mean order crosses the midpoint between the old and the new true order and stays there for 25 steps |

---

## 9. Tests

```bash
python -m pytest -q        # 24 tests, ~10 s
```

* `tests/test_levinson.py` – Levinson–Durbin (scalar and Whittle) against `np.linalg.solve` of the
  block-Toeplitz Yule–Walker system for $n \in \{1,2,3,7\}$ and all orders; exact recovery of $A$ and
  $\Sigma_p = I$ ($p \ge p_0$) from population autocovariances; the stability guard on invalid input; stability of
  the generator.
* `tests/test_online.py` – causality of standardization, experts and gates under a perturbed future; frozen
  residual = Algorithm 1; the fixed-share update rule; vectorized grid = streaming implementation; RLS = ridge
  closed form; RTRL gradient of the learned $(\eta,\alpha)$ = finite differences.

---

## 10. References

1. J. Ding, J. Zhou, and V. Tarokh, "Asymptotically optimal prediction for time-varying data generating
   processes," *IEEE Transactions on Information Theory*, vol. 65, no. 5, p. 3034 ff., 2019.
2. H. Zhou et al., "Informer: Beyond efficient transformer for long sequence time-series forecasting," AAAI 2021
   (ETDataset, <https://github.com/zhouhaoyi/ETDataset>).
3. SILSO World Data Center, "Monthly mean total sunspot number," Royal Observatory of Belgium,
   <https://www.sidc.be/SILSO/datafiles>.
4. H. Akaike, "Statistical predictor identification," *Annals of the Institute of Statistical Mathematics*, 1970 (FPE).
5. M. Herbster and M. K. Warmuth, "Tracking the best expert," *Machine Learning*, 1998 (fixed share).
6. P. Whittle, "On the fitting of multivariate autoregressions, and the approximate canonical factorization of a
   spectral density matrix," *Biometrika*, 1963 (multichannel Levinson–Durbin).
7. D. H. Wolpert, "Stacked generalization," *Neural Networks*, 1992; L. Breiman, "Stacked regressions," *Machine Learning*, 1996.
8. I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," ICLR 2019 (AdamW).
