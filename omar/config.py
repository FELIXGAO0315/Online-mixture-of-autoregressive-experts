"""Single source of truth for paths, protocol constants, datasets and hyperparameter grids.

Every experiment script imports from here, so changing a value here changes it everywhere.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
# results folder; `run_all.py --quick` redirects it to outputs_quick/ so a smoke test never overwrites results
OUTPUTS = Path(os.environ.get("OMAR_OUTPUTS", ROOT / "outputs"))
CACHE = OUTPUTS / "cache"  # expert forecasts / per-run arrays (re-creatable, git-ignored)

OUT_DATA = OUTPUTS / "01_data"
OUT_EXPERTS = OUTPUTS / "02_experts"
OUT_TUNING = OUTPUTS / "03_tuning"
OUT_SANITY = OUTPUTS / "04_sanity"
OUT_MAIN = OUTPUTS / "05_main"
OUT_ABLATION = OUTPUTS / "06_ablation"

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
SUNSPOT_URL = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv"
ETTH1_FILE = DATA_RAW / "ETTh1.csv"
SUNSPOT_FILE = DATA_RAW / "SN_m_tot_V2.0.csv"
SUNSPOT_LAST_MONTH = (2026, 6)  # last definitive month when the experiments were run (file is updated monthly)

# --------------------------------------------------------------------------------------
# protocol (project description, Sec. 1.5)
# --------------------------------------------------------------------------------------
L_DEFAULT = 8  # number of AR experts (orders 1..L)
K_DEFAULT = 5  # error window of the gate
WARMUP_FACTOR = 5  # T0 = 5 L
SEEDS = (0, 1, 2)  # evaluation seeds (gate init + synthetic data generation)
TUNE_SEED = 1000  # synthetic tuning stream (never used for evaluation)
DECAY_BETA = 0.99  # decayed running average of the loss
REAL_TUNE_FRACTION = 0.2  # real data: tune on the first 20 %, report on the last 80 %
REAL_BLOCK = 500  # real data: segment oracle = best expert per block of 500 steps

# synthetic data (Sec. 1.2)
SYN_LENGTH = 10_000
ORDER_CYCLE = (2, 6, 3, 8)
STATIONARY_ORDER = 4
SPECTRAL_RADIUS = 0.9
BURN_IN = 200

# experts (Sec. 1.3)
LD_EPS = 1e-3  # R(0) <- R(0) + eps tr(R(0))/n I
LD_TOL = 1e-4  # stability guard: sigma_max(normalized reflection) < 1 - tol

# gate architecture (Sec. 1.3)
GATE_HIDDEN = 32
GATE_BETAS = (0.9, 0.99)  # time scales of the error EWMAs used as features
GATE_FEATURES = "window_ewma"  # default feature set -> (K+3) L inputs
GATE_X_LAGS = 4  # number of recent X vectors for the "+X" feature ablation
GATE_REPLAY = 1  # replay buffer size (1 = plain single-sample online step)

# --------------------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------------------
DATASETS: dict[str, dict] = {
    "syn_n1_T500": dict(kind="regime", n=1, T=500, label="Synthetic n=1, T=500"),
    "syn_n1_T2000": dict(kind="regime", n=1, T=2000, label="Synthetic n=1, T=2000"),
    "syn_n1_Tinf": dict(kind="regime", n=1, T=None, label="Synthetic n=1, T=inf"),
    "syn_n3_T500": dict(kind="regime", n=3, T=500, label="Synthetic n=3, T=500"),
    "syn_n3_T2000": dict(kind="regime", n=3, T=2000, label="Synthetic n=3, T=2000"),
    "syn_n3_Tinf": dict(kind="regime", n=3, T=None, label="Synthetic n=3, T=inf"),
    "drift_ar1": dict(kind="drift", n=1, label="Drifting AR(1) [1, Ex. 4]"),
    "etth1": dict(kind="etth1", n=7, label="ETTh1 (n=7)"),
    "sunspots": dict(kind="sunspots", n=1, label="Monthly sunspots (n=1)"),
}
REAL_KEYS = [k for k, v in DATASETS.items() if v["kind"] in ("etth1", "sunspots")]
SWITCHING_KEYS = ["syn_n1_T500", "syn_n1_T2000", "syn_n3_T500", "syn_n3_T2000"]
STATIONARY_KEYS = ["syn_n1_Tinf", "syn_n3_Tinf"]
ABLATION_KEYS = ["syn_n1_T500", "syn_n3_T500", "etth1", "sunspots"]

# --------------------------------------------------------------------------------------
# hyperparameter grids (tuned on a separate synthetic stream / first 20 % of real data)
# --------------------------------------------------------------------------------------
GRID = dict(
    lam=(0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995),  # forgetting factor of the autocovariances
    eta=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),  # Algorithm 1 learning rate
    alpha=(0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),  # Algorithm 1 mixing
    lr=(1e-4, 3e-4, 1e-3),  # residual gate AdamW learning rate (<= 1e-3)
    wd=(0.01, 0.1, 1.0),  # residual gate AdamW weight decay
    plain_lr=(3e-4, 1e-3, 3e-3, 1e-2, 3e-2),  # plain MLP gate (ladder level 5): wider lr range, tuned separately
    plain_wd=(0.01, 0.1),
    learned_lr=(1e-3, 3e-3, 1e-2),  # Adam lr of the learned (eta, alpha) ladder level
    ses_alpha=(0.1, 0.3, 0.5, 0.7, 0.9),  # exponential smoothing expert (ablation c)
)
# reduced grids for a fast end-to-end smoke run (--quick)
GRID_QUICK = dict(lam=(0.99, 0.998), eta=(0.1, 1.0, 3.0), alpha=(0.0, 1e-3, 1e-2), lr=(1e-3,), wd=(0.1,),
                  plain_lr=(1e-3,), plain_wd=(0.1,), learned_lr=(3e-3,), ses_alpha=(0.3, 0.7))


def warmup_length(L: int) -> int:
    return WARMUP_FACTOR * L
