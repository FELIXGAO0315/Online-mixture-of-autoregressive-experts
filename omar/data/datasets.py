"""Dataset registry: ``load_dataset(key, seed)`` returns a ready-to-use, causally standardized stream."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config
from .real import load_etth1, load_sunspots
from .standardize import standardize_online
from .synthetic import drifting_ar1, regime_switching_var


@dataclass
class Dataset:
    key: str
    label: str
    seed: int
    X: np.ndarray  # (N, n) causally standardized observations (what every method sees)
    X_raw: np.ndarray  # (N, n) raw observations
    channels: list
    T0: int  # warm-up length 5L
    eval_start: int  # first evaluated index
    eval_end: int  # one past the last evaluated index
    tune_start: int  # tuning window [tune_start, tune_end) inside *this* stream
    tune_end: int
    segments: np.ndarray  # (N,) segment labels used by the segment oracle
    true_order: np.ndarray | None = None
    true_coef: np.ndarray | None = None  # drifting AR(1) only
    switch_points: list = field(default_factory=list)  # indices where a new regime starts
    regime_coefs: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.X.shape[1]

    @property
    def N(self) -> int:
        return self.X.shape[0]

    @property
    def is_synthetic(self) -> bool:
        return self.true_order is not None


def load_dataset(key: str, seed: int = 0, L: int = config.L_DEFAULT, standardize: str = "expanding",
                 eval_start: int | None = None) -> Dataset:
    """Load dataset ``key`` (see ``config.DATASETS``).

    Synthetic data: ``seed`` drives data generation; tuning is done on a *separate* stream
    (``seed = config.TUNE_SEED``), so the tuning window of an evaluation stream is unused and the
    whole stream after warm-up is evaluated.
    Real data: the series is fixed; the first 20 % is the tuning window and the remaining 80 %
    is evaluated exactly once.
    """
    spec = config.DATASETS[key]
    T0 = config.warmup_length(L)
    true_order = true_coef = None
    switch_points, regime_coefs = [], []
    if spec["kind"] == "regime":
        s = regime_switching_var(spec["n"], config.SYN_LENGTH, spec["T"], seed, config.ORDER_CYCLE,
                                 config.STATIONARY_ORDER, config.SPECTRAL_RADIUS, config.BURN_IN)
        X_raw, true_order, segments = s.X, s.true_order, s.regime_id
        regime_coefs = s.regime_coefs
        channels = [f"x{i + 1}" for i in range(spec["n"])]
    elif spec["kind"] == "drift":
        s = drifting_ar1(config.SYN_LENGTH, seed, config.BURN_IN)
        X_raw, true_order, true_coef, segments = s.X, s.true_order, s.true_coef, s.regime_id
        regime_coefs = s.regime_coefs
        channels = ["x1"]
    elif spec["kind"] == "etth1":
        X_raw, channels = load_etth1()
    elif spec["kind"] == "sunspots":
        X_raw, channels = load_sunspots()
    else:
        raise ValueError(spec["kind"])

    N = X_raw.shape[0]
    X = standardize_online(X_raw, T0, standardize)
    if true_order is not None:  # synthetic
        start = T0 if eval_start is None else eval_start
        tune_start, tune_end = start, N
        switch_points = [int(i) for i in np.flatnonzero(np.diff(segments)) + 1]
    else:  # real
        tune_start, tune_end = T0, int(round(config.REAL_TUNE_FRACTION * N))
        start = tune_end
        segments = np.floor_divide(np.arange(N) - start, config.REAL_BLOCK)
    return Dataset(key=key, label=spec["label"], seed=seed, X=X, X_raw=X_raw, channels=channels, T0=T0,
                   eval_start=start, eval_end=N, tune_start=tune_start, tune_end=tune_end,
                   segments=np.asarray(segments), true_order=true_order, true_coef=true_coef,
                   switch_points=switch_points, regime_coefs=regime_coefs)


def data_seed(key: str, seed: int) -> int:
    """Real data are deterministic; only synthetic data depend on the seed (used for caching)."""
    return seed if config.DATASETS[key]["kind"] in ("regime", "drift") else 0
