"""Real-data loaders (ETTh1, SILSO monthly sunspots) and the download helper."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config


def download(url: str, dest: Path, force: bool = False) -> Path:
    """Download ``url`` to ``dest`` unless it already exists."""
    dest = Path(dest)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "omar-course-project/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())
    return dest


def ensure_real_data(force: bool = False) -> None:
    download(config.ETTH1_URL, config.ETTH1_FILE, force)
    download(config.SUNSPOT_URL, config.SUNSPOT_FILE, force)


def load_etth1(path: Path = config.ETTH1_FILE) -> tuple[np.ndarray, list[str]]:
    """ETTh1: hourly, 7 channels; the ``date`` column is dropped."""
    df = pd.read_csv(path)
    df = df.drop(columns=["date"])
    return np.ascontiguousarray(df.to_numpy(dtype=float)), list(df.columns)


def load_sunspots(path: Path = config.SUNSPOT_FILE) -> tuple[np.ndarray, list[str]]:
    """SILSO monthly mean total sunspot number (semicolon separated, column 4 = value).

    The file is updated every month and its last few values are provisional (column 7 = 0). For
    reproducibility we keep definitive values only, up to ``config.SUNSPOT_LAST_MONTH``.
    Missing values are marked -1; they are filled causally with the last observed value.
    """
    df = pd.read_csv(path, sep=";", header=None)
    year, month = config.SUNSPOT_LAST_MONTH
    keep = (df.iloc[:, 6] == 1) & ((df.iloc[:, 0] < year) | ((df.iloc[:, 0] == year) & (df.iloc[:, 1] <= month)))
    df = df[keep]
    x = df.iloc[:, 3].to_numpy(dtype=float)
    x[x < 0] = np.nan
    x = pd.Series(x).ffill().bfill().to_numpy()
    return np.ascontiguousarray(x[:, None]), ["sunspot_number"]
