"""Logging and table helpers: every table is written both as CSV (machine-readable) and Markdown."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


class Logger:
    """Print with a timestamp and append to a log file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self.t0 = time.time()

    def __call__(self, msg: str = "") -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def fmt_num(x, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if isinstance(x, (int, np.integer)):
        return str(x)
    if isinstance(x, (float, np.floating)):
        ax = abs(x)
        if ax != 0 and (ax >= 1e5 or ax < 1e-3):
            return f"{x:.{max(1, digits - 1)}e}"
        if ax >= 100:
            return f"{x:.1f}"
        return f"{x:.{digits}f}"
    return str(x)


def df_to_markdown(df: pd.DataFrame, digits: int = 4, index: bool = False) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in d.itertuples(index=False):  # itertuples keeps per-column dtypes (ints stay ints)
        lines.append("| " + " | ".join(fmt_num(v, digits) for v in row) + " |")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, stem: Path, title: str | None = None, note: str | None = None,
               digits: int = 4, index: bool = False) -> None:
    """Write ``stem.csv`` and ``stem.md``."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=index)
    md = []
    if title:
        md.append(f"### {title}\n")
    md.append(df_to_markdown(df, digits, index))
    if note:
        md.append(f"\n{note}")
    stem.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")


def mean_std(values, digits: int = 4) -> str:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "-"
    m, s = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
    scale = max(1.0, float(np.abs(v).max()))
    m = 0.0 if abs(m) < 1e-9 * scale else m  # exact zeros up to float round-off (e.g. regret of the best expert)
    s = 0.0 if abs(s) < 1e-9 * scale else s
    return f"{fmt_num(m, digits)} ± {fmt_num(s, digits)}"


def summarize(rows: pd.DataFrame, by: list, metrics: list, digits: int = 4) -> pd.DataFrame:
    """Group ``rows`` by ``by`` and format every metric as 'mean ± std' over seeds."""
    out = []
    for keys, g in rows.groupby(by, sort=False, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(by, keys))
        for mtr in metrics:
            rec[mtr] = mean_std(g[mtr].values, digits)
        out.append(rec)
    return pd.DataFrame(out)
