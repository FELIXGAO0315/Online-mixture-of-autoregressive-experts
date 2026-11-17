"""Imported first by every experiment script.

* single-threaded BLAS / torch per process (we parallelize over processes instead; tiny matrices
  are slower with threads),
* makes the repository root importable so that ``python experiments/XX_name.py`` works without
  installing the package.
"""
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

for _stream in (sys.stdout, sys.stderr):  # "±" etc. on Windows consoles
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
