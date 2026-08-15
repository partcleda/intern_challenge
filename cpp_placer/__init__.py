"""ctypes bridge to the C++20 placer.

The library is built on first use if it is missing, so `python test.py` works
from a clean checkout with no extra setup step.
"""

import ctypes
import os
import subprocess
import sys

import numpy as np
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_NAME = "libpartcl_place.dylib" if sys.platform == "darwin" else "libpartcl_place.so"
_LIB_PATH = os.path.join(_DIR, "build", _LIB_NAME)

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        subprocess.run(["bash", os.path.join(_DIR, "build.sh")], check=True)
    lib = ctypes.CDLL(_LIB_PATH)
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    lib.partcl_place.restype = ctypes.c_int
    lib.partcl_place.argtypes = [
        ctypes.c_int, f64, f64, f64, f64,          # n, w, h, x, y
        ctypes.c_int, i32, f64, f64,               # n_pins, pin_cell, pin_ox, pin_oy
        ctypes.c_int, i32, i32,                    # n_edges, edge_a, edge_b
        ctypes.c_double, ctypes.c_uint, ctypes.c_int,  # budget_s, seed, verbose
    ]
    _lib = lib
    return lib


def place(cell_features, pin_features, edge_list, budget_s=2.0, seed=12345, verbose=False):
    """Run the C++ placer. Returns (x, y) numpy float64 arrays of cell centres.

    Pin absolute position is `cell_pos + pin_offset`, exactly as
    wirelength_attraction_loss() computes it.
    """
    lib = _load()
    n = int(cell_features.shape[0])

    w = np.ascontiguousarray(cell_features[:, 4].detach().numpy(), dtype=np.float64)
    h = np.ascontiguousarray(cell_features[:, 5].detach().numpy(), dtype=np.float64)
    x = np.ascontiguousarray(cell_features[:, 2].detach().numpy(), dtype=np.float64)
    y = np.ascontiguousarray(cell_features[:, 3].detach().numpy(), dtype=np.float64)

    pin_cell = np.ascontiguousarray(pin_features[:, 0].detach().numpy(), dtype=np.int32)
    pin_ox = np.ascontiguousarray(pin_features[:, 1].detach().numpy(), dtype=np.float64)
    pin_oy = np.ascontiguousarray(pin_features[:, 2].detach().numpy(), dtype=np.float64)

    e = edge_list.detach().numpy()
    ea = np.ascontiguousarray(e[:, 0], dtype=np.int32)
    eb = np.ascontiguousarray(e[:, 1], dtype=np.int32)

    bad = lib.partcl_place(
        n, w, h, x, y,
        pin_cell.shape[0], pin_cell, pin_ox, pin_oy,
        ea.shape[0], ea, eb,
        ctypes.c_double(budget_s), ctypes.c_uint(seed), ctypes.c_int(1 if verbose else 0),
    )
    if bad != 0:
        raise RuntimeError(f"solver returned a placement with {bad} overlapping cells")
    return x, y


def to_features(cell_features, x, y):
    out = cell_features.clone()
    out[:, 2] = torch.from_numpy(x).to(out.dtype)
    out[:, 3] = torch.from_numpy(y).to(out.dtype)
    return out


# Build/load at import time, not on the first solve: `test.py` times
# train_placement(), and compiling the library inside that window would land a
# one-off ~1s build cost in the reported runtime of test 1.
try:
    _load()
except Exception:  # no compiler available; the PARTCL_SOLVER=torch path still works
    pass
