"""
Shared utilities for placement modules.
"""

import os
from enum import IntEnum


# Feature index enums for cleaner code access
class CellFeatureIdx(IntEnum):
    """Indices for cell feature tensor columns."""
    AREA = 0
    NUM_PINS = 1
    X = 2
    Y = 3
    WIDTH = 4
    HEIGHT = 5


class PinFeatureIdx(IntEnum):
    """Indices for pin feature tensor columns."""
    CELL_IDX = 0
    PIN_X = 1
    PIN_Y = 2
    X = 3
    Y = 4
    WIDTH = 5
    HEIGHT = 6


# Configuration constants
# Macro parameters
MIN_MACRO_AREA = 100.0
MAX_MACRO_AREA = 10000.0

# Standard cell parameters (areas can be 1, 2, or 3)
STANDARD_CELL_AREAS = [1.0, 2.0, 3.0]
STANDARD_CELL_HEIGHT = 1.0

# Pin count parameters
MIN_STANDARD_CELL_PINS = 3
MAX_STANDARD_CELL_PINS = 6

# Output directory (will be set by placement.py)
OUTPUT_DIR = None

# Debug flags (enable via environment variables)
DEBUG_CUDA_OVERLAP = bool(int(os.environ.get("CUDA_OVERLAP_DEBUG", "0")))
FORCE_CPU_OVERLAP = bool(int(os.environ.get("FORCE_CPU_OVERLAP", "0")))

