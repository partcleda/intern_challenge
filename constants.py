# Configuration constants
# Macro parameters
MIN_MACRO_AREA = 100.0
MAX_MACRO_AREA = 10000.0

# Standard cell parameters (areas can be 1, 2, or 3)
STANDARD_CELL_AREAS = [1.0, 2.0, 3.0]
STANDARD_CELL_HEIGHT = 1.0

# Pin parameters
MIN_STANDARD_CELL_PINS = 3
MAX_STANDARD_CELL_PINS = 6
PIN_SIZE = 0.1  # All pins are 0.1 x 0.1


# Placement parameters
CELL_SPACING = 1e-3 # Small spacing to prevent zero-area cells and ensure legal placement
MAX_ROW_WIDTH = 120.0 # Maximum width of a placement row (for initial spread)
