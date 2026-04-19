"""
VLSI Cell Placement Optimization Challenge
==========================================

CHALLENGE OVERVIEW:
You are tasked with implementing a critical component of a chip placement optimizer.
Given a set of cells (circuit components) with fixed sizes and connectivity requirements,
you need to find positions for these cells that:
1. Minimize total wirelength (wiring cost between connected pins)
2. Eliminate all overlaps between cells

YOUR TASK:
Implement the `overlap_repulsion_loss()` function to prevent cells from overlapping.
The function must:
- Be differentiable (uses PyTorch operations for gradient descent)
- Detect when cells overlap in 2D space
- Apply increasing penalties for larger overlaps
- Work efficiently with vectorized operations

SUCCESS CRITERIA:
After running the optimizer with your implementation:
- overlap_count should be 0 (no overlapping cell pairs)
- total_overlap_area should be 0.0 (no overlap)
- wirelength should be minimized
- Visualization should show clean, non-overlapping placement

GETTING STARTED:
1. Read through the existing code to understand the data structures
2. Look at wirelength_attraction_loss() as a reference implementation
3. Implement overlap_repulsion_loss() following the TODO instructions
4. Run main() and check the overlap metrics in the output
5. Tune hyperparameters (lambda_overlap, lambda_wirelength) if needed
6. Generate visualization to verify your solution

BONUS CHALLENGES:
- Improve convergence speed by tuning learning rate or adding momentum
- Implement better initial placement strategy
- Add visualization of optimization progress over time
"""

import os
from enum import IntEnum

import torch
import torch.optim as optim


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
    PIN_X = 1  # Relative to cell corner
    PIN_Y = 2  # Relative to cell corner
    X = 3  # Absolute position
    Y = 4  # Absolute position
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

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======= SETUP =======

def generate_placement_input(num_macros, num_std_cells):
    """Generate synthetic placement input data.

    Args:
        num_macros: Number of macros to generate
        num_std_cells: Number of standard cells to generate

    Returns:
        Tuple of (cell_features, pin_features, edge_list):
            - cell_features: torch.Tensor of shape [N, 6] with columns [area, num_pins, x, y, width, height]
            - pin_features: torch.Tensor of shape [total_pins, 7] with columns
              [cell_instance_index, pin_x, pin_y, x, y, pin_width, pin_height]
            - edge_list: torch.Tensor of shape [E, 2] with [src_pin_idx, tgt_pin_idx]
    """
    total_cells = num_macros + num_std_cells

    # Step 1: Generate macro areas (uniformly distributed between min and max)
    macro_areas = (
        torch.rand(num_macros) * (MAX_MACRO_AREA - MIN_MACRO_AREA) + MIN_MACRO_AREA
    )

    # Step 2: Generate standard cell areas (randomly pick from 1, 2, or 3)
    std_cell_areas = torch.tensor(STANDARD_CELL_AREAS)[
        torch.randint(0, len(STANDARD_CELL_AREAS), (num_std_cells,))
    ]

    # Combine all areas
    areas = torch.cat([macro_areas, std_cell_areas])

    # Step 3: Calculate cell dimensions
    # Macros are square
    macro_widths = torch.sqrt(macro_areas)
    macro_heights = torch.sqrt(macro_areas)

    # Standard cells have fixed height = 1, width = area
    std_cell_widths = std_cell_areas / STANDARD_CELL_HEIGHT
    std_cell_heights = torch.full((num_std_cells,), STANDARD_CELL_HEIGHT)

    # Combine dimensions
    cell_widths = torch.cat([macro_widths, std_cell_widths])
    cell_heights = torch.cat([macro_heights, std_cell_heights])

    # Step 4: Calculate number of pins per cell
    num_pins_per_cell = torch.zeros(total_cells, dtype=torch.int)

    # Macros: between sqrt(area) and 2*sqrt(area) pins
    for i in range(num_macros):
        sqrt_area = int(torch.sqrt(macro_areas[i]).item())
        num_pins_per_cell[i] = torch.randint(sqrt_area, 2 * sqrt_area + 1, (1,)).item()

    # Standard cells: between 3 and 6 pins
    num_pins_per_cell[num_macros:] = torch.randint(
        MIN_STANDARD_CELL_PINS, MAX_STANDARD_CELL_PINS + 1, (num_std_cells,)
    )

    # Step 5: Create cell features tensor [area, num_pins, x, y, width, height]
    cell_features = torch.zeros(total_cells, 6)
    cell_features[:, CellFeatureIdx.AREA] = areas
    cell_features[:, CellFeatureIdx.NUM_PINS] = num_pins_per_cell.float()
    cell_features[:, CellFeatureIdx.X] = 0.0  # x position (initialized to 0)
    cell_features[:, CellFeatureIdx.Y] = 0.0  # y position (initialized to 0)
    cell_features[:, CellFeatureIdx.WIDTH] = cell_widths
    cell_features[:, CellFeatureIdx.HEIGHT] = cell_heights

    # Step 6: Generate pins for each cell
    total_pins = num_pins_per_cell.sum().item()
    pin_features = torch.zeros(total_pins, 7)

    # Fixed pin size for all pins (square pins)
    PIN_SIZE = 0.1  # All pins are 0.1 x 0.1

    pin_idx = 0
    for cell_idx in range(total_cells):
        n_pins = num_pins_per_cell[cell_idx].item()
        cell_width = cell_widths[cell_idx].item()
        cell_height = cell_heights[cell_idx].item()

        # Generate random pin positions within the cell
        # Offset from edges to ensure pins are fully inside
        margin = PIN_SIZE / 2
        if cell_width > 2 * margin and cell_height > 2 * margin:
            pin_x = torch.rand(n_pins) * (cell_width - 2 * margin) + margin
            pin_y = torch.rand(n_pins) * (cell_height - 2 * margin) + margin
        else:
            # For very small cells, just center the pins
            pin_x = torch.full((n_pins,), cell_width / 2)
            pin_y = torch.full((n_pins,), cell_height / 2)

        # Fill pin features
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.CELL_IDX] = cell_idx
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.PIN_X] = (
            pin_x  # relative to cell
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.PIN_Y] = (
            pin_y  # relative to cell
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.X] = (
            pin_x  # absolute (same as relative initially)
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.Y] = (
            pin_y  # absolute (same as relative initially)
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.WIDTH] = PIN_SIZE
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.HEIGHT] = PIN_SIZE

        pin_idx += n_pins

    # Step 7: Generate edges with simple random connectivity
    # Each pin connects to 1-3 random pins (preferring different cells)
    edge_list = []
    avg_edges_per_pin = 2.0

    pin_to_cell = torch.zeros(total_pins, dtype=torch.long)
    pin_idx = 0
    for cell_idx, n_pins in enumerate(num_pins_per_cell):
        pin_to_cell[pin_idx : pin_idx + n_pins] = cell_idx
        pin_idx += n_pins

    # Create adjacency set to avoid duplicate edges
    adjacency = [set() for _ in range(total_pins)]

    for pin_idx in range(total_pins):
        pin_cell = pin_to_cell[pin_idx].item()
        num_connections = torch.randint(1, 4, (1,)).item()  # 1-3 connections per pin

        # Try to connect to pins from different cells
        for _ in range(num_connections):
            # Random candidate
            other_pin = torch.randint(0, total_pins, (1,)).item()

            # Skip self-connections and existing connections
            if other_pin == pin_idx or other_pin in adjacency[pin_idx]:
                continue

            # Add edge (always store smaller index first for consistency)
            if pin_idx < other_pin:
                edge_list.append([pin_idx, other_pin])
            else:
                edge_list.append([other_pin, pin_idx])

            # Update adjacency
            adjacency[pin_idx].add(other_pin)
            adjacency[other_pin].add(pin_idx)

    # Convert to tensor and remove duplicates
    if edge_list:
        edge_list = torch.tensor(edge_list, dtype=torch.long)
        edge_list = torch.unique(edge_list, dim=0)
    else:
        edge_list = torch.zeros((0, 2), dtype=torch.long)

    print(f"\nGenerated placement data:")
    print(f"  Total cells: {total_cells}")
    print(f"  Total pins: {total_pins}")
    print(f"  Total edges: {len(edge_list)}")
    print(f"  Average edges per pin: {2 * len(edge_list) / total_pins:.2f}")

    return cell_features, pin_features, edge_list

# ======= OPTIMIZATION CODE (edit this part) =======

# NOTE:
# This GPU file intentionally mirrors `placement.py` as closely as possible.
# Any divergence is marked with `GPU-ONLY` comments for easier review diffing.

# Extra clearance used in differentiable overlap loss.
# Cells are penalized slightly before true geometric contact to create stronger separation gradients.
_OVERLAP_MARGIN = 0.02

# Tiny safety gap used in deterministic post processing legalization.
# Prevents near-touch numerical re-overlaps after floating point updates.
_LEGALIZE_MARGIN = 1e-3

# Minimum eigenvalue treated as nontrivial in spectral initialization.
# Filters numerical noise or near-zero modes when selecting layout directions.
_SPECTRAL_EIGEN_EPS = 1e-5

# Cell-count cutoff for exact pairwise overlap loss.
# Above this size, switch to sampled overlap loss to avoid O(n^2) memory/runtime.
_EXACT_OVERLAP_THRESHOLD = 700

def wirelength_attraction_loss(cell_features, pin_features, edge_list):
    """Calculate loss based on total wirelength to minimize routing.

    This is a REFERENCE IMPLEMENTATION showing how to write a differentiable loss function.

    The loss computes the Manhattan distance between connected pins and minimizes
    the total wirelength across all edges.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information
        edge_list: [E, 2] tensor with edges

    Returns:
        Scalar loss value
    """
    if edge_list.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Update absolute pin positions based on cell positions
    cell_positions = cell_features[:, 2:4]  # [N, 2]
    cell_indices = pin_features[:, 0].long()

    # Calculate absolute pin positions
    pin_absolute_x = cell_positions[cell_indices, 0] + pin_features[:, 1]
    pin_absolute_y = cell_positions[cell_indices, 1] + pin_features[:, 2]

    # Get source and target pin positions for each edge
    src_pins = edge_list[:, 0].long()
    tgt_pins = edge_list[:, 1].long()

    src_x = pin_absolute_x[src_pins]
    src_y = pin_absolute_y[src_pins]
    tgt_x = pin_absolute_x[tgt_pins]
    tgt_y = pin_absolute_y[tgt_pins]

    eps = 1e-3

    # Smooth differentiable distance in each axis.
    dx = torch.abs(src_x - tgt_x)
    dy = torch.abs(src_y - tgt_y)
    
    smooth_dx = torch.sqrt(dx * dx + eps)
    smooth_dy = torch.sqrt(dy * dy + eps)

    # Average-axis routing distance keeps objective scale stable and smooth.
    total_wirelength = torch.sum(0.5 * (smooth_dx + smooth_dy))

    return total_wirelength / edge_list.shape[0]  # Normalize by number of edges

def overlap_repulsion_loss(cell_features, pin_features, edge_list, margin=_OVERLAP_MARGIN):
    """Calculate loss to prevent cell overlaps.

    TODO: IMPLEMENT THIS FUNCTION

    This is the main challenge. You need to implement a differentiable loss function
    that penalizes overlapping cells. The loss should:

    1. Be zero when no cells overlap
    2. Increase as overlap area increases
    3. Use only differentiable PyTorch operations (no if statements on tensors)
    4. Work efficiently with vectorized operations

    HINTS:
    - Two axis-aligned rectangles overlap if they overlap in BOTH x and y dimensions
    - For rectangles centered at (x1, y1) and (x2, y2) with widths (w1, w2) and heights (h1, h2):
      * x-overlap occurs when |x1 - x2| < (w1 + w2) / 2
      * y-overlap occurs when |y1 - y2| < (h1 + h2) / 2
    - Use torch.relu() to compute positive overlaps: overlap_x = relu((w1+w2)/2 - |x1-x2|)
    - Overlap area = overlap_x * overlap_y
    - Consider all pairs of cells: use broadcasting with unsqueeze
    - Use torch.triu() to avoid counting each pair twice (only consider i < j)
    - Normalize the loss appropriately (by number of pairs or total area)

    RECOMMENDED APPROACH:
    1. Extract positions, widths, heights from cell_features
    2. Compute all pairwise distances using broadcasting:
       positions_i = positions.unsqueeze(1)  # [N, 1, 2]
       positions_j = positions.unsqueeze(0)  # [1, N, 2]
       distances = positions_i - positions_j  # [N, N, 2]
    3. Calculate minimum separation distances for each pair
    4. Use relu to get positive overlap amounts
    5. Multiply overlaps in x and y to get overlap areas
    6. Mask to only consider upper triangle (i < j)
    7. Sum and normalize

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information (not used here)
        edge_list: [E, 2] tensor with edges (not used here)

    Returns:
        Scalar loss value (should be 0 when no overlaps exist)
    """
    
    """Differentiable overlap penalty for all cell pairs."""
    del pin_features, edge_list  # Unused, kept for API compatibility

    # Total number of cells in the current placement.
    N = cell_features.shape[0]

    # No pair exists, so overlap loss is zero.
    if N <= 1:
        return torch.tensor(0.0, requires_grad=True)

    # Use sampled pairs for scalability on large designs.
    if N > _EXACT_OVERLAP_THRESHOLD:
        return _sampled_overlap_repulsion_loss(cell_features, margin=margin, max_pairs=220_000)

    # Cell center coordinates (x, y)
    positions = cell_features[:, 2:4]

    # Cell widths
    widths = cell_features[:, 4]

    # Cell heights.
    heights = cell_features[:, 5]

    # Pairwise center distance along x and y
    dx = (positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0)).abs()
    dy = (positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0)).abs()

    # Required x and y separation for non overlap
    min_sep_x = (widths.unsqueeze(1) + widths.unsqueeze(0)) / 2
    min_sep_y = (heights.unsqueeze(1) + heights.unsqueeze(0)) / 2

    # Positive only when cells overlap (or violate margin) on x and y axes.
    overlap_x = torch.relu(min_sep_x + margin - dx)
    overlap_y = torch.relu(min_sep_y + margin - dy)

    # Overlap area proxy per pair (nonzero only if both axes overlap).
    overlap_area = overlap_x * overlap_y

    # Keep unique pairs i < j only.
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=cell_features.device), diagonal=1)
    
    # Flatten to unique pair overlaps for loss aggregation.
    overlap_area = overlap_area[mask]

    # Linear + quadratic terms: fast cleanup of small overlaps and strong push on large ones.
    loss = (overlap_area + overlap_area.square()).sum()
    num_pairs = N * (N - 1) / 2
    return loss / num_pairs

def _sampled_overlap_repulsion_loss(cell_features, margin=_OVERLAP_MARGIN, max_pairs=220_000):
    """Helper function to estimate overlap loss with random pair sampling for large designs.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        margin: Extra clearance added to minimum spacing during penalty computation
        max_pairs: Number of random candidate pairs to sample

    Returns:
        Scalar sampled overlap loss; zero when no sampled overlaps are found
    """
    N = cell_features.shape[0]
    if N <= 1: return torch.tensor(0.0, requires_grad=True)

    # Keep random sampling tensors on the same device as placement tensors.
    device = cell_features.device

    # Per cell geometry and center coordinates.
    widths = cell_features[:, 4]
    heights = cell_features[:, 5]
    positions = cell_features[:, 2:4]

    # Sample candidate pair endpoints uniformly.
    i = torch.randint(0, N, (max_pairs,), device=device)
    j = torch.randint(0, N, (max_pairs,), device=device)

    # Remove self pairs since a cell cannot overlap with itself.
    valid = i != j
    i = i[valid]
    j = j[valid]

    # Degenerate case: all sampled indices matched, so no valid pair remains.
    if i.numel() == 0: return torch.tensor(0.0, requires_grad=True, device=device)

    # Enforce canonical ordering so pair (a,b) and (b,a) are treated consistently.
    swap = i > j
    i_swapped = torch.where(swap, j, i)
    j_swapped = torch.where(swap, i, j)
    i, j = i_swapped, j_swapped

    # Sampled pairwise center distance along x and y.
    dx = (positions[i, 0] - positions[j, 0]).abs()
    dy = (positions[i, 1] - positions[j, 1]).abs()

    # Minimum x and y separation required for non overlap.
    min_sep_x = (widths[i] + widths[j]) / 2
    min_sep_y = (heights[i] + heights[j]) / 2

    # Positive overlap (or margin violation) along x and y.
    overlap_x = torch.relu(min_sep_x + margin - dx)
    overlap_y = torch.relu(min_sep_y + margin - dy)

    # Overlap proxy area for sampled pairs.
    overlap_area = overlap_x * overlap_y

    # Mean linear + quadratic penalty for stable or strong gradients.
    return (overlap_area + overlap_area.square()).mean()

def _has_overlaps_fast(cell_features, margin=0.0):
    """Helper function to quickly check whether any pair of cells still overlaps.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        margin: Extra spacing treated as overlap for conservative checking

    Returns:
        True if at least one overlap is detected, else False
    """
    N = cell_features.shape[0]
    if N <= 1: return False

    # Extract geometry once to avoid repeated indexing inside checks.
    positions = cell_features[:, 2:4]
    widths = cell_features[:, 4]
    heights = cell_features[:, 5]

    if N <= 3500:

        # Exact O(n^2) check is still affordable for this size range.
        dx = (positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0)).abs()
        dy = (positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0)).abs()
        min_sep_x = (widths.unsqueeze(1) + widths.unsqueeze(0)) / 2
        min_sep_y = (heights.unsqueeze(1) + heights.unsqueeze(0)) / 2
        overlap = (min_sep_x + margin - dx > 0) & (min_sep_y + margin - dy > 0)
        return bool(torch.triu(overlap, diagonal=1).any().item())

    # Very large fallback: probabilistic sampled check to keep runtime bounded.
    device = cell_features.device
    max_pairs = 300_000
    i = torch.randint(0, N, (max_pairs,), device=device)
    j = torch.randint(0, N, (max_pairs,), device=device)
    valid = i != j
    i = i[valid]
    j = j[valid]
    if i.numel() == 0: return False
    dx = (positions[i, 0] - positions[j, 0]).abs()
    dy = (positions[i, 1] - positions[j, 1]).abs()
    min_sep_x = (widths[i] + widths[j]) / 2
    min_sep_y = (heights[i] + heights[j]) / 2
    overlap = (min_sep_x + margin - dx > 0) & (min_sep_y + margin - dy > 0)
    return bool(overlap.any().item())

def _size_adaptive_hyperparams(num_cells):
    """Helper function to return size dependent optimization hyperparameters.

    Args:
        num_cells: Number of cells in the current placement instance

    Returns:
        Dictionary of epoch counts, learning rates, overlap weight and clip/refine settings
    """
    # Small instances can afford longer optimization for better quality.
    if num_cells <= 40:
        return {
            "epochs_pre": 300,
            "epochs_a": 1800,
            "epochs_b": 1400,
            "lambda_overlap": 6000.0,
            "lr_pre": 0.05,
            "lr_a": 0.10,
            "lr_b": 0.06,
            "grad_clip": 5.0,
            "refine_steps": 180,
        }
    
    # Medium-small instances keep strong optimization with slightly lower LR.
    if num_cells <= 90:
        return {
            "epochs_pre": 350,
            "epochs_a": 2100,
            "epochs_b": 1600,
            "lambda_overlap": 7500.0,
            "lr_pre": 0.04,
            "lr_a": 0.085,
            "lr_b": 0.055,
            "grad_clip": 6.0,
            "refine_steps": 180,
        }
    
    # Mid-sized instances balance quality against runtime.
    if num_cells <= 180:
        return {
            "epochs_pre": 450,
            "epochs_a": 2300,
            "epochs_b": 1800,
            "lambda_overlap": 10000.0,
            "lr_pre": 0.035,
            "lr_a": 0.07,
            "lr_b": 0.045,
            "grad_clip": 8.0,
            "refine_steps": 200,
        }
    
    # Larger dense instances need lower LR and stronger overlap weight.
    if num_cells <= 400:
        return {
            "epochs_pre": 600,
            "epochs_a": 2500,
            "epochs_b": 1900,
            "lambda_overlap": 14000.0,
            "lr_pre": 0.03,
            "lr_a": 0.055,
            "lr_b": 0.038,
            "grad_clip": 10.0,
            "refine_steps": 200,
        }
    
    # Large instances shorten schedules to keep total runtime reasonable.
    if num_cells <= 900:
        return {
            "epochs_pre": 250,
            "epochs_a": 900,
            "epochs_b": 600,
            "lambda_overlap": 18000.0,
            "lr_pre": 0.02,
            "lr_a": 0.04,
            "lr_b": 0.03,
            "grad_clip": 10.0,
            "refine_steps": 80,
        }
    
    # Very large instances prioritize robustness and scalability.
    if num_cells <= 1500:
        return {
            "epochs_pre": 0,
            "epochs_a": 500,
            "epochs_b": 260,
            "lambda_overlap": 22000.0,
            "lr_pre": 0.0,
            "lr_a": 0.032,
            "lr_b": 0.025,
            "grad_clip": 12.0,
            "refine_steps": 40,
        }
    
    # Extra-large instances use compact schedules and minimal refinement.
    return {
        "epochs_pre": 0,
        "epochs_a": 140,
        "epochs_b": 80,
        "lambda_overlap": 25000.0,
        "lr_pre": 0.0,
        "lr_a": 0.028,
        "lr_b": 0.022,
        "grad_clip": 12.0,
        "refine_steps": 0,
    }

def _build_cell_adjacency_matrix(pin_features, edge_list, num_cells, device, dtype):
    """Helper function to build a symmetric weighted cell adjacency matrix from pin-level edges.

    Args:
        pin_features: [P, 7] tensor containing owning cell index per pin
        edge_list: [E, 2] tensor of connected pin index pairs
        num_cells: Total number of cells
        device: Target device for created adjacency tensor
        dtype: Target dtype for created adjacency tensor

    Returns:
        [num_cells, num_cells] adjacency tensor, or None if no inter-cell edges exist
    """

    if edge_list.shape[0] == 0: return None

    # Map each pin endpoint in every edge to its owning cell
    pin_to_cell = pin_features[:, PinFeatureIdx.CELL_IDX].long()
    src_cells = pin_to_cell[edge_list[:, 0].long()]
    tgt_cells = pin_to_cell[edge_list[:, 1].long()]

    # Ignore edges that stay within the same cell.
    valid = src_cells != tgt_cells
    if not valid.any(): return None

    src_cells = src_cells[valid]
    tgt_cells = tgt_cells[valid]
    adjacency = torch.zeros((num_cells, num_cells), device=device, dtype=dtype)
    edge_weight = torch.ones(src_cells.shape[0], device=device, dtype=dtype)
    adjacency.index_put_((src_cells, tgt_cells), edge_weight, accumulate=True)
    adjacency.index_put_((tgt_cells, src_cells), edge_weight, accumulate=True)
    return adjacency


def _spectral_initial_placement(cell_features, pin_features, edge_list):
    """Helper function to seed cell coordinates using low frequency Laplacian eigenvectors.

    Args:
        cell_features: [N, 6] tensor with mutable cell positions
        pin_features: [P, 7] tensor with pin-to-cell ownership
        edge_list: [E, 2] tensor with pin-level connectivity

    Returns:
        True if spectral seeding was applied, else False
    """
    num_cells = cell_features.shape[0]
    if num_cells <= 3 or edge_list.shape[0] == 0 or num_cells > _EXACT_OVERLAP_THRESHOLD:
        return False

    device = cell_features.device
    dtype = cell_features.dtype
    adjacency = _build_cell_adjacency_matrix(pin_features, edge_list, num_cells, device, dtype)
    if adjacency is None: return False

    # Build unnormalized graph Laplacian L = D - A.
    degree = adjacency.sum(dim=1)
    laplacian = torch.diag(degree) - adjacency

    # Regularization improves numerical stability for disconnected graphs.
    laplacian = laplacian + torch.eye(num_cells, device=device, dtype=dtype) * 1e-6
    evals, evecs = torch.linalg.eigh(laplacian)
    nontrivial = torch.nonzero(evals > _SPECTRAL_EIGEN_EPS, as_tuple=False).flatten()
    if nontrivial.numel() == 0: return False

    # Use first two non-trivial eigenvectors as x/y layout coordinates.
    x_vec = evecs[:, nontrivial[0]]
    if nontrivial.numel() > 1: y_vec = evecs[:, nontrivial[1]]
    else:
        # Deterministic fallback direction when only one non-trivial mode exists.
        y_vec = torch.linspace(-1.0, 1.0, num_cells, device=device, dtype=dtype)

    total_area = cell_features[:, CellFeatureIdx.AREA].sum()
    max_dim = torch.max(cell_features[:, CellFeatureIdx.WIDTH].max(), cell_features[:, CellFeatureIdx.HEIGHT].max())
    target_span = torch.maximum(total_area.sqrt() * 0.8, max_dim * 1.5)

    def _scale(vec):
        # Normalize each coordinate vector to a common placement span.
        centered = vec - vec.mean()
        span = centered.max() - centered.min()
        if span.abs() < 1e-12: return centered
        return centered / span * target_span

    x_pos = _scale(x_vec)
    y_pos = _scale(y_vec)

    # Small deterministic jitter avoids ties without introducing run-to-run variance.
    jitter = torch.linspace(-0.5, 0.5, num_cells, device=device, dtype=dtype) * (target_span * 0.005)
    cell_features[:, CellFeatureIdx.X] = x_pos + jitter
    cell_features[:, CellFeatureIdx.Y] = y_pos - jitter
    return True

def _wirelength_prefit(
    cell_features,
    pin_features,
    edge_list,
    steps,
    lr,
    grad_clip,
    loss_history,
):
    """Helper function to run a short wirelength only optimization warm start.

    Args:
        cell_features: [N, 6] tensor; updated in place with fitted positions
        pin_features: [P, 7] tensor with pin metadata
        edge_list: [E, 2] tensor with pin connectivity
        steps: Number of warm-start optimization steps
        lr: Adam learning rate for warm start
        grad_clip: Maximum gradient norm for position updates
        loss_history: Dict collecting optimization loss traces
    """

    if steps <= 0 or edge_list.shape[0] == 0: return

    # Optimize only cell centers while keeping geometry fixed.
    positions = cell_features[:, 2:4].clone().detach().requires_grad_(True)
    optimizer = optim.Adam([positions], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.2)

    for _ in range(steps):
        optimizer.zero_grad()
        cell_features_current = cell_features.clone()
        cell_features_current[:, 2:4] = positions

        wl_loss = wirelength_attraction_loss(cell_features_current, pin_features, edge_list)
        wl_loss.backward()

        # Clip to avoid unstable jumps on dense random graphs.
        torch.nn.utils.clip_grad_norm_([positions], max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        loss_history["total_loss"].append(wl_loss.item())
        loss_history["wirelength_loss"].append(wl_loss.item())
        loss_history["overlap_loss"].append(0.0)

    cell_features[:, 2:4] = positions.detach()

def _force_legal_shelf_pack(cell_features, spacing=0.02):
    """Helper function to fallback legalizer that packs cells into non-overlapping shelves.

    Args:
        cell_features: [N, 6] tensor; updated in place with legal packed positions
        spacing: Gap inserted between neighboring cells and rows
    """
    with torch.no_grad():
        # Read geometry and current positions for ordering heuristics.
        widths = cell_features[:, 4]
        heights = cell_features[:, 5]
        positions = cell_features[:, 2:4]
        num_cells = cell_features.shape[0]

        total_area = cell_features[:, 0].sum()
        max_width = widths.max()
        target_row_width = torch.maximum(total_area.sqrt() * 1.4, max_width * 4.0).item()

        # Preserve approximate locality from current placement by x ordering.
        order = torch.argsort(positions[:, 0])
        x_cursor = 0.0
        y_cursor = 0.0
        row_height = 0.0
        packed = torch.zeros_like(positions)

        for idx in order.tolist():
            w = float(widths[idx].item())
            h = float(heights[idx].item())

            # Start a new shelf when current row capacity is exceeded.
            if x_cursor > 0.0 and (x_cursor + w) > target_row_width:
                y_cursor += row_height + spacing
                x_cursor = 0.0
                row_height = 0.0

            # Place each cell at shelf center coordinates.
            packed[idx, 0] = x_cursor + (w / 2.0)
            packed[idx, 1] = y_cursor + (h / 2.0)
            x_cursor += w + spacing
            if h > row_height: row_height = h

        # Recenter around origin for numerical stability.
        packed[:, 0] -= packed[:, 0].mean()
        packed[:, 1] -= packed[:, 1].mean()
        cell_features[:, 2:4] = packed


def _legalize_overlaps(cell_features, max_iters=120, margin=_LEGALIZE_MARGIN):
    """Helper function to resolve remaining overlaps with iterative pairwise displacement.

    Args:
        cell_features: [N, 6] tensor; positions are updated in place
        max_iters: Maximum legalization iterations
        margin: Extra clearance enforced between neighboring cells
    """
    with torch.no_grad():
        # Extract geometry and mutable centers.
        positions = cell_features[:, 2:4]
        widths = cell_features[:, 4]
        heights = cell_features[:, 5]
        areas = cell_features[:, 0]
        num_cells = cell_features.shape[0]

        for _ in range(max_iters):

            # Compute pairwise center distances and required non-overlap spacing.
            dx = (positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0)).abs()
            dy = (positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0)).abs()

            min_sep_x = (widths.unsqueeze(1) + widths.unsqueeze(0)) / 2
            min_sep_y = (heights.unsqueeze(1) + heights.unsqueeze(0)) / 2

            overlap_x = torch.relu(min_sep_x + margin - dx)
            overlap_y = torch.relu(min_sep_y + margin - dy)
            mask = torch.triu((overlap_x > 0) & (overlap_y > 0), diagonal=1)

            # Stop early as soon as no overlapping pair remains.
            if not mask.any(): break

            i_idx, j_idx = torch.nonzero(mask, as_tuple=True)
            pair_overlap_x = overlap_x[i_idx, j_idx]
            pair_overlap_y = overlap_y[i_idx, j_idx]
            move_in_x = pair_overlap_x <= pair_overlap_y
            required_sep = torch.where(move_in_x, pair_overlap_x + margin, pair_overlap_y + margin)

            dir_x = torch.sign(positions[j_idx, 0] - positions[i_idx, 0])
            dir_y = torch.sign(positions[j_idx, 1] - positions[i_idx, 1])

            # Deterministic fallback when two centers align exactly.
            fallback = torch.where(
                ((i_idx + j_idx) % 2 == 0),
                torch.ones_like(dir_x),
                -torch.ones_like(dir_x),
            )
            dir_x = torch.where(dir_x == 0, fallback, dir_x)
            dir_y = torch.where(dir_y == 0, fallback, dir_y)

            direction = torch.stack(
                [
                    torch.where(move_in_x, dir_x, torch.zeros_like(dir_x)),
                    torch.where(move_in_x, torch.zeros_like(dir_y), dir_y),
                ],
                dim=1,
            )

            area_i = areas[i_idx]
            area_j = areas[j_idx]
            area_total = area_i + area_j + 1e-8

            # Move smaller cells more than larger cells to preserve macro placement quality.
            move_i = area_j / area_total
            move_j = area_i / area_total

            disp_i = -direction * (required_sep * move_i).unsqueeze(1)
            disp_j = direction * (required_sep * move_j).unsqueeze(1)

            delta = torch.zeros_like(positions)
            counts = torch.zeros(num_cells, 1, device=positions.device, dtype=positions.dtype)
            delta.index_add_(0, i_idx, disp_i)
            delta.index_add_(0, j_idx, disp_j)

            # Average accumulated displacement for cells in multiple overlap pairs.
            ones = torch.ones(i_idx.shape[0], 1, device=positions.device, dtype=positions.dtype)
            counts.index_add_(0, i_idx, ones)
            counts.index_add_(0, j_idx, ones)

            positions += 0.85 * delta / counts.clamp_min(1.0)


def _wirelength_refinement(
    cell_features,
    pin_features,
    edge_list,
    steps,
    lr,
    lambda_overlap,
    grad_clip,
    loss_history,
):
    """Helper function to run short WL-driven refinement while preserving legality pressure.

    Args:
        cell_features: [N, 6] tensor; updated in place with refined positions
        pin_features: [P, 7] tensor with pin metadata
        edge_list: [E, 2] tensor with pin connectivity
        steps: Number of refinement optimization steps
        lr: Adam learning rate during refinement
        lambda_overlap: Overlap penalty multiplier during refinement
        grad_clip: Maximum gradient norm for position updates
        loss_history: Dict collecting optimization loss traces
    """
    if steps <= 0 or edge_list.shape[0] == 0: return

    # Optimize only position coordinates in this refinement stage.
    positions = cell_features[:, 2:4].clone().detach().requires_grad_(True)
    optimizer = optim.Adam([positions], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.2)

    for _ in range(steps):
        optimizer.zero_grad()
        cell_features_current = cell_features.clone()
        cell_features_current[:, 2:4] = positions

        wl_loss = wirelength_attraction_loss(cell_features_current, pin_features, edge_list)
        overlap_loss = overlap_repulsion_loss(cell_features_current, pin_features, edge_list)
        total_loss = 10.0 * wl_loss + lambda_overlap * overlap_loss
        total_loss.backward()

        # Clip gradients for stable updates near legal boundaries.
        torch.nn.utils.clip_grad_norm_([positions], max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        loss_history["total_loss"].append(total_loss.item())
        loss_history["wirelength_loss"].append(wl_loss.item())
        loss_history["overlap_loss"].append(overlap_loss.item())

    cell_features[:, 2:4] = positions.detach()

def train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=1000,
    lr=0.01,
    lambda_wirelength=1.0,
    lambda_overlap=10.0,
    verbose=True,
    log_interval=100,
):
    """Train the placement optimization using gradient descent.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity
        num_epochs: Number of optimization iterations
        lr: Learning rate for Adam optimizer
        lambda_wirelength: Weight for wirelength loss
        lambda_overlap: Weight for overlap loss
        verbose: Whether to print progress
        log_interval: How often to print progress

    Returns:
        Dictionary with:
            - final_cell_features: Optimized cell positions
            - initial_cell_features: Original cell positions (for comparison)
            - loss_history: Loss values over time
    """
    # Clone features and create learnable positions
    cell_features = cell_features.clone()
    initial_cell_features = cell_features.clone()
    num_cells = cell_features.shape[0]

    # Automatically tune runtime by instance size.
    hp = _size_adaptive_hyperparams(num_cells)
    epochs_pre = hp["epochs_pre"]
    epochs_a = hp["epochs_a"]
    epochs_b = hp["epochs_b"]
    lr_pre = hp["lr_pre"]
    lr_a = hp["lr_a"]
    lr_b = hp["lr_b"]
    grad_clip = hp["grad_clip"]
    refine_steps = hp["refine_steps"]
    lambda_overlap = hp["lambda_overlap"]

    loss_history = {"total_loss": [], "wirelength_loss": [], "overlap_loss": []}

    # Spectral seed + short WL prefit to start from a low WL topology.
    _spectral_initial_placement(cell_features, pin_features, edge_list)
    _wirelength_prefit(
        cell_features,
        pin_features,
        edge_list,
        steps=epochs_pre,
        lr=lr_pre,
        grad_clip=grad_clip,
        loss_history=loss_history,
    )
    initial_cell_features = cell_features.clone()

    # Make only cell positions require gradients
    cell_positions = cell_features[:, 2:4].clone().detach()
    cell_positions.requires_grad_(True)

    # Phase A: keep WL active while ramping overlap pressure.
    # Adam handles noisy gradients from mixed WL and overlap objectives.
    optimizer_a = optim.Adam([cell_positions], lr=lr_a)

    # Cosine annealing smoothly decays LR to stabilize late Phase A updates.
    scheduler_a = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_a, T_max=epochs_a, eta_min=lr_a * 0.02
    )

    # Track consecutive near-zero-overlap epochs for early phase transition.
    zero_overlap_streak = 0

    # Default phase end assumes full schedule unless early-stop triggers.
    phase_a_end = epochs_a

    for epoch in range(epochs_a):
        # Reset gradients before each optimization step.
        optimizer_a.zero_grad()

        # Normalized progress scalar used for schedule interpolation.
        t = epoch / max(epochs_a - 1, 1)

        # Increase overlap weight over time to enforce legality progressively.
        current_lambda_overlap = 10.0 + (lambda_overlap - 10.0) * t

        # Keep WL weight fixed in Phase A to avoid overpowering overlap cleanup.
        current_lambda_wirelength = 1.0

        # Build a view of current placement state with live position tensor.
        cell_features_current = cell_features.clone()
        cell_features_current[:, 2:4] = cell_positions

        # Compute wirelength and overlap terms for joint optimization.
        wl_loss = wirelength_attraction_loss(cell_features_current, pin_features, edge_list)
        overlap_loss = overlap_repulsion_loss(cell_features_current, pin_features, edge_list)
        total_loss = current_lambda_wirelength * wl_loss + current_lambda_overlap * overlap_loss
        
        # Backpropagate into cell positions only.
        total_loss.backward()

        # Clip gradients to prevent unstable jumps from large overlap forces.
        torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=grad_clip)
        
        # Optimizer and scheduler step for this epoch.
        optimizer_a.step()
        scheduler_a.step()

        # Read scalar overlap for logging and convergence checks.
        overlap_value = overlap_loss.item()

        # Count streak length of effectively overlap free epochs.
        zero_overlap_streak = zero_overlap_streak + 1 if overlap_value < 1e-10 else 0

        # Append losses to history for later diagnostics/plots.
        loss_history["total_loss"].append(total_loss.item())
        loss_history["wirelength_loss"].append(wl_loss.item())
        loss_history["overlap_loss"].append(overlap_value)

        if verbose and (epoch % log_interval == 0):
            print(
                f"Phase A {epoch}/{epochs_a}: "
                f"WL={wl_loss.item():.6f}, OV={overlap_value:.8f}"
            )

        # Exit Phase A once overlap is stably cleared for enough epochs.
        if zero_overlap_streak >= 80 and epoch >= int(epochs_a * 0.4):
            phase_a_end = epoch + 1
            if verbose:
                print(f"Phase A converged at epoch {epoch}, moving to Phase B.")
            break

    # Phase B: improve wirelength while keeping overlap penalty alive.
    # Reallocate unused Phase A epochs into Phase B for better WL refinement.
    remaining = epochs_a - phase_a_end
    total_phase_b_epochs = epochs_b + remaining

    # New optimizer starts Phase B with a lower learning rate.
    optimizer_b = optim.Adam([cell_positions], lr=lr_b)

    # Multi-step decay sharpens convergence near the end of Phase B.
    scheduler_b = optim.lr_scheduler.MultiStepLR(
        optimizer_b,
        milestones=[int(total_phase_b_epochs * 0.7), int(total_phase_b_epochs * 0.9)],
        gamma=0.35,
    )

    for epoch in range(total_phase_b_epochs):
        # Reset gradients before this Phase B step.
        optimizer_b.zero_grad()

        # Normalized phase progress drives WL/overlap weight schedules.
        t = epoch / max(total_phase_b_epochs - 1, 1)

        # Gradually prioritize WL minimization in Phase B.
        current_lambda_wirelength = 3.0 + 12.0 * t

        # Keep overlap penalty active but taper it down over time.
        current_lambda_overlap = lambda_overlap * (0.42 - 0.22 * t)

        # Rebuild placement snapshot from static geometry + learnable positions.
        cell_features_current = cell_features.clone()
        cell_features_current[:, 2:4] = cell_positions

        # Compute composite loss under Phase B weights.
        wl_loss = wirelength_attraction_loss(cell_features_current, pin_features, edge_list)
        overlap_loss = overlap_repulsion_loss(cell_features_current, pin_features, edge_list)
        total_loss = current_lambda_wirelength * wl_loss + current_lambda_overlap * overlap_loss
        
        # Backpropagate
        total_loss.backward()

        # Clip gradients for stability in dense designs.
        torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=grad_clip)
        
        # Advance optimizer and scheduler for this epoch.
        optimizer_b.step()
        scheduler_b.step()

        # Record losses for analysis and leaderboard debugging.
        loss_history["total_loss"].append(total_loss.item())
        loss_history["wirelength_loss"].append(wl_loss.item())
        loss_history["overlap_loss"].append(overlap_loss.item())

        if verbose and (epoch % log_interval == 0 or epoch == total_phase_b_epochs - 1):
            print(
                f"Phase B {epoch}/{total_phase_b_epochs}: "
                f"WL={wl_loss.item():.6f}, OV={overlap_loss.item():.8f}"
            )

    # Materialize final optimized coordinates into an output tensor.
    final_cell_features = cell_features.clone()
    final_cell_features[:, 2:4] = cell_positions.detach()

    # Hard cleanup for any residual contacts, then WL polish while preserving legality.
    if num_cells <= 300:
        # Small cases get stronger legalization for strict zero-overlap closure.
        pre_legalize_iters = 200
        post_legalize_iters = 500
        legalize_margin = 0.02

    elif num_cells <= 1000:
        # Medium cases use moderate legalization effort.
        pre_legalize_iters = 120
        post_legalize_iters = 220
        legalize_margin = 0.015

    else:
        # Large cases use lighter legalization to contain runtime.
        pre_legalize_iters = 60
        post_legalize_iters = 80
        legalize_margin = 0.01

    # First deterministic legalization removes most remaining overlaps.
    _legalize_overlaps(
        final_cell_features,
        max_iters=pre_legalize_iters,
        margin=legalize_margin,
    )

    # Short WL focused polish runs with overlap penalty still active.
    _wirelength_refinement(
        final_cell_features,
        pin_features,
        edge_list,
        steps=refine_steps,
        lr=lr_b * 0.8,
        lambda_overlap=lambda_overlap * 0.35,
        grad_clip=grad_clip,
        loss_history=loss_history,
    )

    # Final legalization pass ensures robust geometric separation.
    _legalize_overlaps(
        final_cell_features,
        max_iters=post_legalize_iters,
        margin=legalize_margin,
    )

    # Escalate legalization only when needed, keeping WL impact very small.
    if _has_overlaps_fast(final_cell_features):

        # Multiple rounds avoid local oscillations in dense corner cases.
        rounds = 2 if num_cells > 1000 else 4
        schedule = (
            [(0.008, 180), (0.012, 260), (0.018, 360), (0.025, 520)]
            if num_cells > 1000
            else [(0.01, 260), (0.015, 360), (0.02, 520), (0.03, 700), (0.05, 900)]
        )

        for _ in range(rounds):
            for margin, iters in schedule:
                _legalize_overlaps(final_cell_features, max_iters=iters, margin=margin)
                if not _has_overlaps_fast(final_cell_features):
                    break
            if not _has_overlaps_fast(final_cell_features):
                break

    # GPU-ONLY: exact CPU overlap check for small or medium cases.
    # tiny residual overlaps that sampled GPU checks might miss.
    if num_cells <= 1200:
        if len(calculate_cells_with_overlaps(final_cell_features.detach().cpu())) > 0:
    
            # GPU-ONLY fallback: force legal placement if exact checker finds overlaps.
            _force_legal_shelf_pack(final_cell_features, spacing=0.02)

    # Guaranteed legality fallback for very large designs.
    if num_cells > 1000 and _has_overlaps_fast(final_cell_features):
        _force_legal_shelf_pack(final_cell_features, spacing=0.02)

    return {
        "final_cell_features": final_cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": loss_history,
    }

# ======= FINAL EVALUATION CODE (Don't edit this part) =======

def calculate_overlap_metrics(cell_features):
    """Calculate ground truth overlap statistics (non-differentiable).

    This function provides exact overlap measurements for evaluation and reporting.
    Unlike the loss function, this does NOT need to be differentiable.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]

    Returns:
        Dictionary with:
            - overlap_count: number of overlapping cell pairs (int)
            - total_overlap_area: sum of all overlap areas (float)
            - max_overlap_area: largest single overlap area (float)
            - overlap_percentage: percentage of total area that overlaps (float)
    """
    N = cell_features.shape[0]
    if N <= 1:
        return {
            "overlap_count": 0,
            "total_overlap_area": 0.0,
            "max_overlap_area": 0.0,
            "overlap_percentage": 0.0,
        }

    # Extract cell properties
    positions = cell_features[:, 2:4].detach().numpy()  # [N, 2]
    widths = cell_features[:, 4].detach().numpy()  # [N]
    heights = cell_features[:, 5].detach().numpy()  # [N]
    areas = cell_features[:, 0].detach().numpy()  # [N]

    overlap_count = 0
    total_overlap_area = 0.0
    max_overlap_area = 0.0
    overlap_areas = []

    # Check all pairs
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate center-to-center distances
            dx = abs(positions[i, 0] - positions[j, 0])
            dy = abs(positions[i, 1] - positions[j, 1])

            # Minimum separation for non-overlap
            min_sep_x = (widths[i] + widths[j]) / 2
            min_sep_y = (heights[i] + heights[j]) / 2

            # Calculate overlap amounts
            overlap_x = max(0, min_sep_x - dx)
            overlap_y = max(0, min_sep_y - dy)

            # Overlap occurs only if both x and y overlap
            if overlap_x > 0 and overlap_y > 0:
                overlap_area = overlap_x * overlap_y
                overlap_count += 1
                total_overlap_area += overlap_area
                max_overlap_area = max(max_overlap_area, overlap_area)
                overlap_areas.append(overlap_area)

    # Calculate percentage of total area
    total_area = sum(areas)
    overlap_percentage = (overlap_count / N * 100) if total_area > 0 else 0.0

    return {
        "overlap_count": overlap_count,
        "total_overlap_area": total_overlap_area,
        "max_overlap_area": max_overlap_area,
        "overlap_percentage": overlap_percentage,
    }


def calculate_cells_with_overlaps(cell_features):
    """Calculate number of cells involved in at least one overlap.

    This metric matches the test suite evaluation criteria.

    Args:
        cell_features: [N, 6] tensor with cell properties

    Returns:
        Set of cell indices that have overlaps with other cells
    """
    N = cell_features.shape[0]
    if N <= 1:
        return set()

    # Extract cell properties
    positions = cell_features[:, 2:4].detach().numpy()
    widths = cell_features[:, 4].detach().numpy()
    heights = cell_features[:, 5].detach().numpy()

    cells_with_overlaps = set()

    # Check all pairs
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate center-to-center distances
            dx = abs(positions[i, 0] - positions[j, 0])
            dy = abs(positions[i, 1] - positions[j, 1])

            # Minimum separation for non-overlap
            min_sep_x = (widths[i] + widths[j]) / 2
            min_sep_y = (heights[i] + heights[j]) / 2

            # Calculate overlap amounts
            overlap_x = max(0, min_sep_x - dx)
            overlap_y = max(0, min_sep_y - dy)

            # Overlap occurs only if both x and y overlap
            if overlap_x > 0 and overlap_y > 0:
                cells_with_overlaps.add(i)
                cells_with_overlaps.add(j)

    return cells_with_overlaps


def calculate_normalized_metrics(cell_features, pin_features, edge_list):
    """Calculate normalized overlap and wirelength metrics for test suite.

    These metrics match the evaluation criteria in the test suite.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity

    Returns:
        Dictionary with:
            - overlap_ratio: (num cells with overlaps / total cells)
            - normalized_wl: (wirelength / num nets) / sqrt(total area)
            - num_cells_with_overlaps: number of unique cells involved in overlaps
            - total_cells: total number of cells
            - num_nets: number of nets (edges)
    """
    N = cell_features.shape[0]

    # Calculate overlap metric: num cells with overlaps / total cells
    cells_with_overlaps = calculate_cells_with_overlaps(cell_features)
    num_cells_with_overlaps = len(cells_with_overlaps)
    overlap_ratio = num_cells_with_overlaps / N if N > 0 else 0.0

    # Calculate wirelength metric: (wirelength / num nets) / sqrt(total area)
    if edge_list.shape[0] == 0:
        normalized_wl = 0.0
        num_nets = 0
    else:
        # Calculate total wirelength using the loss function (unnormalized)
        wl_loss = wirelength_attraction_loss(cell_features, pin_features, edge_list)
        total_wirelength = wl_loss.item() * edge_list.shape[0]  # Undo normalization

        # Calculate total area
        total_area = cell_features[:, 0].sum().item()

        num_nets = edge_list.shape[0]

        # Normalize: (wirelength / net) / sqrt(area)
        # This gives a dimensionless quality metric independent of design size
        normalized_wl = (total_wirelength / num_nets) / (total_area ** 0.5) if total_area > 0 else 0.0

    return {
        "overlap_ratio": overlap_ratio,
        "normalized_wl": normalized_wl,
        "num_cells_with_overlaps": num_cells_with_overlaps,
        "total_cells": N,
        "num_nets": num_nets,
    }


def plot_placement(
    initial_cell_features,
    final_cell_features,
    pin_features,
    edge_list,
    filename="placement_result.png",
):
    """Create side-by-side visualization of initial vs final placement.

    Args:
        initial_cell_features: Initial cell positions and properties
        final_cell_features: Optimized cell positions and properties
        pin_features: Pin information
        edge_list: Edge connectivity
        filename: Output filename for the plot
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot both initial and final placements
        for ax, cell_features, title in [
            (ax1, initial_cell_features, "Initial Placement"),
            (ax2, final_cell_features, "Final Placement"),
        ]:
            N = cell_features.shape[0]
            positions = cell_features[:, 2:4].detach().numpy()
            widths = cell_features[:, 4].detach().numpy()
            heights = cell_features[:, 5].detach().numpy()

            # Draw cells
            for i in range(N):
                x = positions[i, 0] - widths[i] / 2
                y = positions[i, 1] - heights[i] / 2
                rect = Rectangle(
                    (x, y),
                    widths[i],
                    heights[i],
                    fill=True,
                    facecolor="lightblue",
                    edgecolor="darkblue",
                    linewidth=0.5,
                    alpha=0.7,
                )
                ax.add_patch(rect)

            # Calculate and display overlap metrics
            metrics = calculate_overlap_metrics(cell_features)

            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title(
                f"{title}\n"
                f"Overlaps: {metrics['overlap_count']}, "
                f"Total Overlap Area: {metrics['total_overlap_area']:.2f}",
                fontsize=12,
            )

            # Set axis limits with margin
            all_x = positions[:, 0]
            all_y = positions[:, 1]
            margin = 10
            ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
            ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    except ImportError as e:
        print(f"Could not create visualization: {e}")
        print("Install matplotlib to enable visualization: pip install matplotlib")

# ======= MAIN FUNCTION =======

def main():
    """Main function demonstrating the placement optimization challenge."""
    print("=" * 70)
    print("VLSI CELL PLACEMENT OPTIMIZATION CHALLENGE")
    print("=" * 70)
    print("\nObjective: Implement overlap_repulsion_loss() to eliminate cell overlaps")
    print("while minimizing wirelength.\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Generate placement problem
    num_macros = 3
    num_std_cells = 50

    print(f"Generating placement problem:")
    print(f"  - {num_macros} macros")
    print(f"  - {num_std_cells} standard cells")

    cell_features, pin_features, edge_list = generate_placement_input(
        num_macros, num_std_cells
    )

    # Initialize positions with random spread to reduce initial overlaps
    total_cells = cell_features.shape[0]
    spread_radius = 30.0
    angles = torch.rand(total_cells) * 2 * 3.14159
    radii = torch.rand(total_cells) * spread_radius

    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)

    # Calculate initial metrics
    print("\n" + "=" * 70)
    print("INITIAL STATE")
    print("=" * 70)
    initial_metrics = calculate_overlap_metrics(cell_features)
    print(f"Overlap count: {initial_metrics['overlap_count']}")
    print(f"Total overlap area: {initial_metrics['total_overlap_area']:.2f}")
    print(f"Max overlap area: {initial_metrics['max_overlap_area']:.2f}")
    print(f"Overlap percentage: {initial_metrics['overlap_percentage']:.2f}%")

    # Run optimization
    print("\n" + "=" * 70)
    print("RUNNING OPTIMIZATION")
    print("=" * 70)

    result = train_placement(
        cell_features,
        pin_features,
        edge_list,
        verbose=True,
        log_interval=200,
    )

    # Calculate final metrics (both detailed and normalized)
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    final_cell_features = result["final_cell_features"]

    # Detailed metrics
    final_metrics = calculate_overlap_metrics(final_cell_features)
    print(f"Overlap count (pairs): {final_metrics['overlap_count']}")
    print(f"Total overlap area: {final_metrics['total_overlap_area']:.2f}")
    print(f"Max overlap area: {final_metrics['max_overlap_area']:.2f}")

    # Normalized metrics (matching test suite)
    print("\n" + "-" * 70)
    print("TEST SUITE METRICS (for leaderboard)")
    print("-" * 70)
    normalized_metrics = calculate_normalized_metrics(
        final_cell_features, pin_features, edge_list
    )
    print(f"Overlap Ratio: {normalized_metrics['overlap_ratio']:.4f} "
          f"({normalized_metrics['num_cells_with_overlaps']}/{normalized_metrics['total_cells']} cells)")
    print(f"Normalized Wirelength: {normalized_metrics['normalized_wl']:.4f}")

    # Success check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA")
    print("=" * 70)
    if normalized_metrics["num_cells_with_overlaps"] == 0:
        print("✓ PASS: No overlapping cells!")
        print("✓ PASS: Overlap ratio is 0.0")
        print("\nCongratulations! Your implementation successfully eliminated all overlaps.")
        print(f"Your normalized wirelength: {normalized_metrics['normalized_wl']:.4f}")
    else:
        print("✗ FAIL: Overlaps still exist")
        print(f"  Need to eliminate overlaps in {normalized_metrics['num_cells_with_overlaps']} cells")
        print("\nSuggestions:")
        print("  1. Check your overlap_repulsion_loss() implementation")
        print("  2. Change lambdas (try increasing lambda_overlap)")
        print("  3. Change learning rate or number of epochs")

    # Generate visualization
    plot_placement(
        result["initial_cell_features"],
        result["final_cell_features"],
        pin_features,
        edge_list,
        filename="placement_result.png",
    )

if __name__ == "__main__":
    main()