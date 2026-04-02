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
- Improve convergence speed by tuning learning rate
- Implement better initial placement strategy
- Add visualization of optimization progress over time
"""

import math
import os
from enum import IntEnum
from itertools import product

import torch
import torch.optim as optim

from initialization import (
    analytic_quadratic_wl_initial as _analytic_quadratic_wl_initial,
    apply_initial_placement as _apply_initial_placement,
    replace_lite_initial as _replace_lite_initial,
)
from perturbations import (
    combined_placement_objective_value as _combined_placement_objective_value,
    greedy_cell_position_swaps as _greedy_cell_position_swaps,
    maybe_apply_position_noise_kick as _maybe_apply_position_noise_kick,
)
from analytics import (
    calculate_min_possible_normalized_wl,
    print_adjacency_matrix_and_stats,
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


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
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/plots"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    cell_features = cell_features.to(DEVICE)
    pin_features = pin_features.to(DEVICE)
    edge_list = edge_list.to(DEVICE)

    return cell_features, pin_features, edge_list

# ======= OPTIMIZATION CODE (edit this part) =======

def wirelength_attraction_loss(cell_features, pin_features, edge_list, use_smooth_manhattan=False):
    """Calculate loss based on total wirelength to minimize routing.

    This is a REFERENCE IMPLEMENTATION showing how to write a differentiable loss function.

    The loss is mean edge cost using smooth Manhattan distance
    sqrt(Δx²+α²) + sqrt(Δy²+α²) between connected pins (α small → L1).

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information
        edge_list: [E, 2] tensor with edges

    Returns:
        Scalar loss value
    """
    if edge_list.shape[0] == 0:
        return torch.tensor(0.0, device=cell_features.device, requires_grad=True)

    # Update absolute pin positions based on cell positions
    cell_positions = cell_features[:, 2:4]  # [N, 2]
    cell_indices = pin_features[:, 0].long()

    # Calculate absolute pin positions

    # BUG FIX: THE PREVIOUS IMPLEMENTATION USED X, Y AS CENTER OF THE CELL IN MOST PLACES BUT CORNER HERE. THIS FIXES HERE.
    # pin_absolute_x = cell_positions[cell_indices, 0] - cell_features[cell_indices, 4] / 2 + pin_features[:, 1]
    # pin_absolute_y = cell_positions[cell_indices, 1] - cell_features[cell_indices, 5] / 2 + pin_features[:, 2]

    # FORMER IMPLEMENTATION:
    pin_absolute_x = cell_positions[cell_indices, 0] + pin_features[:, 1]
    pin_absolute_y = cell_positions[cell_indices, 1] + pin_features[:, 2]

    # Get source and target pin positions for each edge
    src_pins = edge_list[:, 0].long()
    tgt_pins = edge_list[:, 1].long()

    src_x = pin_absolute_x[src_pins]
    src_y = pin_absolute_y[src_pins]
    tgt_x = pin_absolute_x[tgt_pins]
    tgt_y = pin_absolute_y[tgt_pins]

    # BUG FIX: THE PREVIOUS IMPLEMENTATION IS SMOOTH MAX, NOT MANHATTAN. THIS PROVIDES BOTH OPTIONS.
    if not use_smooth_manhattan:
        # Previous implementation (not true Manhattan — log-sum-exp ≈ max(|Δx|,|Δy|)):
        alpha = 0.1
        dx = torch.abs(src_x - tgt_x)
        dy = torch.abs(src_y - tgt_y)
        smooth_result = alpha * torch.logsumexp(
            torch.stack([dx / alpha, dy / alpha], dim=0), dim=0
        )
        # smooth_result = dx + dy
    else:
        # Smooth Manhattan: smooth_abs(Δx) + smooth_abs(Δy), smooth_abs(t) = sqrt(t² + α²)
        alpha = 0.1  # smoothing scale; approaches |Δx|+|Δy| as alpha -> 0
        delta_x = src_x - tgt_x
        delta_y = src_y - tgt_y
        smooth_result = torch.sqrt(delta_x * delta_x + alpha * alpha) + torch.sqrt(
            delta_y * delta_y + alpha * alpha
        )

    # Total wirelength
    total_wirelength = torch.sum(smooth_result)

    return total_wirelength / edge_list.shape[0]  # Normalize by number of edges


def overlap_repulsion_loss(cell_features, pin_features, edge_list):
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
    N = cell_features.shape[0]
    if N <= 1:
        # Scalar zero on the same graph as positions (not a disconnected tensor).
        return cell_features[:, 2:4].sum() * 0.0

    # TODO: Implement overlap detection and loss calculation here
    #
    # Your implementation should:
    # 1. Extract cell positions, widths, and heights
    # 2. Compute pairwise overlaps using vectorized operations
    # 3. Return a scalar loss that is zero when no overlaps exist
    #
    # Delete this placeholder and add your implementation:

    # Placeholder - returns a constant loss (REPLACE THIS!)
    # return torch.tensor(1.0, requires_grad=True)

    # Extract cell properties
    positions = cell_features[:, 2:4]  # [N, 2]
    widths = cell_features[:, 4]  # [N]
    heights = cell_features[:, 5]  # [N]

    # Pairwise center distances [N, N]
    dx = torch.abs(positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0))
    dy = torch.abs(positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0))

    # Pairwise minimum separations needed to avoid overlap [N, N]
    min_sep_x = 0.5 * (widths.unsqueeze(1) + widths.unsqueeze(0))
    min_sep_y = 0.5 * (heights.unsqueeze(1) + heights.unsqueeze(0))

    # Positive overlap along each axis (0 if no overlap on that axis)
    overlap_x = torch.relu(min_sep_x - dx)
    overlap_y = torch.relu(min_sep_y - dy)

    # Overlap area for each cell pair [N, N]
    overlap_area = overlap_x * overlap_y

    # Penetration-depth term provides stronger directional gradients,
    # especially for thin/sliver overlaps where area alone can be small.
    overlap_depth_sq = overlap_x ** 2 + overlap_y ** 2

    # Keep only unique pairs (i < j) to avoid self/duplicate counting
    pair_mask = torch.triu(
        torch.ones((N, N), dtype=torch.bool, device=cell_features.device), diagonal=1
    )
    overlap_area_pairs = overlap_area[pair_mask]
    overlap_depth_sq_pairs = overlap_depth_sq[pair_mask]

    # Ignore pairwise contributions below eps so float noise does not yield nonzero
    # loss when rectangles are separated (true overlap area is exactly zero).
    pair_eps = 1e-16
    has_ov = overlap_area_pairs > pair_eps
    zero = torch.zeros_like(overlap_area_pairs)
    overlap_area_pairs = torch.where(has_ov, overlap_area_pairs, zero)
    overlap_depth_sq_pairs = torch.where(has_ov, overlap_depth_sq_pairs, zero)

    # Composite penalty:
    # - area term reduces total overlap
    # - area^2 heavily punishes large collisions
    # - depth^2 creates strong gradients for narrow overlaps
    loss_area = overlap_area_pairs.sum() / N
    loss_area_quadratic = torch.sum(overlap_area_pairs ** 2) / (N * N)
    loss_depth = overlap_depth_sq_pairs.sum() / N
    return loss_area + 0.2 * loss_area_quadratic + 0.5 * loss_depth


def overlap_repulsion_loss_fast(cell_features, pin_features, edge_list, max_neighbors=512):
    """Same penalty as ``overlap_repulsion_loss`` but near-linear for spread-out layouts.

    Sorts cells by x-coordinate and uses a sliding window: for each offset
    k = 1..W it evaluates all (i, i+k) pairs as a single vectorized batch.
    Because the array is x-sorted, once the minimum x-gap at an offset exceeds
    the largest possible overlap width the loop terminates early.

    Complexity: O(N * W) where W is the effective window depth (number of
    x-neighbours within overlap range).  For well-spread placements W << N,
    giving near-linear runtime.  Worst case (all cells at the same x) degrades
    to O(N * max_neighbors) which is still bounded.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: not used
        edge_list: not used
        max_neighbors: cap on sliding-window depth to bound worst case.

    Returns:
        Scalar loss value compatible with ``overlap_repulsion_loss``.
    """
    N = cell_features.shape[0]
    if N <= 1:
        return cell_features[:, 2:4].sum() * 0.0

    positions = cell_features[:, 2:4]
    half_w = cell_features[:, 4] * 0.5
    half_h = cell_features[:, 5] * 0.5

    sort_idx = torch.argsort(positions[:, 0].detach())
    sx = positions[sort_idx, 0]
    sy = positions[sort_idx, 1]
    shw = half_w[sort_idx]
    shh = half_h[sort_idx]

    max_sep = (half_w.max() + half_w.max()).detach()

    sum_area = positions.new_tensor(0.0)
    sum_area_sq = positions.new_tensor(0.0)
    sum_depth_sq = positions.new_tensor(0.0)
    pair_eps = 1e-12
    W = min(max_neighbors, N - 1)

    for offset in range(1, W + 1):
        n_p = N - offset
        dx = sx[offset:] - sx[:n_p]
        if dx.min() > max_sep:
            break
        dy = torch.abs(sy[offset:] - sy[:n_p])
        sep_x = shw[offset:] + shw[:n_p]
        sep_y = shh[offset:] + shh[:n_p]
        ov_x = torch.relu(sep_x - dx)
        ov_y = torch.relu(sep_y - dy)
        area = ov_x * ov_y
        depth_sq = ov_x ** 2 + ov_y ** 2
        valid = area > pair_eps
        area = area * valid
        depth_sq = depth_sq * valid
        sum_area = sum_area + area.sum()
        sum_area_sq = sum_area_sq + (area ** 2).sum()
        sum_depth_sq = sum_depth_sq + depth_sq.sum()

    loss_area = sum_area / N
    loss_area_quadratic = sum_area_sq / (N * N)
    loss_depth = sum_depth_sq / N
    return loss_area + 0.2 * loss_area_quadratic + 0.5 * loss_depth


def density_overflow_loss(
    cell_features,
    bin_size=10.0,
    target_density=1.0,
    kernel_radius=2,
    penalty_exponent=2.0,
):
    """Penalize regions where smooth splatted cell density exceeds ``target_density``.

    Each cell splats its area onto a uniform grid with a separable tent (triangle)
    kernel over ``(2*kernel_radius+1)^2`` bins so gradients are smooth. Bin
    density is (weighted area sum) / (bin area).

    Overflow mass is summed over bins (crowding is not diluted by empty bins) and
    normalized by cell count: ``sum(relu(d - t)^p) / N``. Use ``penalty_exponent``
    > 2 for much stronger gradients on large peaks.

    ``cell_features`` positions are treated as rectangle **centers** (same as
    ``overlap_repulsion_loss``).
    """
    N = cell_features.shape[0]
    if N == 0:
        return cell_features[:, 0].sum() * 0.0
    if N == 1:
        return cell_features[:, 2:4].sum() * 0.0

    device = cell_features.device
    dtype = cell_features.dtype
    R = int(kernel_radius)
    if R < 0:
        raise ValueError("kernel_radius must be non-negative")
    h = float(bin_size)
    if h <= 0:
        raise ValueError("bin_size must be positive")
    bin_area = h * h

    centers = cell_features[:, 2:4]
    areas = cell_features[:, CellFeatureIdx.AREA].clamp(min=0)
    half_w = cell_features[:, CellFeatureIdx.WIDTH] * 0.5
    half_h = cell_features[:, CellFeatureIdx.HEIGHT] * 0.5

    pad = float(R + 1) * h
    xmin = (centers[:, 0] - half_w).min() - pad
    xmax = (centers[:, 0] + half_w).max() + pad
    ymin = (centers[:, 1] - half_h).min() - pad
    ymax = (centers[:, 1] + half_h).max() + pad

    span_x = xmax - xmin
    span_y = ymax - ymin
    nx = max(1, int(torch.ceil(span_x / h).item()))
    ny = max(1, int(torch.ceil(span_y / h).item()))

    tx = (centers[:, 0] - xmin) / h
    ty = (centers[:, 1] - ymin) / h
    ix = torch.floor(tx).long()
    iy = torch.floor(ty).long()

    rng = torch.arange(-R, R + 1, device=device)
    di_offsets, dj_offsets = torch.meshgrid(rng, rng, indexing="ij")
    di_flat = di_offsets.reshape(-1)  # [K]
    dj_flat = dj_offsets.reshape(-1)  # [K]

    bx_all = ix.unsqueeze(1) + di_flat.unsqueeze(0)  # [N, K]
    by_all = iy.unsqueeze(1) + dj_flat.unsqueeze(0)  # [N, K]
    cx = bx_all.to(dtype) + 0.5
    cy = by_all.to(dtype) + 0.5
    denom = float(R + 1)
    wx = torch.relu(1.0 - (tx.unsqueeze(1) - cx).abs() / denom)
    wy = torch.relu(1.0 - (ty.unsqueeze(1) - cy).abs() / denom)
    w_acc = wx * wy

    valid = (bx_all >= 0) & (bx_all < nx) & (by_all >= 0) & (by_all < ny)
    w_acc = w_acc * valid.to(dtype)
    w_sum = w_acc.sum(dim=1, keepdim=True).clamp(min=1e-12)
    w_norm = w_acc / w_sum

    contrib = areas.unsqueeze(1) * w_norm / bin_area
    contrib = contrib * valid.to(dtype)
    flat = (by_all * nx + bx_all).long()

    density_flat = torch.zeros(nx * ny, device=device, dtype=dtype)
    m = valid.reshape(-1)
    density_flat.index_add_(0, flat.reshape(-1)[m], contrib.reshape(-1)[m])

    d = density_flat.view(ny, nx)
    if torch.is_tensor(target_density):
        t = target_density.to(device=device, dtype=dtype)
    else:
        t = torch.tensor(float(target_density), device=device, dtype=dtype)
    excess = torch.relu(d - t)
    p = float(penalty_exponent)
    if p <= 0.0:
        raise ValueError("penalty_exponent must be positive")
    return (excess ** p).sum() / float(max(N, 1))


def _build_lr_scheduler(
    optimizer,
    schedule,
    num_epochs,
    lr,
    min_lr,
    warm_restart_T_0=None,
    warm_restart_T_mult=2,
):
    """Create a step-after-epoch LR scheduler, or None for fixed lr.

    schedule names:
        - cosine: CosineAnnealingLR from lr down to min_lr
        - cosine_warm_restarts: CosineAnnealingWarmRestarts (periodic LR spikes)
        - linear: linear multiplicative decay from lr to min_lr
        - exponential: per-epoch decay so lr * gamma^num_epochs ≈ min_lr
        - step: StepLR every ~quarter of training, gamma=0.5
        - multistep: drops at ~33% and ~66% of training
        - onecycle: OneCycleLR peaking at lr, ending near min_lr
        - constant / none: no scheduler (lr fixed at initial value)
    """
    name = schedule.lower().strip()
    if name in ("none", "constant"):
        return None
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(num_epochs, 1), eta_min=min_lr
        )
    if name in ("cosine_warm_restarts", "cosine_warm_restart"):
        t0 = warm_restart_T_0
        if t0 is None:
            t0 = max(num_epochs // 5, 1)
        else:
            t0 = int(t0)
        if t0 < 1:
            raise ValueError("warm_restart_T_0 must be >= 1")
        t_mult = int(warm_restart_T_mult)
        if t_mult < 1:
            raise ValueError("warm_restart_T_mult must be >= 1")
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=t_mult, eta_min=float(min_lr)
        )
    if name == "linear":
        lr = float(lr)
        if lr <= 0:
            raise ValueError("lr must be positive for linear schedule")
        min_ratio = min(float(min_lr), lr) / lr

        def lr_lambda(epoch):
            if num_epochs <= 1:
                return 1.0
            t = epoch / (num_epochs - 1)
            return (1.0 - t) + t * min_ratio

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if name == "exponential":
        lr = float(lr)
        if num_epochs < 1 or lr <= 0:
            gamma = 1.0
        else:
            ratio = max(float(min_lr) / lr, 1e-12)
            gamma = ratio ** (1.0 / max(num_epochs, 1))
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if name == "step":
        step_size = max(num_epochs // 4, 1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    if name == "multistep":
        milestones = sorted(
            {int(num_epochs * f) for f in (0.33, 0.66)}
        )
        milestones = [m for m in milestones if 0 < m < num_epochs]
        if not milestones:
            milestones = [max(num_epochs // 2, 1)]
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.5
        )
    if name == "onecycle":
        lr = float(lr)
        min_lr = float(min_lr)
        if num_epochs < 2:
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(num_epochs, 1), eta_min=min_lr
            )
        div_factor = max(lr / max(min_lr, 1e-12), 1.0)
        final_div_factor = max(lr / (min_lr * div_factor), 1.0)
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=num_epochs,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )
    raise ValueError(
        f"Unknown lr_schedule {schedule!r}. "
        "Use: cosine, cosine_warm_restarts, linear, exponential, step, "
        "multistep, onecycle, constant."
    )


def train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=20000,
    lr=1.0,
    min_lr=1e-5,
    use_density_loss=False,
    optimizer=None,
    momentum=0.9,
    lambda_density=2500.0,
    lambda_density_ramp_power=1.0,
    lambda_density_min_fraction=0.12,
    density_bin_size=10.0,
    density_penalty_exponent=3.0,
    density_kernel_radius=2,
    density_target_start=5.0,
    density_target_end=0.1,
    lambda_wl_in_wl_phase=1.0,
    lambda_ov_in_wl_phase=0.0,
    lambda_wl_in_ov_phase=1.0,
    lambda_ov_in_ov_phase=5.0,
    lambda_ov_with_density=50.0,
    overlap_focus_loss_threshold=1e-12,
    overlap_dominant_head_fraction=0.0,
    overlap_dominant_tail_fraction=0.5,
    use_efficient_overlap_loss=True,
    phase_transition_lr_boost=2.0,
    final_lr_boost=1.0,
    final_lr_boost_epochs=0,
    greedy_swap_enabled=True,
    greedy_swap_after_init=True,
    greedy_swap_trigger_threshold=10.0,
    greedy_swap_max_rounds=100,
    greedy_swap_max_pairs_per_round=5000,
    greedy_swap_max_cell_area=50.0,
    greedy_swap_seed=None,
    periodic_relayout_interval=5000,
    periodic_replace_lite=False,
    periodic_quadratic_wl=False,
    position_noise_kick_enabled=False,
    position_noise_kick_interval=10000,
    position_noise_kick_std=10.0,
    position_noise_kick_until_fraction=0.6,
    lr_schedule="cosine",
    warm_restart_T_0=None,
    warm_restart_T_mult=2,
    lr_warmup_epochs=0,
    betas=(0.9, 0.999),
    weight_decay=1e-5,
    norm_max=100.0,
    initial_placement="replace_lite_noisy",
    eplace_lite_iters=100,
    eplace_lite_step=0.2,
    replace_lite_margin=1.2,
    replace_lite_row_gap=0.5,
    replace_lite_post_noise_std=0.5,
    random_spread_radius=None,
    plot_initial_placement=True,
    plot_initial_placement_filename="placement_before_after_initial_placement.png",
    plot_initial_swaps=True,
    plot_initial_swaps_filename="placement_before_after_initial_swaps.png",
    plot_epoch_snapshots=True,
    plot_epoch_snapshot_interval=10000,
    plot_epoch_snapshot_prefix="placement_epoch",
    verbose=True,
    log_interval=100,
):
    """Train the placement optimization using gradient descent.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity
        num_epochs: Number of optimization iterations
        lr: Initial learning rate for Adam optimizer (or peak for onecycle)
        min_lr: Floor/target end LR (cosine eta_min, linear endpoint, etc.)
        use_density_loss: If True, optimize ``λ_wl·WL + λ_d(epoch)·density`` with a smooth
            grid density overflow term (see ``density_overflow_loss``) and anneal
            ``target_density`` from ``density_target_start`` to ``density_target_end``.
            Overlap-dominant windows are disabled. If False, use overlap pairing loss
            and the head/tail overlap-dominant schedule below.
        optimizer: ``"adam"``, ``"sgd"``, or ``"sgd_nesterov"``. ``None`` chooses
            ``sgd_nesterov`` when ``use_density_loss`` else ``adam``.
        momentum: SGD momentum (used when optimizer is SGD / Nesterov).
        lambda_density: Final multiplier on density loss at the end of training.
        lambda_density_ramp_power: Power ``r`` for the ramp factor
            ``prog=epoch/(num_epochs-1)``; ramp is ``prog**r``.
        lambda_density_min_fraction: Minimum fraction of ``lambda_density`` applied
            from the first epoch (so spreading pressure is never fully off). Effective
            weight is ``lambda_density * (min_fraction + (1-min_fraction) * ramp)``.
        density_bin_size: Grid bin width/height for density field.
        density_penalty_exponent: Power on per-bin overflow (default 3); larger
            punishes crowded bins much more aggressively than empty ones.
        density_kernel_radius: Tent-kernel half-width in bins (nonnegative int).
        density_target_start / density_target_end: Annealed overflow threshold
            (higher start = more tolerant early).
        lambda_wl_in_ov_phase: Weight for wirelength loss in overlap-dominant phase
        lambda_wl_in_wl_phase: Weight for wirelength loss in wirelength-dominant phase
        lambda_ov_in_ov_phase: Weight for overlap loss in overlap-dominant phase
        lambda_ov_in_wl_phase: Weight for overlap loss in wirelength-dominant phase
        overlap_focus_loss_threshold: While ``overlap_repulsion_loss`` (raw, unweighted)
            is **strictly greater** than this **and** the run is in an overlap-dominant
            window (head and/or tail), optimize **overlap only**
            (``lambda_overlap * overlap_loss``). Otherwise use wirelength + overlap.
            Align with pair mask scale in ``overlap_repulsion_loss`` (~1e-12); larger
            values hand off to WL earlier (softer overlaps still get WL gradient).
        overlap_dominant_head_fraction: Fraction of training (from the start) in which
            overlap-only mode is allowed. ``0.2`` → only epochs in the first 20%.
            ``0`` disables head overlap-dominant mode; ``1`` allows it from epoch 0 onward.
        overlap_dominant_tail_fraction: Fraction of training (from the end) in which the
            overlap-only branch above is allowed. ``0.2`` → only epochs in the last 20%.
            ``0`` disables overlap-only mode entirely; ``1`` keeps the old behavior (allowed
            from epoch 0).
        greedy_swap_enabled: If True, run pairwise center swaps that accept only when
            ``λ_wl·WL + λ_ov·overlap`` improves.
        greedy_swap_after_init: Run that search once after ``_apply_initial_placement``.
        greedy_swap_trigger_threshold: During training, when the combined objective
            **crosses upward** (previous epoch ``≤`` threshold ``<`` current), run another
            greedy pass. ``None`` disables re-entry (init-only swaps).
        greedy_swap_max_rounds / greedy_swap_max_pairs_per_round: Cap work per pass.
        greedy_swap_max_cell_area: Only cells with ``cell_features[:, AREA] <=`` this
            value are allowed in swaps (so a pair with a larger cell is never tried).
            ``None`` or ``<= 0`` disables the filter (all cells eligible).
        greedy_swap_seed: Optional RNG seed for shuffling pair order (``None`` = nondeterministic).
        periodic_relayout_interval: Run optional periodic relayout passes every N epochs.
        periodic_replace_lite: If True, run replace-lite on current state every interval.
        periodic_quadratic_wl: If True, run quadratic-WL init on current state every interval.
        position_noise_kick_enabled: If True, periodically add Gaussian noise kicks.
        position_noise_kick_interval: Apply a noise kick every N epochs.
        position_noise_kick_std: Standard deviation of Gaussian kick per axis.
        position_noise_kick_until_fraction: Disable noise kicks after this fraction of
            training. ``0.6`` means kicks only occur in the first 60% of epochs.
        lr_schedule: Name passed to _build_lr_scheduler (cosine, linear, ...)
        warm_restart_T_0: For cosine_warm_restarts, first restart period in epochs
            (default: max(num_epochs // 5, 1)).
        warm_restart_T_mult: For cosine_warm_restarts, multiplier for period length
            after each restart (PyTorch default is 1; we use 2).
        lr_warmup_epochs: Linear LR warmup for this many epochs: lr scales from
            ``lr / warmup`` to ``lr``, then the main ``lr_schedule`` runs for the
            remaining epochs (re-timed to ``num_epochs - lr_warmup_epochs``). 0 disables.
        betas: Adam (beta1, beta2) for moment decay; passed to torch.optim.Adam.
        weight_decay: Adam L2 penalty coefficient.
        norm_max: Gradient clip max norm for ``cell_positions``.
        initial_placement: Init strategy: ``preserve``, ``random``, ``eplace_lite``
            (overlap spreading), ``quadratic_wl`` (Laplacian eigenvector / quadratic WL
            layout), ``random_then_quadratic_wl``, ``replace_lite`` (Fiedler + shelf),
            ``replace_lite_noisy``, ``random_then_replace_lite``,
            ``random_then_replace_lite_noisy``. Aliases: see ``_apply_initial_placement``.
        eplace_lite_iters / eplace_lite_step: Iteration count and step for eplace_lite.
        replace_lite_margin / replace_lite_row_gap: Bounding strip and row spacing for replace_lite.
        replace_lite_post_noise_std: Standard deviation per axis for Gaussian jitter on (x, y)
            after ``replace_lite`` in the ``*_noisy`` / ``*_gaussian`` placement modes;
            ignored for other modes. 0 disables jitter in those modes.
        random_spread_radius: Disk radius for random init; None → sqrt(total area) * 0.6.
        plot_initial_placement: If True, save a side-by-side figure (via ``plot_placement``)
            of cell positions before vs after ``_apply_initial_placement`` only (before greedy
            init swaps).
        plot_initial_placement_filename: Basename under ``OUTPUT_DIR`` for that figure.
        plot_initial_swaps: If True and greedy init swaps run, save a side-by-side figure
            of positions after initial placement vs after those swaps.
        plot_initial_swaps_filename: Basename under ``OUTPUT_DIR`` for the swap comparison.
        plot_epoch_snapshots: If True, save periodic training snapshots.
        plot_epoch_snapshot_interval: Save a snapshot every N epochs (1-based epoch index).
        plot_epoch_snapshot_prefix: Prefix for snapshot filenames, e.g.
            ``placement_epoch_10000.png``.
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
    cell_features_before_init = cell_features.clone()
    _apply_initial_placement(
        initial_placement,
        cell_features,
        pin_features,
        edge_list,
        random_spread_radius=random_spread_radius,
        eplace_lite_iters=eplace_lite_iters,
        eplace_lite_step=eplace_lite_step,
        replace_lite_margin=replace_lite_margin,
        replace_lite_row_gap=replace_lite_row_gap,
        replace_lite_post_noise_std=replace_lite_post_noise_std,
    )
    cell_features_after_initial_placement = cell_features.clone()

    swap_gen = None
    if greedy_swap_seed is not None:
        swap_gen = torch.Generator(device="cpu")
        swap_gen.manual_seed(int(greedy_swap_seed))
    n_acc_init_swaps = 0
    if greedy_swap_enabled and greedy_swap_after_init:
        n_acc_init_swaps = _greedy_cell_position_swaps(
            cell_features,
            cell_features[:, 2:4],
            pin_features,
            edge_list,
            max_rounds=int(greedy_swap_max_rounds),
            max_pairs_per_round=greedy_swap_max_pairs_per_round,
            generator=swap_gen,
            max_swap_cell_area=greedy_swap_max_cell_area
        )
        if verbose and n_acc_init_swaps > 0:
            print(
                f"Greedy position swaps (after init): accepted {n_acc_init_swaps} swaps."
            )

    if plot_initial_placement:
        mode_lbl = str(initial_placement)
        plot_placement(
            cell_features_before_init,
            cell_features_after_initial_placement,
            pin_features,
            edge_list,
            filename=plot_initial_placement_filename,
            titles=(
                "Before initial placement",
                f"After initial placement ({mode_lbl})",
            ),
        )
        if verbose:
            out_ip = os.path.join(OUTPUT_DIR, plot_initial_placement_filename)
            print(f"Saved initial-placement preview: {out_ip}")

    if (
        plot_initial_swaps
        and greedy_swap_enabled
        and greedy_swap_after_init
    ):
        plot_placement(
            cell_features_after_initial_placement,
            cell_features.clone(),
            pin_features,
            edge_list,
            filename=plot_initial_swaps_filename,
            titles=(
                "Before greedy init swaps",
                f"After greedy init swaps ({n_acc_init_swaps} accepted)",
            ),
        )
        if verbose:
            out_sw = os.path.join(OUTPUT_DIR, plot_initial_swaps_filename)
            print(f"Saved initial-swap preview: {out_sw}")
    initial_cell_features = cell_features.clone()

    # Make only cell positions require gradients
    cell_positions = cell_features[:, 2:4].clone().detach()
    cell_positions.requires_grad_(True)

    b1, b2 = float(betas[0]), float(betas[1])
    wd = float(weight_decay)
    if wd < 0.0:
        raise ValueError("weight_decay must be non-negative")
    opt_name = optimizer
    if opt_name is None:
        opt_name = "adam"
    opt_name = str(opt_name).lower().replace("-", "_")
    mom = float(momentum)
    if mom < 0.0:
        raise ValueError("momentum must be non-negative")
    if opt_name in ("adam",):
        if not (0.0 < b1 < 1.0 and 0.0 < b2 < 1.0):
            raise ValueError("betas must be a pair of values strictly between 0 and 1")
        torch_optimizer = optim.Adam(
            [cell_positions], lr=lr, betas=(b1, b2), weight_decay=wd
        )
    elif opt_name in ("sgd", "sgd_nesterov", "nesterov"):
        use_nesterov = opt_name in ("sgd_nesterov", "nesterov")
        if use_nesterov and mom <= 0.0:
            raise ValueError("Nesterov SGD requires momentum > 0")
        torch_optimizer = optim.SGD(
            [cell_positions],
            lr=lr,
            momentum=mom,
            weight_decay=wd,
            nesterov=use_nesterov,
        )
    else:
        raise ValueError(
            f"Unknown optimizer {opt_name!r}; use 'adam', 'sgd', or 'sgd_nesterov'"
        )

    lr_float = float(lr)
    warm_w = max(0, min(int(lr_warmup_epochs), num_epochs))
    post_warm_epochs = num_epochs - warm_w
    if post_warm_epochs <= 0:
        scheduler = None
    else:
        scheduler = _build_lr_scheduler(
            torch_optimizer,
            lr_schedule,
            post_warm_epochs,
            lr_float,
            min_lr,
            warm_restart_T_0=warm_restart_T_0,
            warm_restart_T_mult=warm_restart_T_mult,
        )

    # Track loss history
    loss_history = {
        "total_loss": [],
        "wirelength_loss": [],
        "overlap_loss": [],
    }
    if use_density_loss:
        loss_history["density_loss"] = []

    # Sanity check inputs.
    odhf = float(overlap_dominant_head_fraction)
    odtf = float(overlap_dominant_tail_fraction)
    kick_until_fraction = float(position_noise_kick_until_fraction)
    if not 0.0 <= odhf <= 1.0:
        raise ValueError("overlap_dominant_head_fraction must be in [0, 1]")
    if not 0.0 <= odtf <= 1.0:
        raise ValueError("overlap_dominant_tail_fraction must be in [0, 1]")
    if not 0.0 <= kick_until_fraction <= 1.0:
        raise ValueError("position_noise_kick_until_fraction must be in [0, 1]")
    if use_density_loss:
        if float(density_bin_size) <= 0.0:
            raise ValueError("density_bin_size must be positive")
        if float(lambda_density) < 0.0:
            raise ValueError("lambda_density must be non-negative")
        if float(lambda_density_ramp_power) < 0.0:
            raise ValueError("lambda_density_ramp_power must be non-negative")
        if int(density_kernel_radius) < 0:
            raise ValueError("density_kernel_radius must be non-negative")
        mf = float(lambda_density_min_fraction)
        if not 0.0 <= mf <= 1.0:
            raise ValueError("lambda_density_min_fraction must be in [0, 1]")
        if float(density_penalty_exponent) <= 0.0:
            raise ValueError("density_penalty_exponent must be positive")
    overlap_dominant_head_end_epoch = int(odhf * num_epochs)
    overlap_dominant_start_epoch = int((1.0 - odtf) * num_epochs)
    kick_until_epoch = int(kick_until_fraction * num_epochs)

    # Increase OV phase loss as N increases.
    N = cell_features.shape[0]
    lambda_ov_in_ov_phase_ = float(lambda_ov_in_ov_phase)
    print(f'Example contains {N} cells.')
    if N > 50:
      lambda_ov_in_ov_phase_ = float(lambda_ov_in_ov_phase) * float(N) / 10.0
    if N > 2000:
      # Use final boost for large inputs where overlap is harder.
      final_lr_boost = float(final_lr_boost) * float(N)
      # Also add head OV period.
      lambda_ov_in_ov_phase_ = float(lambda_ov_in_ov_phase) * float(N) * 25.0
      overlap_dominant_head_fraction += 0.1
      lambda_ov_in_wl_phase += 0.1

    best_score = None  # (overlap_loss, wirelength), lexicographic min
    best_cell_positions = None
    best_epoch = None

    # Training loop
    for epoch in range(num_epochs):
        # Reset optimizer state after a phase transition, otherwise gradients will explode.
        if (not use_density_loss) and epoch > 0 and (
            epoch == overlap_dominant_head_end_epoch
            or epoch == overlap_dominant_start_epoch
        ):
            torch_optimizer.state.clear()
            boost = float(phase_transition_lr_boost)
            if boost > 1.0:
                for g in torch_optimizer.param_groups:
                    g["lr"] = min(g["lr"] * boost, lr_float)

        torch_optimizer.zero_grad()

        # Functional construction — no clone, no in-place op, clean autograd graph.
        cell_features_current = torch.cat(
            [cell_features[:, :2], cell_positions, cell_features[:, 4:]], dim=1
        )

        # Calculate losses
        wl_loss = wirelength_attraction_loss(
            cell_features_current, pin_features, edge_list, use_smooth_manhattan=False
        )
        if use_efficient_overlap_loss and N > 1024*8:
            overlap_loss = overlap_repulsion_loss_fast(
                cell_features_current, pin_features, edge_list
            )
        else:
            overlap_loss = overlap_repulsion_loss(
                cell_features_current, pin_features, edge_list
            )

        if overlap_loss.item() > overlap_focus_loss_threshold:
            final_boost = float(final_lr_boost)
            final_boost_n = int(final_lr_boost_epochs)
            if final_boost > 1.0 and final_boost_n > 0 and epoch >= num_epochs - final_boost_n:
                lambda_ov_in_ov_phase_ = float(lambda_ov_in_ov_phase) * final_boost

        if use_density_loss:
            prog = float(epoch) / float(max(num_epochs - 1, 1))
            ramp = prog ** float(lambda_density_ramp_power)
            mf = float(lambda_density_min_fraction)
            lambda_d_cur = float(lambda_density) * (
                mf + (1.0 - mf) * ramp
            )
            td_cur = float(density_target_start) + (
                float(density_target_end) - float(density_target_start)
            ) * prog
            density_loss = density_overflow_loss(
                cell_features_current,
                bin_size=float(density_bin_size),
                target_density=td_cur,
                kernel_radius=int(density_kernel_radius),
                penalty_exponent=float(density_penalty_exponent),
            )
            total_loss = float(lambda_wl_in_wl_phase) * wl_loss + lambda_d_cur * density_loss + lambda_ov_with_density * overlap_loss
        else:
            threshold = float(overlap_focus_loss_threshold)
            overlap_dominant_window = (
                epoch < overlap_dominant_head_end_epoch
                or epoch >= overlap_dominant_start_epoch
            )
            overlap_dominant = overlap_dominant_window and float(
                overlap_loss.detach()
            ) > threshold
            # overlap_dominant = overlap_dominant_window
            if overlap_dominant:
                total_loss = float(lambda_ov_in_ov_phase_) * overlap_loss + float(
                    lambda_wl_in_ov_phase
                ) * wl_loss
            else:
                total_loss = float(lambda_wl_in_wl_phase) * wl_loss + float(
                    lambda_ov_in_wl_phase
                ) * overlap_loss

        # Backward pass
        total_loss.backward()

        # Adjustable clipping to prevent gradient explosion.
        torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=float(norm_max))

        # Snapshot BEFORE the optimizer step so saved positions match the loss.
        with torch.no_grad():
            wl_score = wl_loss.item()
            ov_score = overlap_loss.item()
            # Save the best placement so far.
            if best_score is None or (ov_score, wl_score) < best_score:
                # n_overlap_cells = calculate_cells_with_overlaps(cell_features_scoring)
                # if n_overlap_cells == 0:
                best_score = (ov_score, wl_score)
                best_cell_positions = cell_positions.detach().clone()
                best_epoch = epoch

        # Update positions
        torch_optimizer.step()
        if scheduler is not None and epoch >= warm_w:
            scheduler.step()

        _maybe_apply_position_noise_kick(
            cell_positions,
            epoch,
            enabled=position_noise_kick_enabled,
            interval=position_noise_kick_interval,
            kick_std=position_noise_kick_std,
            kick_until_epoch=kick_until_epoch,
            swap_gen=swap_gen,
            verbose=verbose,
        )

        if (
            epoch % 1000 == 0
            and greedy_swap_enabled
            and greedy_swap_trigger_threshold is not None
            and float(greedy_swap_trigger_threshold) < wl_score
        ):
            n_acc = _greedy_cell_position_swaps(
                cell_features,
                cell_positions,
                pin_features,
                edge_list,
                max_rounds=int(greedy_swap_max_rounds),
                max_pairs_per_round=greedy_swap_max_pairs_per_round,
                generator=swap_gen,
                max_swap_cell_area=greedy_swap_max_cell_area,
            )
            cell_features[:, 2:4] = cell_positions.detach()
            if verbose and n_acc > 0:
                print(
                    f"Epoch {epoch}: greedy swaps (combined loss crossed above "
                    f"{float(greedy_swap_trigger_threshold):g}) accepted {n_acc} swaps."
                )

        if (
            plot_epoch_snapshots
            and int(plot_epoch_snapshot_interval) > 0
            and ((epoch + 1) % int(plot_epoch_snapshot_interval) == 0)
        ):
            snap_filename = f"{plot_epoch_snapshot_prefix}_{epoch + 1}.png"
            plot_placement(
                initial_cell_features,
                cell_features_current,
                pin_features,
                edge_list,
                filename=snap_filename,
                titles=(
                    "Initial placement",
                    f"Placement at epoch {epoch + 1}",
                ),
            )
            if verbose:
                out_snap = os.path.join(OUTPUT_DIR, snap_filename)
                print(f"Saved epoch snapshot: {out_snap}")

        # Record losses
        loss_history["total_loss"].append(total_loss.item())
        loss_history["wirelength_loss"].append(wl_loss.item())
        loss_history["overlap_loss"].append(overlap_loss.item())
        if use_density_loss:
            loss_history["density_loss"].append(density_loss.item())

        # Log progress
        if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            current_lr = torch_optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}/{num_epochs}:")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  Wirelength Loss: {wl_loss.item():.6f}")
            if use_density_loss:
                prog = float(epoch) / float(max(num_epochs - 1, 1))
                ramp = prog ** float(lambda_density_ramp_power)
                mf = float(lambda_density_min_fraction)
                lambda_d_cur = float(lambda_density) * (
                    mf + (1.0 - mf) * ramp
                )
                td_cur = float(density_target_start) + (
                    float(density_target_end) - float(density_target_start)
                ) * prog
                print(f"  Density Loss: {density_loss.item():.6f}")
                print(
                    f"  λ_density (eff.): {lambda_d_cur:.6f}, "
                    f"target_density: {td_cur:.4f}, "
                    f"p={float(density_penalty_exponent):g}"
                )
            print(f"  Overlap Loss (diag): {overlap_loss.item():.6f}")

    # Return best-seen placement (not necessarily last epoch)
    final_cell_features = cell_features.clone()
    if best_cell_positions is not None:
        final_cell_features[:, 2:4] = best_cell_positions
    else:
        final_cell_features[:, 2:4] = cell_positions.detach()

    if best_cell_positions is not None:
        if verbose or True:
            print(
                f"Best placement snapshot at epoch {best_epoch} "
                f"(overlap_loss={best_score[0]:.6f}, "
                f"wirelength_loss={best_score[1]:.6f})"
            )

    return {
        "final_cell_features": final_cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": loss_history,
    }


def hyperparameter_search():
    """Grid search over training hyperparameters on one fixed placement instance.

    Edit the local variables at the top of this function to change the problem,
    search grids, and logging. Trials run with ``use_density_loss=True``; grids
    for overlap-dominant lambdas are still passed through but ignored by training.
    Density strength hyperparameters (``lambda_density``, min fraction, ramp power,
    penalty exponent, bin/grid targets) are grid-searched via ``itertools.product`` —
    shrink any list to a single value to reduce the number of trials.
    Scoring matches the test harness: fewer cells in overlaps, then lower overlap
    ratio, then lower normalized wirelength.

    Returns:
        Dictionary with best_hyperparameters, best_metrics (calculate_normalized_metrics),
        and result (train_placement return dict for the best run).
    """
    # --- instance & run controls ---
    num_macros = 3
    num_std_cells = 50
    seed = 42
    verbose_trials = False
    log_interval = 200

    # --- search grids ---
    num_epochs_grid = [5000]
    lr_grid = [1.0]
    min_lr_grid = [1e-5]
    lambda_wl_in_ov_phase_grid = [1.0]
    lambda_ov_in_ov_phase_grid = [50.0]
    # Wirelength weight in density runs (only λ_wl_wl matters when use_density_loss=True).
    lambda_wl_in_wl_phase_grid = [150.0]
    lambda_ov_in_wl_phase_grid = [5.0]
    overlap_dominant_head_fraction_grid = [1.0]
    overlap_dominant_tail_fraction_grid = [0.0]
    # Raw overlap_repulsion_loss above this => overlap-only phase for that step.
    overlap_focus_loss_threshold_grid = [0.1]
    lr_schedule_grid = ["cosine"]
    beta1_grid = [0.9]
    beta2_grid = [0.999]
    weight_decay_grid = [1e-5]
    norm_max_grid = [100.0]
    lr_warmup_epochs_grid = [0]
    # Gaussian jitter std (per x/y) after replace_lite in *_noisy modes; ignored otherwise.
    replace_lite_post_noise_std_grid = [5.0]
    # None => greedy swaps only after init; float => also when combined loss crosses up through it.
    greedy_swap_trigger_threshold_grid = [50.0]
    # Density objective (trials use use_density_loss=True). Cartesian product × other grids.
    density_bin_size_grid = [10.0]
    density_kernel_radius_grid = [4]
    density_target_start_grid = [4.0]
    density_target_end_grid = [0.08]
    lambda_density_grid = [2000.0, 5500.0]
    lambda_density_min_fraction_grid = [0.05, 0.18]
    lambda_density_ramp_power_grid = [1.0]
    density_penalty_exponent_grid = [3.0, 4.5]
    initial_placement_grid = [
        # "preserve",  # keep cloned x,y (after hyperparameter_search disk preset)
        # "random",  # uniform disk (uses random_spread_radius in the trial loop)
        # "eplace_lite",  # overlap spreading iterations
        # "replace_lite",  # Fiedler order + row shelf
        # "random_then_replace_lite",  # disk sample then replace_lite
        "replace_lite_noisy",  # replace_lite then Gaussian center jitter
        # "random_then_replace_lite_noisy",  # disk sample then replace_lite then Gaussian center jitter
        # "quadratic_wl",
        # "random_then_quadratic_wl",
    ]

    torch.manual_seed(seed)
    cell_features, pin_features, edge_list = generate_placement_input(
        num_macros, num_std_cells
    )

    total_cells = cell_features.shape[0]
    spread_radius = 30.0
    dev = cell_features.device
    angles = torch.rand(total_cells, device=dev) * 2 * 3.14159
    radii = torch.rand(total_cells, device=dev) * spread_radius
    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)

    base_combos = list(
        product(
            num_epochs_grid,
            lr_grid,
            min_lr_grid,
            lambda_wl_in_ov_phase_grid,
            lambda_wl_in_wl_phase_grid,
            lambda_ov_in_ov_phase_grid,
            lambda_ov_in_wl_phase_grid,
            overlap_dominant_head_fraction_grid,
            overlap_dominant_tail_fraction_grid,
            overlap_focus_loss_threshold_grid,
            lr_schedule_grid,
            beta1_grid,
            beta2_grid,
            weight_decay_grid,
            norm_max_grid,
            lr_warmup_epochs_grid,
            replace_lite_post_noise_std_grid,
            greedy_swap_trigger_threshold_grid,
            initial_placement_grid,
            density_bin_size_grid,
            density_kernel_radius_grid,
            density_target_start_grid,
            density_target_end_grid,
            lambda_density_grid,
            lambda_density_min_fraction_grid,
            lambda_density_ramp_power_grid,
            density_penalty_exponent_grid,
        )
    )
    if not base_combos:
        raise ValueError(
            "hyperparameter_search: grid produced no combinations (check grids at top)"
        )

    best_key = None
    best_hyperparameters = None
    best_metrics = None
    best_result = None

    ep_desc = (
        f"epochs={num_epochs_grid[0]}"
        if len(num_epochs_grid) == 1
        else f"epochs in {sorted(num_epochs_grid)}"
    )
    print(
        f"Hyperparameter search: {len(base_combos)} combos, "
        f"{ep_desc} "
        f"({num_macros} macros, {num_std_cells} std cells, seed={seed})\n"
    )

    for trial_idx, (
        n_ep,
        lr,
        min_lr,
        lambda_wl_in_ov_phase,
        lambda_wl_in_wl_phase,
        lambda_ov_in_ov_phase,
        lambda_ov_in_wl_phase,
        odhf,
        odtf,
        ov_focus_thr,
        sched,
        b1,
        b2,
        wd,
        nmax,
        warm_ep,
        rpl_noise_std,
        gst,
        init_pl,
        d_bin,
        d_kr,
        d_ts,
        d_te,
        d_ld,
        d_ldmf,
        d_ldrp,
        d_dpe,
    ) in enumerate(base_combos):
        _pl = str(init_pl).lower().replace("-", "_")
        use_spread = _pl in (
            "random",
            "random_then_replace_lite",
            "random_replace_lite",
            "rand_replace_lite",
            "random_then_replacelite",
            "random_then_replace_lite_noisy",
            "random_then_replace_lite_gaussian",
            "random_replace_lite_noisy",
            "rand_replace_lite_noisy",
            "random_then_replacelite_noisy",
            "random_then_quadratic_wl",
            "random_then_analytic_wl",
            "random_quadratic_wl",
        )
        result = train_placement(
            cell_features,
            pin_features,
            edge_list,
            use_density_loss=True,
            density_bin_size=float(d_bin),
            density_kernel_radius=int(d_kr),
            density_target_start=float(d_ts),
            density_target_end=float(d_te),
            lambda_density=float(d_ld),
            lambda_density_min_fraction=float(d_ldmf),
            lambda_density_ramp_power=float(d_ldrp),
            density_penalty_exponent=float(d_dpe),
            num_epochs=n_ep,
            lr=lr,
            min_lr=min_lr,
            lambda_wl_in_ov_phase=lambda_wl_in_ov_phase,
            lambda_wl_in_wl_phase=lambda_wl_in_wl_phase,
            lambda_ov_in_ov_phase=lambda_ov_in_ov_phase,
            lambda_ov_in_wl_phase=lambda_ov_in_wl_phase,
            overlap_dominant_head_fraction=float(odhf),
            overlap_dominant_tail_fraction=float(odtf),
            overlap_focus_loss_threshold=float(ov_focus_thr),
            lr_schedule=sched,
            betas=(float(b1), float(b2)),
            weight_decay=float(wd),
            norm_max=float(nmax),
            lr_warmup_epochs=int(warm_ep),
            replace_lite_post_noise_std=float(rpl_noise_std),
            greedy_swap_trigger_threshold=(
                None if gst is None else float(gst)
            ),
            initial_placement=init_pl,
            random_spread_radius=spread_radius if use_spread else None,
            plot_initial_placement=False,
            plot_initial_swaps=False,
            plot_epoch_snapshots=False,
            verbose=verbose_trials,
            log_interval=log_interval,
        )
        metrics = calculate_normalized_metrics(
            result["final_cell_features"], pin_features, edge_list
        )
        key = (
            metrics["num_cells_with_overlaps"],
            metrics["overlap_ratio"],
            metrics["normalized_wl"],
        )
        improved = best_key is None or key < best_key
        if improved:
            best_key = key
            best_hyperparameters = {
                "use_density_loss": True,
                "lr": lr,
                "min_lr": min_lr,
                "lambda_wl_in_ov_phase": lambda_wl_in_ov_phase,
                "lambda_wl_in_wl_phase": lambda_wl_in_wl_phase,
                "lambda_ov_in_ov_phase": lambda_ov_in_ov_phase,
                "lambda_ov_in_wl_phase": lambda_ov_in_wl_phase,
                "overlap_dominant_head_fraction": float(odhf),
                "overlap_dominant_tail_fraction": float(odtf),
                "overlap_focus_loss_threshold": float(ov_focus_thr),
                "lr_schedule": sched,
                "betas": (float(b1), float(b2)),
                "weight_decay": float(wd),
                "norm_max": float(nmax),
                "num_epochs": n_ep,
                "lr_warmup_epochs": int(warm_ep),
                "replace_lite_post_noise_std": float(rpl_noise_std),
                "greedy_swap_trigger_threshold": gst,
                "initial_placement": init_pl,
                "density_bin_size": float(d_bin),
                "density_kernel_radius": int(d_kr),
                "density_target_start": float(d_ts),
                "density_target_end": float(d_te),
                "lambda_density": float(d_ld),
                "lambda_density_min_fraction": float(d_ldmf),
                "lambda_density_ramp_power": float(d_ldrp),
                "density_penalty_exponent": float(d_dpe),
            }
            best_metrics = metrics
            best_result = result

        print(
            f"[{trial_idx + 1}/{len(base_combos)}] "
            f"init={init_pl} ep={n_ep} warm={warm_ep} rpl_noise={rpl_noise_std} "
            f"head={odhf} tail={odtf} ov_focus={ov_focus_thr} swap_thr={gst} "
            f"β=({b1},{b2}) wd={wd} norm_max={nmax} sched={sched} lr={lr} min_lr={min_lr} "
            f"λ_wl_ov={lambda_wl_in_ov_phase} λ_wl_wl={lambda_wl_in_wl_phase} "
            f"λ_ov_ov={lambda_ov_in_ov_phase} λ_ov_wl={lambda_ov_in_wl_phase} "
            f"dens_bin={d_bin} dens_R={d_kr} ts={d_ts} te={d_te} "
            f"λd={d_ld} mf={d_ldmf} ramp={d_ldrp} p={d_dpe} | "
            f"cells_ov={metrics['num_cells_with_overlaps']}/"
            f"{metrics['total_cells']} "
            f"nwl={metrics['normalized_wl']:.4f}"
            + ("  << best so far" if improved else "")
        )

    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETERS")
    print("=" * 70)
    for k, v in best_hyperparameters.items():
        print(f"  {k}: {v}")
    print("\nBest metrics (test-suite style):")
    print(
        f"  overlap_ratio: {best_metrics['overlap_ratio']:.6f} "
        f"({best_metrics['num_cells_with_overlaps']}/{best_metrics['total_cells']} cells)"
    )
    print(f"  normalized_wl: {best_metrics['normalized_wl']:.6f}")

    return {
        "best_hyperparameters": best_hyperparameters,
        "best_metrics": best_metrics,
        "result": best_result,
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
    positions = cell_features[:, 2:4].detach().cpu().numpy()  # [N, 2]
    widths = cell_features[:, 4].detach().cpu().numpy()  # [N]
    heights = cell_features[:, 5].detach().cpu().numpy()  # [N]
    areas = cell_features[:, 0].detach().cpu().numpy()  # [N]

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
    positions = cell_features[:, 2:4].detach().cpu().numpy()
    widths = cell_features[:, 4].detach().cpu().numpy()
    heights = cell_features[:, 5].detach().cpu().numpy()

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
    titles=None,
):
    """Create side-by-side visualization of initial vs final placement.

    Args:
        initial_cell_features: Initial cell positions and properties
        final_cell_features: Optimized cell positions and properties
        pin_features: Pin information
        edge_list: Edge connectivity
        filename: Output filename for the plot
        titles: Optional ``(left_title, right_title)``. Defaults to generic labels.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        left_title = titles[0] if titles else "Initial Placement"
        right_title = titles[1] if titles else "Final Placement"

        # Plot both initial and final placements
        for ax, cell_features, title in [
            (ax1, initial_cell_features, left_title),
            (ax2, final_cell_features, right_title),
        ]:
            N = cell_features.shape[0]
            positions = cell_features[:, 2:4].detach().cpu().numpy()
            widths = cell_features[:, 4].detach().cpu().numpy()
            heights = cell_features[:, 5].detach().cpu().numpy()

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
    print(f"\nUsing device: {DEVICE}")
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
    dev = cell_features.device
    angles = torch.rand(total_cells, device=dev) * 2 * 3.14159
    radii = torch.rand(total_cells, device=dev) * spread_radius

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

    # Search hyperparameter space with grid search:

    # hyperparameter_search()

    # Analytical tools:

    # cell_features, pin_features, edge_list = generate_placement_input(
    #     num_macros=3, num_std_cells=50
    # )
    # print_adjacency_matrix_and_stats(cell_features, pin_features, edge_list)
    # print(calculate_min_possible_normalized_wl(cell_features, pin_features, edge_list))
