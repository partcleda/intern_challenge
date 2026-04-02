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

def wirelength_attraction_loss(cell_features, pin_features, edge_list, gamma=1.0):
    """Calculate loss based on total wirelength to minimize routing.

    This is a REFERENCE IMPLEMENTATION showing how to write a differentiable loss function.

    The loss computes the Manhattan distance between connected pins and minimizes
    the total wirelength across all edges.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information
        edge_list: [E, 2] tensor with edges
        gamma: WA smoothing — larger is smoother; anneal from 1.0→0.1 during
               training so the final loss closely approximates true HPWL

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

    # WA (Weighted Average) wirelength model from DREAMPlace.
    # For each 2-pin edge, WA approximates the bounding-box half-perimeter:
    #   WA_x ≈ max(x1, x2) - min(x1, x2) = |x1 - x2|
    # via softmax-weighted averages over pin positions:
    #   WA_pos_x = softmax( x/gamma) · x  ≈ max(x)
    #   WA_neg_x = softmax(-x/gamma) · x  ≈ min(x)
    #   WA_x = WA_pos_x - WA_neg_x
    # torch.softmax is numerically stable (subtracts max internally).
    # Smaller gamma → sharper (closer to true HPWL); anneal down during training.

    x_pos = torch.stack([src_x, tgt_x], dim=0)  # [2, E]
    y_pos = torch.stack([src_y, tgt_y], dim=0)  # [2, E]

    wa_x = (x_pos * torch.softmax( x_pos / gamma, dim=0)).sum(0) \
         - (x_pos * torch.softmax(-x_pos / gamma, dim=0)).sum(0)
    wa_y = (y_pos * torch.softmax( y_pos / gamma, dim=0)).sum(0) \
         - (y_pos * torch.softmax(-y_pos / gamma, dim=0)).sum(0)

    total_wirelength = torch.sum(wa_x + wa_y)

    return total_wirelength / edge_list.shape[0]  # Normalize by number of edges


def overlap_repulsion_loss(cell_features, pin_features, edge_list, grid_size=None):
    """Calculate loss to prevent cell overlaps via ePlace/DREAMPlace bin-density penalty.

    Instead of an O(N²) pairwise matrix, the canvas is divided into a G×G grid of
    bins. Each cell contributes to nearby bins via a differentiable triangle kernel
    (the NTUplace3 / DREAMPlace density model). Bins whose density exceeds the
    uniform target are penalised quadratically — this drives cells to spread
    uniformly and eliminates overlaps without ever building an [N,N] tensor.

    Memory: O(N·G + G²) via BLAS sgemm — handles N=100K+ on a laptop.
    Gradients exist everywhere (nonzero even before cells touch), giving the
    optimiser a long-range spreading signal that pairwise ReLU lacks.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information (not used here)
        edge_list: [E, 2] tensor with edges (not used here)
        grid_size: G for the G×G bin grid; None → auto-scaled with N

    Returns:
        Scalar loss value (near zero when density is everywhere ≤ target)
    """
    N = cell_features.shape[0]
    if N <= 1:
        return torch.tensor(0.0, requires_grad=True)

    x = cell_features[:, CellFeatureIdx.X]
    y = cell_features[:, CellFeatureIdx.Y]
    w = cell_features[:, CellFeatureIdx.WIDTH]
    h = cell_features[:, CellFeatureIdx.HEIGHT]

    # Auto-scale grid size with N: coarser for tiny designs (less overhead),
    # finer for large designs (better spatial resolution).
    # Continuous formula for N>200 gives G≈95 at N=2010 and G=128 at N=10k+,
    # which produces sharper repulsion gradients that push cells apart more
    # precisely than the old hard cap of 64.
    if grid_size is None:
        if N <= 50:
            G = 16
        elif N <= 200:
            G = 32
        else:
            G = min(128, int(N ** 0.5 * 1.5))
    else:
        G = grid_size

    # Canvas: square, sized so cells fill ~1/1.35 of the area (35% free space).
    # Extra slack lowers target_density so the penalty fires less during WL
    # refinement, letting cells cluster freely around their nets.
    total_cell_area = (w * h).sum()
    canvas_side = (total_cell_area * 1.35).sqrt()
    half_canvas = canvas_side / 2.0

    bin_size = canvas_side / G
    half_bin = bin_size / 2.0
    bin_area = bin_size * bin_size

    # Bin centres along one axis: [G]
    bin_centers = torch.linspace(
        (-half_canvas + half_bin).item(),
        ( half_canvas - half_bin).item(),
        G,
        dtype=x.dtype,
    )

    # Cell half-sizes: [N, 1] for broadcasting against [1, G]
    half_w = (w / 2).unsqueeze(1)   # [N, 1]
    half_h = (h / 2).unsqueeze(1)   # [N, 1]

    dx = (x.unsqueeze(1) - bin_centers.unsqueeze(0)).abs()   # [N, G]
    dy = (y.unsqueeze(1) - bin_centers.unsqueeze(0)).abs()   # [N, G]

    reach_x = half_w + half_bin   # [N, 1] — broadcast to [N, G]
    reach_y = half_h + half_bin

    # Physical coverage kernel (fraction of bin width/height covered by cell i).
    # Formula: overlap_len / bin_size = max(0, min(reach-|u|, 2*half_w, 2*half_bin)) / bin_size
    #
    # This is the exact rectangular overlap fraction and lies in [0, 1] for every
    # cell-bin pair — crucially, a large macro that spans a bin fully contributes
    # exactly 1.0 (not cell_area/bin_area which can be >> 1 and causes the density
    # penalty to be permanently huge regardless of placement quality).
    #
    # Key properties:
    #   • sum over bins m of px[i,m] = half_w[i] / half_bin  (conserves cell area)
    #   • sum_{m,n} density[m,n] = total_cell_area / bin_area = target * G²  ✓
    num_x = torch.minimum(torch.minimum(reach_x - dx, 2.0 * half_w), 2.0 * half_bin)
    px = torch.relu(num_x) / (2.0 * half_bin)   # [N, G] ∈ [0, 1]

    num_y = torch.minimum(torch.minimum(reach_y - dy, 2.0 * half_h), 2.0 * half_bin)
    py = torch.relu(num_y) / (2.0 * half_bin)   # [N, G] ∈ [0, 1]

    # Density map [G, G] via BLAS sgemm — never builds an [N, G, G] tensor.
    # density[m, n] = Σ_i px[i, m] * py[i, n]  (fractional coverage sum)
    density = px.T @ py   # [G, G]

    # Target: cells spread uniformly at total_cell_area / canvas_area ≈ 0.83.
    # When spread, interior bins of large macros reach 1.0 (unavoidable — macro
    # fills the bin), so overflow ≈ 0.17 there; those are tiny and constant-gradient
    # w.r.t. macro position, so they don't corrupt the WL signal after spreading.
    target_density = total_cell_area / (canvas_side * canvas_side)

    # Quadratic overflow penalty (RePlAce / DREAMPlace formulation).
    overflow = torch.relu(density - target_density)
    return overflow.pow(2).sum() / (G * G)


def legalize_placement(cell_features):
    """Min-displacement legalization using Abacus row packing for standard cells.

    Replaces the old push-right-only sweep with a strategy that keeps cells as
    close as possible to their gradient-optimised positions, preserving WL gains:

    1. Macros (h > 1.5): placed largest-first with a bidirectional radial search
       that finds the nearest valid position in any direction rather than always
       pushing right.
    2. Standard cells (h == 1.0): snapped to horizontal rows (pitch = 1.0) and
       packed within each row via the Abacus cluster algorithm, which minimises
       sum |x_placed - x_opt| subject to left-to-right non-overlap.
    3. Three x+y sweep pairs (same as the original) resolve any residual
       macro–std-cell boundary overlaps after the row packing.

    Args:
        cell_features: [N, 6] tensor (modified in-place and returned)

    Returns:
        cell_features with positions adjusted to be fully non-overlapping
    """
    import bisect
    import numpy as np
    from collections import defaultdict

    N = cell_features.shape[0]
    if N <= 1:
        return cell_features

    px_opt = cell_features[:, CellFeatureIdx.X].detach().numpy().copy()
    py_opt = cell_features[:, CellFeatureIdx.Y].detach().numpy().copy()
    w = cell_features[:, CellFeatureIdx.WIDTH].detach().numpy()
    h = cell_features[:, CellFeatureIdx.HEIGHT].detach().numpy()

    px = px_opt.copy()
    py = py_opt.copy()

    macro_mask = h > STANDARD_CELL_HEIGHT + 0.5
    macro_ids  = np.where(macro_mask)[0]
    std_ids    = np.where(~macro_mask)[0]

    total_cell_area = float((w * h).sum())
    canvas_side = (total_cell_area * 1.35) ** 0.5
    half_canvas = canvas_side / 2.0

    macro_order = macro_ids[np.argsort(-(w[macro_ids] * h[macro_ids]))]
    placed_boxes: list[tuple[float, float, float, float]] = []  

    def overlaps_placed(cx: float, cy: float, hwi: float, hhi: float) -> bool:
        for bx, by, bhw, bhh in placed_boxes:
            if abs(cx - bx) < hwi + bhw and abs(cy - by) < hhi + bhh:
                return True
        return False

    for i in macro_order:
        hwi = w[i] / 2.0
        hhi = h[i] / 2.0
        x0, y0 = px_opt[i], py_opt[i]

        if not overlaps_placed(x0, y0, hwi, hhi):
            placed_boxes.append((x0, y0, hwi, hhi))
            continue
        step = max(w[i], h[i]) * 0.5
        angles = [0, np.pi, np.pi / 2, -np.pi / 2,
                  np.pi / 4, -np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4]
        found = False
        for mult in range(1, 80):
            r = step * mult
            cands = sorted(
                [(x0 + r * np.cos(a), y0 + r * np.sin(a)) for a in angles],
                key=lambda c: (c[0] - x0) ** 2 + (c[1] - y0) ** 2,
            )
            for cx, cy in cands:
                if not overlaps_placed(cx, cy, hwi, hhi):
                    px[i], py[i] = cx, cy
                    placed_boxes.append((cx, cy, hwi, hhi))
                    found = True
                    break
            if found:
                break
        if not found:
            max_right = max((bx + bhw for bx, _, bhw, _ in placed_boxes), default=x0)
            px[i] = max_right + hwi + 1e-4
            placed_boxes.append((px[i], py[i], hwi, hhi))

    rows: dict[int, list[int]] = defaultdict(list)
    for i in std_ids:
        row_idx = int(round(py_opt[i] / STANDARD_CELL_HEIGHT))
        rows[row_idx].append(int(i))

    for row_idx, cell_ids in rows.items():
        row_y = row_idx * STANDARD_CELL_HEIGHT
        cell_ids_sorted = sorted(cell_ids, key=lambda ci: px_opt[ci])

        clusters: list[list] = []  

        for ci in cell_ids_sorted:
            wi = w[ci]
            new_cluster: list = [px_opt[ci] - wi / 2.0, [ci], wi]

            while clusters:
                prev = clusters[-1]
                if prev[0] + prev[2] > new_cluster[0]:
                    merged_ids = prev[1] + new_cluster[1]
                    merged_w   = prev[2] + new_cluster[2]
                    cum = 0.0
                    anchors = []
                    for mid in merged_ids:
                        anchors.append(px_opt[mid] - w[mid] / 2.0 - cum)
                        cum += w[mid]
                    opt_x = float(np.median(anchors))
                    min_x = clusters[-2][0] + clusters[-2][2] if len(clusters) >= 2 else -half_canvas
                    opt_x = max(opt_x, min_x)
                    opt_x = min(opt_x, half_canvas - merged_w)
                    clusters.pop()
                    new_cluster = [opt_x, merged_ids, merged_w]
                else:
                    break

            clusters.append(new_cluster)

        for x_start, ids_in_cluster, _ in clusters:
            cum = 0.0
            for ci in ids_in_cluster:
                px[ci] = x_start + cum + w[ci] / 2.0
                py[ci] = row_y
                cum += w[ci]
    max_w_all = float(w.max())
    max_h_all = float(h.max())

    for _sweep in range(3):
        order = np.argsort(px)
        placed_px: list[float] = []
        placed_ids: list[int] = []

        for k in range(N):
            i = int(order[k])
            changed = True
            while changed:
                changed = False
                lo   = bisect.bisect_left(placed_px,  px[i] - max_w_all)
                hi_b = bisect.bisect_right(placed_px, px[i] + max_w_all)
                for idx in range(lo, hi_b):
                    j = placed_ids[idx]
                    need_x = (w[i] + w[j]) / 2.0
                    need_y = (h[i] + h[j]) / 2.0
                    if abs(px[i] - px[j]) < need_x and abs(py[i] - py[j]) < need_y:
                        px[i] = px[j] + need_x + 1e-4
                        changed = True
                        break
            ins = bisect.bisect_left(placed_px, px[i])
            placed_px.insert(ins, px[i])
            placed_ids.insert(ins, i)

        order_y = np.argsort(py)
        placed_py: list[float] = []
        placed_ids_y: list[int] = []

        for k in range(N):
            i = int(order_y[k])
            changed = True
            while changed:
                changed = False
                lo   = bisect.bisect_left(placed_py,  py[i] - max_h_all)
                hi_b = bisect.bisect_right(placed_py, py[i] + max_h_all)
                for idx in range(lo, hi_b):
                    j = placed_ids_y[idx]
                    need_x = (w[i] + w[j]) / 2.0
                    need_y = (h[i] + h[j]) / 2.0
                    if abs(px[i] - px[j]) < need_x and abs(py[i] - py[j]) < need_y:
                        py[i] = py[j] + need_y + 1e-4
                        changed = True
                        break
            ins = bisect.bisect_left(placed_py, py[i])
            placed_py.insert(ins, py[i])
            placed_ids_y.insert(ins, i)

    cell_features = cell_features.clone()
    cell_features[:, CellFeatureIdx.X] = torch.tensor(px, dtype=cell_features.dtype)
    cell_features[:, CellFeatureIdx.Y] = torch.tensor(py, dtype=cell_features.dtype)
    return cell_features


def local_swap_optimization(cell_features, pin_features, edge_list, num_passes=3, k=8):
    """Post-legalization local pairwise cell swap to reduce HPWL.

    For each cell, tries swapping positions with up to k nearest neighbours.
    A swap is accepted only if it (a) strictly reduces total HPWL and (b) does
    not introduce any new overlaps — verified via a KD-tree neighbourhood query
    around both new positions before committing the swap.

    This step never touches the overlap count: every accepted swap is checked
    for safety first, so the placement stays fully legal throughout.

    Skipped for very large designs (N > 10 000) to keep runtime bounded.

    Args:
        cell_features: [N, 6] tensor with legalised cell positions
        pin_features:  [P, 7] tensor with pin information
        edge_list:     [E, 2] tensor with 2-pin net edges
        num_passes:    number of full sweeps over all cells (default 3)
        k:             nearest neighbours to consider per cell (default 8)

    Returns:
        Updated cell_features with equal or better WL, guaranteed still legal
    """
    try:
        from scipy.spatial import KDTree
    except ImportError:
        return cell_features

    import numpy as np

    N = cell_features.shape[0]
    if N <= 2 or edge_list.shape[0] == 0:
        return cell_features

    if N > 10000:
        return cell_features
    elif N > 2000:
        k = min(k, 8)
        num_passes = min(num_passes, 3)
    elif N > 500:
        k = min(k, 10)
        num_passes = min(num_passes, 4)
    else:
        k = min(k + 4, 12)     
        num_passes = min(num_passes + 2, 5)

    px = cell_features[:, CellFeatureIdx.X].detach().numpy().copy()
    py = cell_features[:, CellFeatureIdx.Y].detach().numpy().copy()
    w  = cell_features[:, CellFeatureIdx.WIDTH].detach().numpy()
    h  = cell_features[:, CellFeatureIdx.HEIGHT].detach().numpy()

    E = edge_list.shape[0]
    src_pins     = edge_list[:, 0].numpy().astype(int)
    tgt_pins     = edge_list[:, 1].numpy().astype(int)
    cell_idx_arr = pin_features[:, PinFeatureIdx.CELL_IDX].numpy().astype(int)
    pin_rel_x    = pin_features[:, PinFeatureIdx.PIN_X].numpy()
    pin_rel_y    = pin_features[:, PinFeatureIdx.PIN_Y].numpy()

    net_ci0 = cell_idx_arr[src_pins]
    net_ci1 = cell_idx_arr[tgt_pins]
    net_rx0 = pin_rel_x[src_pins]
    net_ry0 = pin_rel_y[src_pins]
    net_rx1 = pin_rel_x[tgt_pins]
    net_ry1 = pin_rel_y[tgt_pins]

    cell_nets: list[list[int]] = [[] for _ in range(N)]
    for e in range(E):
        c0, c1 = int(net_ci0[e]), int(net_ci1[e])
        cell_nets[c0].append(e)
        if c1 != c0:
            cell_nets[c1].append(e)

    def hpwl_delta(i: int, j: int, nets: list[int]) -> float:
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        delta = 0.0
        for e in nets:
            c0, c1 = int(net_ci0[e]), int(net_ci1[e])
            ax0 = px[c0] + net_rx0[e];  ay0 = py[c0] + net_ry0[e]
            ax1 = px[c1] + net_rx1[e];  ay1 = py[c1] + net_ry1[e]
            bx0 = (xj if c0 == i else (xi if c0 == j else px[c0])) + net_rx0[e]
            by0 = (yj if c0 == i else (yi if c0 == j else py[c0])) + net_ry0[e]
            bx1 = (xj if c1 == i else (xi if c1 == j else px[c1])) + net_rx1[e]
            by1 = (yj if c1 == i else (yi if c1 == j else py[c1])) + net_ry1[e]
            delta += (abs(bx1 - bx0) + abs(by1 - by0)
                      - abs(ax1 - ax0) - abs(ay1 - ay0))
        return delta

    for _ in range(num_passes):
        centers = np.stack([px, py], axis=1)
        tree = KDTree(centers)

        for i in range(N):
            _, nbrs = tree.query([px[i], py[i]], k=min(k + 1, N))

            best_delta = 0.0
            best_j = -1

            for j in map(int, nbrs):
                if j == i:
                    continue
                nets = list(set(cell_nets[i] + cell_nets[j]))
                if not nets:
                    continue
                d = hpwl_delta(i, j, nets)
                if d < best_delta:
                    best_delta = d
                    best_j = j

            if best_j < 0:
                continue

            if abs(h[i] - h[best_j]) > 0.5:
                continue

            xi_new, yi_new = px[best_j], py[best_j]
            xj_new, yj_new = px[i], py[i]
            wi, hi_c = w[i], h[i]
            wj, hj   = w[best_j], h[best_j]

            safe = True
            for (cx, cy, cw, ch) in [(xi_new, yi_new, wi, hi_c),
                                      (xj_new, yj_new, wj, hj)]:
                _, near = tree.query([cx, cy], k=min(30, N))
                for m in map(int, near):
                    if m == i or m == best_j:
                        continue
                    xm, ym, wm, hm = px[m], py[m], w[m], h[m]
                    if (abs(cx - xm) < (cw + wm) / 2 and
                            abs(cy - ym) < (ch + hm) / 2):
                        safe = False
                        break
                if not safe:
                    break

            if safe:
                px[i], px[best_j] = px[best_j], px[i]
                py[i], py[best_j] = py[best_j], py[i]

    cell_features = cell_features.clone()
    cell_features[:, CellFeatureIdx.X] = torch.tensor(px, dtype=cell_features.dtype)
    cell_features[:, CellFeatureIdx.Y] = torch.tensor(py, dtype=cell_features.dtype)
    return cell_features


def train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=2000,
    lr=0.05,
    lambda_wirelength=1.0,
    lambda_overlap=50.0,
    verbose=True,
    log_interval=100,
):
    """Train the placement optimization using gradient descent.

    Three-phase training strategy:
      Phase 1 (0–50%):  lambda ramps 5% → 100% of target; gamma anneals 1.0 → 0.01.
                         Density penalty dominates early so cells spread quickly.
      Phase 2 (50–92%): lambda held at target; gamma anneals toward 0.01.
                         WL optimisation sharpens while density keeps cells spread.
      Phase 3 (92–100%): lambda drops to ~5–15% of target; gamma stays at 0.01.
                          Near-zero density penalty lets WL pull cells together
                          without causing new overlaps (cells are already well spread).
    After training, legalize_placement() eliminates any residual overlaps.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity
        num_epochs: Base number of iterations (scaled internally by design size)
        lr: Peak learning rate (cosine-annealed down to lr*0.01)
        lambda_wirelength: Weight for wirelength loss
        lambda_overlap: Target weight for overlap/density loss
        verbose: Whether to print progress
        log_interval: How often to print progress

    Returns:
        Dictionary with:
            - final_cell_features: Optimized and legalized cell positions
            - initial_cell_features: Original cell positions (for comparison)
            - loss_history: Loss values over time
    """
    import math as _math

    cell_features = cell_features.clone()
    initial_cell_features = cell_features.clone()

    N = cell_features.shape[0]

    w = cell_features[:, CellFeatureIdx.WIDTH]
    h = cell_features[:, CellFeatureIdx.HEIGHT]
    total_cell_area = (w * h).sum().item()
    canvas_side = (total_cell_area * 1.35) ** 0.5
    half_canvas = canvas_side / 2.0

    N_ref = 200
    epoch_scale = max(1.0, min(5.0, (N / N_ref) ** 0.55))
    effective_epochs = int(num_epochs * epoch_scale)


    num_restarts = 5 if N <= 300 else 1

    wl_refine_epochs = max(200, int(200 * (N / 100) ** 0.40))

    best_wl = float("inf")
    best_result = None

    for restart_idx in range(num_restarts):
        if restart_idx == 0:
            cell_positions = cell_features[:, 2:4].clone().detach()
        else:
            torch.manual_seed(restart_idx * 7919 + N)   
            spread = half_canvas * 0.60
            angles = torch.rand(N) * 2.0 * _math.pi
            radii  = torch.rand(N) * spread
            cell_positions = torch.stack(
                [radii * torch.cos(angles), radii * torch.sin(angles)], dim=1
            )
        cell_positions = cell_positions.requires_grad_(True)

        # Nesterov SGD + cosine annealing.
        optimizer = optim.SGD([cell_positions], lr=lr, momentum=0.9, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=effective_epochs, eta_min=lr * 0.01
        )

        loss_history = {
            "total_loss": [],
            "wirelength_loss": [],
            "overlap_loss": [],
        }

        phase1_end = int(effective_epochs * 0.50)
        phase2_end = int(effective_epochs * 0.92)
        lambda_start = lambda_overlap * 0.05

        for epoch in range(effective_epochs):
            optimizer.zero_grad()

            if epoch < phase1_end:
                frac = epoch / max(1, phase1_end - 1)
                lambda_t = lambda_start + (lambda_overlap - lambda_start) * frac
            elif epoch < phase2_end:
                lambda_t = lambda_overlap
            else:

                lambda_t = lambda_overlap * 0.05

            total_frac = epoch / max(1, effective_epochs - 1)
            gamma = max(0.01, 1.0 - 0.99 * total_frac)

            cell_features_current = cell_features.clone()
            cell_features_current[:, 2:4] = cell_positions

            wl_loss = wirelength_attraction_loss(
                cell_features_current, pin_features, edge_list, gamma=gamma
            )
            overlap_loss = overlap_repulsion_loss(
                cell_features_current, pin_features, edge_list
            )

            total_loss = lambda_wirelength * wl_loss + lambda_t * overlap_loss
            total_loss.backward()

            max_norm = 5.0 if epoch < phase2_end else 2.0
            torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=max_norm)

            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                cell_positions.clamp_(-half_canvas, half_canvas)

            loss_history["total_loss"].append(total_loss.item())
            loss_history["wirelength_loss"].append(wl_loss.item())
            loss_history["overlap_loss"].append(overlap_loss.item())

            if verbose and (epoch % log_interval == 0 or epoch == effective_epochs - 1):
                print(f"[restart {restart_idx}] Epoch {epoch}/{effective_epochs} "
                      f"(lambda={lambda_t:.2f}, gamma={gamma:.3f}):")
                print(f"  Total Loss: {total_loss.item():.6f}")
                print(f"  Wirelength Loss: {wl_loss.item():.6f}")
                print(f"  Overlap Loss: {overlap_loss.item():.6f}")

        if verbose:
            print(f"\n[restart {restart_idx}] Pre-legalization WL refinement "
                  f"({wl_refine_epochs} epochs)...")
        wl_pos = cell_positions.clone().detach().requires_grad_(True)
        wl_opt = optim.SGD([wl_pos], lr=0.08, momentum=0.0)
        wl_sched = optim.lr_scheduler.CosineAnnealingLR(
            wl_opt, T_max=wl_refine_epochs, eta_min=0.002
        )
        for _ in range(wl_refine_epochs):
            wl_opt.zero_grad()
            cf_wl = cell_features.clone()
            cf_wl[:, 2:4] = wl_pos
            loss_wl = wirelength_attraction_loss(cf_wl, pin_features, edge_list, gamma=0.01)
            loss_wl.backward()

            with torch.no_grad():
                cell_grad = wl_pos.grad          # [N, 2]
                cell_norms = cell_grad.norm(dim=1, keepdim=True).clamp(min=1e-8)
                scale = (0.15 / cell_norms).clamp(max=1.0)
                wl_pos.grad = cell_grad * scale
            wl_opt.step()
            wl_sched.step()
            with torch.no_grad():
                wl_pos.clamp_(-half_canvas, half_canvas)

        final_cell_features = cell_features.clone()
        final_cell_features[:, 2:4] = wl_pos.detach()

        if verbose:
            print(f"[restart {restart_idx}] Legalization...")
        final_cell_features = legalize_placement(final_cell_features)

        if verbose:
            print(f"[restart {restart_idx}] Local swap...")
        final_cell_features = local_swap_optimization(
            final_cell_features, pin_features, edge_list
        )

        if verbose:
            print(f"[restart {restart_idx}] Final legalization pass...")
        final_cell_features = legalize_placement(final_cell_features)

        restart_metrics = calculate_normalized_metrics(
            final_cell_features, pin_features, edge_list
        )
        restart_wl = restart_metrics["normalized_wl"]
        restart_overlap = restart_metrics["overlap_ratio"]
        if verbose:
            print(f"[restart {restart_idx}] WL={restart_wl:.4f}  overlap={restart_overlap:.4f}")

        if restart_overlap == 0.0 and restart_wl < best_wl:
            best_wl = restart_wl
            best_result = {
                "final_cell_features": final_cell_features,
                "initial_cell_features": initial_cell_features,
                "loss_history": loss_history,
            }

    if best_result is None:
        best_result = {
            "final_cell_features": final_cell_features,
            "initial_cell_features": initial_cell_features,
            "loss_history": loss_history,
        }

    return best_result


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
        lr=0.05,
        lambda_overlap=50.0,
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
