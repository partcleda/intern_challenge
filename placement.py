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
from collections import defaultdict
from enum import IntEnum

import numpy as np
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
    cell_features[:, CellFeatureIdx.WIDTH] = cell_widths
    cell_features[:, CellFeatureIdx.HEIGHT] = cell_heights

    # Step 6: Generate pins for each cell
    total_pins = num_pins_per_cell.sum().item()
    pin_features = torch.zeros(total_pins, 7)

    # Fixed pin size for all pins (square pins)
    PIN_SIZE = 0.1  # All pins are 0.1 x 0.1

    pin_cell_idx = torch.repeat_interleave(
        torch.arange(total_cells), num_pins_per_cell
    )
    pin_cell_w = cell_widths[pin_cell_idx]
    pin_cell_h = cell_heights[pin_cell_idx]
    margin = PIN_SIZE / 2
    raw_rx = torch.rand(total_pins)
    raw_ry = torch.rand(total_pins)
    usable_w = torch.clamp(pin_cell_w - 2 * margin, min=0.0)
    usable_h = torch.clamp(pin_cell_h - 2 * margin, min=0.0)
    has_space = (usable_w > 0) & (usable_h > 0)
    pin_x = torch.where(has_space, margin + raw_rx * usable_w, pin_cell_w / 2)
    pin_y = torch.where(has_space, margin + raw_ry * usable_h, pin_cell_h / 2)

    # Fill pin features
    pin_features[:, PinFeatureIdx.CELL_IDX] = pin_cell_idx.float()
    pin_features[:, PinFeatureIdx.PIN_X] = pin_x
    pin_features[:, PinFeatureIdx.PIN_Y] = pin_y
    pin_features[:, PinFeatureIdx.X] = pin_x
    pin_features[:, PinFeatureIdx.Y] = pin_y
    pin_features[:, PinFeatureIdx.WIDTH] = PIN_SIZE
    pin_features[:, PinFeatureIdx.HEIGHT] = PIN_SIZE

    # Step 7: Generate edges with simple random connectivity
    num_conn_per_pin = torch.randint(1, 4, (total_pins,))
    total_candidates = num_conn_per_pin.sum().item()
    src_pins = torch.repeat_interleave(
        torch.arange(total_pins), num_conn_per_pin
    )
    tgt_pins = torch.randint(0, total_pins, (total_candidates,))
    valid = src_pins != tgt_pins
    src_pins = src_pins[valid]
    tgt_pins = tgt_pins[valid]
    lo = torch.min(src_pins, tgt_pins)
    hi = torch.max(src_pins, tgt_pins)
    edge_hash = lo.long() * total_pins + hi.long()
    edge_hash = torch.unique(edge_hash)
    edge_list = torch.stack(
        [edge_hash // total_pins, edge_hash % total_pins], dim=1
    )

    print(f"\nGenerated placement data:")
    print(f"  Total cells: {total_cells}")
    print(f"  Total pins: {total_pins}")
    print(f"  Total edges: {len(edge_list)}")
    print(f"  Average edges per pin: {2 * len(edge_list) / total_pins:.2f}")

    return cell_features, pin_features, edge_list

# ======= OPTIMIZATION CODE (edit this part) =======

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

    # Calculate smooth approximation of Manhattan distance
    # Using log-sum-exp approximation for differentiability
    alpha = 0.1  # Smoothing parameter
    dx = torch.abs(pin_absolute_x[src_pins] - pin_absolute_x[tgt_pins])
    dy = torch.abs(pin_absolute_y[src_pins] - pin_absolute_y[tgt_pins])

    # Smooth L1 distance with numerical stability
    smooth_manhattan = alpha * torch.logsumexp(
        torch.stack([dx / alpha, dy / alpha], dim=0), dim=0
    )

    # Total wirelength
    return torch.sum(smooth_manhattan) / edge_list.shape[0]  # Normalize by number of edges


def _analytical_place(cell_features, pin_features, edge_list, iters=120):
    """Spectral initial placement (Gordian/Kraftwerk2-style).

    Uses iterative weighted averaging on the connectivity graph
    to find an initial analytical placement.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity
        iters: Number of averaging iterations

    Returns:
        [N, 2] tensor with initial cell positions
    """
    N = cell_features.shape[0]
    if edge_list.shape[0] == 0:
        return torch.zeros(N, 2)

    cell_idx = pin_features[:, 0].long()
    src_cells = cell_idx[edge_list[:, 0].long()]
    tgt_cells = cell_idx[edge_list[:, 1].long()]
    valid = src_cells != tgt_cells
    src_cells = src_cells[valid]
    tgt_cells = tgt_cells[valid]

    row = torch.cat([src_cells, tgt_cells])
    col = torch.cat([tgt_cells, src_cells])
    vals = torch.ones(row.shape[0], dtype=torch.float32)
    adj = torch.sparse_coo_tensor(
        torch.stack([row, col]), vals, (N, N)
    ).coalesce()

    degree = torch.clamp(torch.sparse.sum(adj, dim=1).to_dense(), min=1.0)
    inv_deg = 1.0 / degree
    total_area = cell_features[:, 0].sum().item()
    spread = (total_area ** 0.5) * 0.1

    pos = torch.zeros(N, 2)
    pos[:, 0] = torch.linspace(-spread, spread, N)
    pos[:, 1] = torch.randn(N) * spread * 0.1

    for _ in range(iters):
        new_pos = torch.sparse.mm(adj, pos) * inv_deg.unsqueeze(1)
        pos = 0.8 * new_pos + 0.2 * pos

    return pos


def _legalize_from_analytical(
    cell_features, analytical_pos, pin_features=None, edge_list=None
):
    """Row-based legalization (Tetris/Abacus-style).

    Places macros first by area (largest first), then standard cells
    in rows guided by analytical positions and macro connectivity.

    Args:
        cell_features: [N, 6] tensor with cell properties
        analytical_pos: [N, 2] tensor with analytical positions
        pin_features: Optional pin features for connectivity-aware placement
        edge_list: Optional edge list for connectivity-aware placement

    Returns:
        [N, 2] tensor with legalized cell positions
    """
    N = cell_features.shape[0]
    widths = cell_features[:, 4].numpy().astype(np.float64)
    heights = cell_features[:, 5].numpy().astype(np.float64)
    areas = cell_features[:, 0].numpy().astype(np.float64)
    hh = heights / 2.0
    total_area = float(areas.sum())
    target_width = (total_area * 1.5) ** 0.5
    margin = 0.05

    is_macro = heights > 1.5
    macro_idx = np.where(is_macro)[0]
    std_idx = np.where(~is_macro)[0]
    Nm, Ns = len(macro_idx), len(std_idx)
    positions = np.zeros((N, 2))

    apos = (
        analytical_pos.numpy().astype(np.float64)
        if isinstance(analytical_pos, torch.Tensor)
        else analytical_pos.astype(np.float64)
    )

    # Place macros by area (largest first)
    if Nm > 0:
        macro_order = macro_idx[np.argsort(areas[macro_idx])[::-1]]
        x_pos, y_pos, row_height = 0.0, 0.0, 0.0
        for mi in macro_order:
            w, h = widths[mi], heights[mi]
            if x_pos + w > target_width and x_pos > 0:
                y_pos += row_height + margin
                x_pos = 0.0
                row_height = 0.0
            positions[mi, 0] = x_pos + w / 2
            positions[mi, 1] = y_pos + h / 2
            x_pos += w + margin
            row_height = max(row_height, h)

    # Place standard cells in rows
    if Ns > 0:
        ideal_pos = apos[std_idx].copy()

        # Adjust positions based on macro connectivity
        if (
            pin_features is not None
            and edge_list is not None
            and edge_list.shape[0] > 0
        ):
            cidx = pin_features[:, 0].long().numpy()
            src_cells = cidx[edge_list[:, 0].long().numpy()]
            tgt_cells = cidx[edge_list[:, 1].long().numpy()]
            valid = src_cells != tgt_cells
            sc, tc = src_cells[valid], tgt_cells[valid]

            macro_target_sum = np.zeros((N, 2))
            macro_target_cnt = np.zeros(N)

            mask1 = is_macro[sc] & ~is_macro[tc]
            np.add.at(macro_target_sum, tc[mask1], positions[sc[mask1]])
            np.add.at(macro_target_cnt, tc[mask1], 1.0)

            mask2 = is_macro[tc] & ~is_macro[sc]
            np.add.at(macro_target_sum, sc[mask2], positions[tc[mask2]])
            np.add.at(macro_target_cnt, sc[mask2], 1.0)

            for si, ci in enumerate(std_idx):
                if macro_target_cnt[ci] > 0:
                    macro_center = macro_target_sum[ci] / macro_target_cnt[ci]
                    ideal_pos[si] = 0.7 * macro_center + 0.3 * apos[ci]

        row_id = (ideal_pos[:, 1] / 0.8).astype(np.int64)
        sort_key = row_id.astype(np.float64) * 1e6 + ideal_pos[:, 0]
        std_order = np.argsort(sort_key)

        std_y_start = (
            (max(positions[mi, 1] + hh[mi] for mi in macro_idx) + margin)
            if Nm > 0
            else 0.0
        )
        x_pos, y_pos, row_height = 0.0, std_y_start, 0.0

        for rank in std_order:
            ci = std_idx[rank]
            w, h = widths[ci], heights[ci]
            if x_pos + w > target_width and x_pos > 0:
                y_pos += row_height + margin
                x_pos = 0.0
                row_height = 0.0
            positions[ci, 0] = x_pos + w / 2
            positions[ci, 1] = y_pos + h / 2
            x_pos += w + margin
            row_height = max(row_height, h)

    return torch.from_numpy(positions).float()


def _push_overlapping(
    positions, widths, heights, idx_a, idx_b, displacements, areas=None
):
    """Push apart overlapping cell pairs.

    Computes overlap amounts and applies displacement forces
    proportional to overlap, weighted by cell area.

    Args:
        positions: [N, 2] array of cell positions
        widths: [N] array of cell widths
        heights: [N] array of cell heights
        idx_a: Array of first cell indices in pairs
        idx_b: Array of second cell indices in pairs
        displacements: [N, 2] array to accumulate displacement vectors
        areas: Optional [N] array of cell areas for weighted pushing

    Returns:
        True if any overlaps were found and resolved
    """
    dx = positions[idx_a, 0] - positions[idx_b, 0]
    dy = positions[idx_a, 1] - positions[idx_b, 1]
    adx, ady = np.abs(dx), np.abs(dy)
    min_sep_x = (widths[idx_a] + widths[idx_b]) / 2
    min_sep_y = (heights[idx_a] + heights[idx_b]) / 2
    overlap_x, overlap_y = min_sep_x - adx, min_sep_y - ady
    overlapping = (overlap_x > 0) & (overlap_y > 0)

    if not overlapping.any():
        return False

    ov_x, ov_y = overlap_x[overlapping], overlap_y[overlapping]
    ia, ib = idx_a[overlapping], idx_b[overlapping]
    d_x, d_y = dx[overlapping], dy[overlapping]

    if areas is not None:
        total = areas[ia] + areas[ib]
        frac_a = np.clip(areas[ib] / total, 0.2, 0.8)
        frac_b = 1.0 - frac_a
    else:
        frac_a = frac_b = np.full(len(ia), 0.5)

    px_mask = (ov_x <= ov_y).astype(np.float64)
    py_mask = 1.0 - px_mask

    xs = np.sign(d_x)
    xs[xs == 0] = 1.0
    np.add.at(displacements[:, 0], ia, (ov_x + 0.02) * px_mask * frac_a * xs)
    np.add.at(displacements[:, 0], ib, -(ov_x + 0.02) * px_mask * frac_b * xs)

    ys = np.sign(d_y)
    ys[ys == 0] = 1.0
    np.add.at(displacements[:, 1], ia, (ov_y + 0.02) * py_mask * frac_a * ys)
    np.add.at(displacements[:, 1], ib, -(ov_y + 0.02) * py_mask * frac_b * ys)

    return True


def _resolve_overlaps(cell_features, max_iters=300):
    """Iterative overlap resolution via displacement forces.

    Repeatedly detects overlapping cell pairs and pushes them apart
    using sweep-based pair detection for efficiency.

    Args:
        cell_features: [N, 6] tensor with cell properties
        max_iters: Maximum number of resolution iterations

    Returns:
        Updated cell_features tensor with resolved positions
    """
    N = cell_features.shape[0]
    positions = cell_features[:, 2:4].detach().clone().numpy().astype(np.float64)
    widths = cell_features[:, 4].detach().numpy().astype(np.float64)
    heights = cell_features[:, 5].detach().numpy().astype(np.float64)
    cell_areas = widths * heights

    is_macro = heights > 1.5
    macro_idx, std_idx = np.where(is_macro)[0], np.where(~is_macro)[0]
    Nm, Ns = len(macro_idx), len(std_idx)
    max_std_w = widths[std_idx].max() if Ns > 0 else 0.0

    for iteration in range(max_iters):
        any_overlap = False
        displacements = np.zeros_like(positions)

        # Macro-macro overlaps
        if Nm > 1:
            for ii in range(Nm):
                for jj in range(ii + 1, Nm):
                    i, j = macro_idx[ii], macro_idx[jj]
                    hit = _push_overlapping(
                        positions, widths, heights,
                        np.array([i]), np.array([j]),
                        displacements, areas=cell_areas,
                    )
                    any_overlap = any_overlap or hit

        # Macro-standard cell overlaps
        if Nm > 0 and Ns > 0:
            for mi in macro_idx:
                dx = np.abs(positions[std_idx, 0] - positions[mi, 0])
                dy = np.abs(positions[std_idx, 1] - positions[mi, 1])
                possible = (
                    (dx < (widths[mi] + widths[std_idx]) / 2)
                    & (dy < (heights[mi] + heights[std_idx]) / 2)
                )
                if possible.any():
                    nearby = std_idx[possible]
                    hit = _push_overlapping(
                        positions, widths, heights,
                        np.full(len(nearby), mi, dtype=np.intp), nearby,
                        displacements, areas=cell_areas,
                    )
                    any_overlap = any_overlap or hit

        # Standard cell-standard cell overlaps (sweep-based)
        if Ns > 1:
            dim = iteration % 2
            order = np.argsort(positions[std_idx, dim])
            sg = std_idx[order]
            sp, sw, sh = positions[sg], widths[sg], heights[sg]

            for k in range(1, min(200, Ns)):
                n = Ns - k
                gap = sp[k:, dim] - sp[:n, dim]
                if gap.min() > max_std_w:
                    break
                adx = np.abs(sp[k:, 0] - sp[:n, 0])
                ady = np.abs(sp[k:, 1] - sp[:n, 1])
                ov_x = (sw[:n] + sw[k:]) / 2 - adx
                ov_y = (sh[:n] + sh[k:]) / 2 - ady
                hit = (ov_x > 0) & (ov_y > 0)

                if not hit.any():
                    continue

                any_overlap = True
                ox, oy = ov_x[hit], ov_y[hit]
                ia, ib = sg[:n][hit], sg[k:][hit]
                px_mask = (ox <= oy).astype(np.float64)
                py_mask = 1.0 - px_mask

                d_x = sp[:n, 0][hit] - sp[k:, 0][hit]
                xs = np.sign(d_x)
                xs[xs == 0] = 1.0
                np.add.at(
                    displacements[:, 0], ia,
                    (ox + 0.02) * px_mask * 0.5 * xs,
                )
                np.add.at(
                    displacements[:, 0], ib,
                    -(ox + 0.02) * px_mask * 0.5 * xs,
                )

                d_y = sp[:n, 1][hit] - sp[k:, 1][hit]
                ys = np.sign(d_y)
                ys[ys == 0] = 1.0
                np.add.at(
                    displacements[:, 1], ia,
                    (oy + 0.02) * py_mask * 0.5 * ys,
                )
                np.add.at(
                    displacements[:, 1], ib,
                    -(oy + 0.02) * py_mask * 0.5 * ys,
                )

        if not any_overlap:
            break
        positions += displacements

    result = cell_features.clone()
    result[:, 2:4] = torch.from_numpy(positions).float()
    return result


class _SpatialGrid:
    """Grid-based spatial index for O(1) amortized overlap queries.

    Divides the placement area into grid cells and maintains
    cell-to-bucket mappings for efficient neighbor lookups.
    """
    __slots__ = ('grid', 'cell_keys', 'gs', 'macro_list')

    def __init__(self, pos, is_macro, grid_size):
        self.gs = grid_size
        self.grid = defaultdict(list)
        self.cell_keys = {}
        self.macro_list = list(np.where(is_macro)[0])
        for i in range(len(pos)):
            key = (int(pos[i, 0] // grid_size), int(pos[i, 1] // grid_size))
            self.cell_keys[i] = key
            self.grid[key].append(i)

    def update(self, i, old_x, old_y, new_x, new_y):
        gs = self.gs
        old_key = (int(old_x // gs), int(old_y // gs))
        new_key = (int(new_x // gs), int(new_y // gs))
        if old_key != new_key:
            try:
                self.grid[old_key].remove(i)
            except ValueError:
                pass
            if not self.grid[old_key]:
                del self.grid[old_key]
            self.cell_keys[i] = new_key
            self.grid[new_key].append(i)

    def check_overlap(self, i, nx, ny, pos, hw, hh, sr):
        gs = self.gs
        gx, gy = int(nx // gs), int(ny // gs)
        hwi, hhi = hw[i], hh[i]
        for mi in self.macro_list:
            if (
                mi != i
                and abs(nx - pos[mi, 0]) < hwi + hw[mi]
                and abs(ny - pos[mi, 1]) < hhi + hh[mi]
            ):
                return True
        for dx in range(-sr, sr + 1):
            for dy in range(-sr, sr + 1):
                bucket = self.grid.get((gx + dx, gy + dy))
                if bucket is None:
                    continue
                for j in bucket:
                    if (
                        j != i
                        and abs(nx - pos[j, 0]) < hwi + hw[j]
                        and abs(ny - pos[j, 1]) < hhi + hh[j]
                    ):
                        return True
        return False

    def check_overlap_skip(self, i, nx, ny, pos, hw, hh, sr, skip):
        gs = self.gs
        gx, gy = int(nx // gs), int(ny // gs)
        hwi, hhi = hw[i], hh[i]
        for mi in self.macro_list:
            if (
                mi != i
                and mi != skip
                and abs(nx - pos[mi, 0]) < hwi + hw[mi]
                and abs(ny - pos[mi, 1]) < hhi + hh[mi]
            ):
                return True
        for dx in range(-sr, sr + 1):
            for dy in range(-sr, sr + 1):
                bucket = self.grid.get((gx + dx, gy + dy))
                if bucket is None:
                    continue
                for j in bucket:
                    if (
                        j != i
                        and j != skip
                        and abs(nx - pos[j, 0]) < hwi + hw[j]
                        and abs(ny - pos[j, 1]) < hhi + hh[j]
                    ):
                        return True
        return False


def _swap_refine(cell_features, pin_features, edge_list, max_passes=3):
    """Detailed placement: pairwise cell swapping (FastDP-style).

    Tries swapping pairs of cells and keeps swaps that reduce wirelength
    without introducing overlaps.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity
        max_passes: Maximum number of swap passes

    Returns:
        Updated cell_features tensor with refined positions
    """
    N = cell_features.shape[0]
    if N > 2500 or edge_list.shape[0] == 0:
        return cell_features

    pos = cell_features[:, 2:4].detach().clone().numpy().astype(np.float64)
    w = cell_features[:, 4].detach().numpy().astype(np.float64)
    h = cell_features[:, 5].detach().numpy().astype(np.float64)
    hw, hh = w / 2.0, h / 2.0

    cidx = pin_features[:, 0].long().numpy()
    prx = pin_features[:, 1].detach().numpy().astype(np.float64)
    pry = pin_features[:, 2].detach().numpy().astype(np.float64)
    src = edge_list[:, 0].long().numpy()
    tgt = edge_list[:, 1].long().numpy()
    E = len(src)

    px, py = pos[cidx, 0] + prx, pos[cidx, 1] + pry
    src_cells, tgt_cells = cidx[src], cidx[tgt]

    # Build per-cell pin and edge indices
    pin_order = np.argsort(cidx)
    sorted_cidx = cidx[pin_order]
    pin_starts = np.searchsorted(sorted_cidx, np.arange(N), side='left')
    pin_ends = np.searchsorted(sorted_cidx, np.arange(N), side='right')

    ec_all = np.concatenate([src_cells, tgt_cells])
    ei_all = np.concatenate([np.arange(E), np.arange(E)])
    eord = np.argsort(ec_all)
    sec, sei = ec_all[eord], ei_all[eord]
    es = np.searchsorted(sec, np.arange(N), side='left')
    ee = np.searchsorted(sec, np.arange(N), side='right')
    cedges = [
        np.unique(sei[es[c]:ee[c]]).astype(np.intp) for c in range(N)
    ]

    # Build swap candidate list
    valid_edges = src_cells != tgt_cells
    sc_v, tc_v = src_cells[valid_edges], tgt_cells[valid_edges]
    lo_e, hi_e = np.minimum(sc_v, tc_v), np.maximum(sc_v, tc_v)
    ukeys = np.unique(lo_e.astype(np.int64) * N + hi_e.astype(np.int64))
    swap_list = list(zip(
        (ukeys // N).astype(np.intp).tolist(),
        (ukeys % N).astype(np.intp).tolist(),
    ))

    if N <= 500:
        existing = set(swap_list)
        for i in range(N):
            for j in range(i + 1, N):
                existing.add((i, j))
        swap_list = sorted(existing)

    is_macro = h > 1.5
    use_grid = N > 300

    if use_grid:
        max_std_hw = hw[~is_macro].max() if (~is_macro).any() else 1.0
        grid_size = max(max_std_hw * 4, 2.0)
        sgrid = _SpatialGrid(pos, is_macro, grid_size)
        sr_std = 2
        sr_macro = (
            int(np.ceil((hw[is_macro].max() + max_std_hw) / grid_size)) + 1
            if is_macro.any()
            else 2
        )

        def no_ov(ci, nx, ny, skip):
            sr = sr_macro if is_macro[ci] else sr_std
            return not sgrid.check_overlap_skip(
                ci, nx, ny, pos, hw, hh, sr, skip
            )
    else:
        sgrid = None

        def no_ov(ci, nx, ny, skip):
            ox = np.abs(nx - pos[:, 0])
            oy = np.abs(ny - pos[:, 1])
            ox[ci] = 1e18
            ox[skip] = 1e18
            return not np.any((ox < hw[ci] + hw) & (oy < hh[ci] + hh))

    for _ in range(max_passes):
        improved = False

        for i, j in swap_list:
            if not no_ov(i, pos[j, 0], pos[j, 1], j):
                continue
            if not no_ov(j, pos[i, 0], pos[i, 1], i):
                continue

            ae = np.union1d(cedges[i], cedges[j])
            ad = np.abs(px[src[ae]] - px[tgt[ae]])
            bd = np.abs(py[src[ae]] - py[tgt[ae]])
            mx = np.maximum(ad, bd)
            old_wl = (
                0.1 * np.log(np.exp((ad - mx) * 10) + np.exp((bd - mx) * 10))
                + mx
            ).sum()

            oix, oiy = pos[i, 0], pos[i, 1]
            ojx, ojy = pos[j, 0], pos[j, 1]
            pos[i, 0], pos[j, 0] = ojx, oix
            pos[i, 1], pos[j, 1] = ojy, oiy

            pi = pin_order[pin_starts[i]:pin_ends[i]]
            pj = pin_order[pin_starts[j]:pin_ends[j]]
            if len(pi):
                px[pi] = pos[i, 0] + prx[pi]
                py[pi] = pos[i, 1] + pry[pi]
            if len(pj):
                px[pj] = pos[j, 0] + prx[pj]
                py[pj] = pos[j, 1] + pry[pj]

            ad = np.abs(px[src[ae]] - px[tgt[ae]])
            bd = np.abs(py[src[ae]] - py[tgt[ae]])
            mx = np.maximum(ad, bd)
            new_wl = (
                0.1 * np.log(np.exp((ad - mx) * 10) + np.exp((bd - mx) * 10))
                + mx
            ).sum()

            if new_wl < old_wl - 1e-6:
                improved = True
                if sgrid:
                    sgrid.update(i, oix, oiy, pos[i, 0], pos[i, 1])
                    sgrid.update(j, ojx, ojy, pos[j, 0], pos[j, 1])
            else:
                pos[i, 0], pos[j, 0] = oix, ojx
                pos[i, 1], pos[j, 1] = oiy, ojy
                if len(pi):
                    px[pi] = oix + prx[pi]
                    py[pi] = oiy + pry[pi]
                if len(pj):
                    px[pj] = ojx + prx[pj]
                    py[pj] = ojy + pry[pj]

        if not improved:
            break

    result = cell_features.clone()
    result[:, 2:4] = torch.from_numpy(pos).float()
    return result


def _slide_refine(cell_features, pin_features, edge_list, max_passes=15):
    """Detailed placement: single-cell sliding (NTUPlace3-style).

    Moves individual cells toward their optimal positions (based on
    connected pin locations) without introducing overlaps.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity
        max_passes: Maximum number of sliding passes

    Returns:
        Updated cell_features tensor with refined positions
    """
    N = cell_features.shape[0]
    if edge_list.shape[0] == 0:
        return cell_features

    pos = cell_features[:, 2:4].detach().clone().numpy().astype(np.float64)
    w = cell_features[:, 4].detach().numpy().astype(np.float64)
    h = cell_features[:, 5].detach().numpy().astype(np.float64)
    hw, hh = w / 2.0, h / 2.0

    cidx = pin_features[:, 0].long().numpy()
    prx = pin_features[:, 1].detach().numpy().astype(np.float64)
    pry = pin_features[:, 2].detach().numpy().astype(np.float64)
    src = edge_list[:, 0].long().numpy()
    tgt = edge_list[:, 1].long().numpy()
    E = len(src)

    px, py = pos[cidx, 0] + prx, pos[cidx, 1] + pry

    # Build per-cell pin and edge indices
    pin_order = np.argsort(cidx)
    sorted_cidx = cidx[pin_order]
    pin_starts = np.searchsorted(sorted_cidx, np.arange(N), side='left')
    pin_ends = np.searchsorted(sorted_cidx, np.arange(N), side='right')

    src_cells, tgt_cells = cidx[src], cidx[tgt]
    ec_all = np.concatenate([src_cells, tgt_cells])
    ei_all = np.concatenate([np.arange(E), np.arange(E)])
    eord = np.argsort(ec_all)
    sec, sei = ec_all[eord], ei_all[eord]
    es = np.searchsorted(sec, np.arange(N), side='left')
    ee = np.searchsorted(sec, np.arange(N), side='right')
    cedge_arr = [
        np.unique(sei[es[c]:ee[c]]).astype(np.intp) for c in range(N)
    ]

    # Build inter-cell partner lists
    inter_partners = [None] * N
    for i in range(N):
        ei = cedge_arr[i]
        if len(ei) == 0:
            continue
        si, ti = src[ei], tgt[ei]
        mask = cidx[si] != cidx[ti]
        if not mask.any():
            continue
        ie = ei[mask]
        s, t = src[ie], tgt[ie]
        inter_partners[i] = np.where(cidx[s] == i, t, s)

    _a, _ia = 0.1, 10.0

    def cell_wl(i):
        edges = cedge_arr[i]
        if len(edges) == 0:
            return 0.0
        s, t = src[edges], tgt[edges]
        adx, ady = np.abs(px[s] - px[t]), np.abs(py[s] - py[t])
        mx = np.maximum(adx, ady)
        return (
            _a * np.log(np.exp((adx - mx) * _ia) + np.exp((ady - mx) * _ia))
            + mx
        ).sum()

    is_macro = h > 1.5
    use_grid = N > 300

    if use_grid:
        max_std_hw = hw[~is_macro].max() if (~is_macro).any() else 1.0
        gs = max(max_std_hw * 4, 2.0)
        sgrid = _SpatialGrid(pos, is_macro, gs)
        sr_std = 2
        sr_macro = (
            int(np.ceil((hw[is_macro].max() + max_std_hw) / gs)) + 1
            if is_macro.any()
            else 2
        )

        def no_overlap(i, nx, ny):
            sr = sr_macro if is_macro[i] else sr_std
            return not sgrid.check_overlap(i, nx, ny, pos, hw, hh, sr)
    else:
        sgrid = None

        def no_overlap(i, nx, ny):
            ox = np.abs(nx - pos[:, 0])
            oy = np.abs(ny - pos[:, 1])
            ox[i] = 1e18
            return not np.any((ox < hw[i] + hw) & (oy < hh[i] + hh))

    def apply_move(i, nx, ny):
        ox, oy = pos[i, 0], pos[i, 1]
        pos[i, 0] = nx
        pos[i, 1] = ny
        pi = pin_order[pin_starts[i]:pin_ends[i]]
        if len(pi):
            px[pi] = nx + prx[pi]
            py[pi] = ny + pry[pi]
        if sgrid:
            sgrid.update(i, ox, oy, nx, ny)

    def undo_move(i, ox, oy, cx, cy):
        pos[i, 0] = ox
        pos[i, 1] = oy
        pi = pin_order[pin_starts[i]:pin_ends[i]]
        if len(pi):
            px[pi] = ox + prx[pi]
            py[pi] = oy + pry[pi]
        if sgrid:
            sgrid.update(i, cx, cy, ox, oy)

    def try_move(i, nx, ny, ow):
        if not no_overlap(i, nx, ny):
            return 0.0
        ox, oy = pos[i, 0], pos[i, 1]
        apply_move(i, nx, ny)
        nw = cell_wl(i)
        if nw < ow - 1e-8:
            return ow - nw
        undo_move(i, ox, oy, nx, ny)
        return 0.0

    def try_bisect(i, ddx, ddy, ow):
        fx, fy = pos[i, 0] + ddx, pos[i, 1] + ddy
        if no_overlap(i, fx, fy):
            ox, oy = pos[i, 0], pos[i, 1]
            apply_move(i, fx, fy)
            nw = cell_wl(i)
            if nw < ow - 1e-8:
                return ow - nw
            undo_move(i, ox, oy, fx, fy)
            return 0.0
        lo, hi_b = 0.0, 1.0
        for _ in range(5):
            mid = (lo + hi_b) / 2.0
            if no_overlap(i, pos[i, 0] + mid * ddx, pos[i, 1] + mid * ddy):
                lo = mid
            else:
                hi_b = mid
        if lo < 0.02:
            return 0.0
        bx, by = pos[i, 0] + lo * ddx, pos[i, 1] + lo * ddy
        ox, oy = pos[i, 0], pos[i, 1]
        apply_move(i, bx, by)
        nw = cell_wl(i)
        if nw < ow - 1e-8:
            return ow - nw
        undo_move(i, ox, oy, bx, by)
        return 0.0

    def compute_grad(i):
        p = inter_partners[i]
        if p is None:
            return 0.0, 0.0
        ddx, ddy = pos[i, 0] - px[p], pos[i, 1] - py[p]
        adx, ady = np.abs(ddx), np.abs(ddy)
        mx = np.maximum(adx, ady)
        ex = np.exp((adx - mx) * _ia)
        ey = np.exp((ady - mx) * _ia)
        d = ex + ey
        sx = np.sign(ddx)
        sx[sx == 0] = 1.0
        sy = np.sign(ddy)
        sy[sy == 0] = 1.0
        return float(np.sum(ex / d * sx)), float(np.sum(ey / d * sy))

    _fracs = (1.0, 0.5, 0.25, 0.125)

    for pass_idx in range(max_passes):
        total_imp = 0.0
        order = (
            np.arange(N) if pass_idx % 2 == 0
            else np.arange(N - 1, -1, -1)
        )

        for i in order:
            p = inter_partners[i]
            if p is None:
                continue
            ow = cell_wl(i)
            if ow < 1e-10:
                continue

            mx, my = np.mean(px[p]), np.mean(py[p])
            dm, dn = mx - pos[i, 0], my - pos[i, 1]
            gx, gy = compute_grad(i)
            gn = max(np.sqrt(gx * gx + gy * gy), 1e-12)
            ss = max(abs(dm), abs(dn), 1.0)

            # Try gradient-based move
            moved = False
            for f in _fracs:
                imp = try_move(
                    i,
                    pos[i, 0] - f * ss * gx / gn,
                    pos[i, 1] - f * ss * gy / gn,
                    ow,
                )
                if imp > 0:
                    total_imp += imp
                    moved = True
                    break

            # Try mean-based move
            if not moved:
                for f in _fracs:
                    imp = try_move(
                        i, pos[i, 0] + f * dm, pos[i, 1] + f * dn, ow
                    )
                    if imp > 0:
                        total_imp += imp
                        moved = True
                        break

            # Try bisection move
            if not moved and max(abs(dm), abs(dn)) > 1e-6:
                imp = try_bisect(i, dm, dn, ow)
                if imp > 0:
                    total_imp += imp
                    moved = True

            # Try axis-aligned moves
            if not moved:
                cw = ow
                for f in _fracs[:3]:
                    imp = try_move(i, pos[i, 0] + f * dm, pos[i, 1], cw)
                    if imp > 0:
                        total_imp += imp
                        cw -= imp
                        break
                for f in _fracs[:3]:
                    imp = try_move(i, pos[i, 0], pos[i, 1] + f * dn, cw)
                    if imp > 0:
                        total_imp += imp
                        break

            # Try axis-aligned bisection
            if not moved:
                cw = cell_wl(i)
                if abs(dm) > 1e-6:
                    imp = try_bisect(i, dm, 0.0, cw)
                    if imp > 0:
                        total_imp += imp
                        cw -= imp
                if abs(dn) > 1e-6:
                    imp = try_bisect(i, 0.0, dn, cw)
                    if imp > 0:
                        total_imp += imp

        if total_imp < 1e-6:
            break

    result = cell_features.clone()
    result[:, 2:4] = torch.from_numpy(pos).float()
    return result


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
        return torch.tensor(0.0, requires_grad=True)

    # TODO: Implement overlap detection and loss calculation here
    #
    # Your implementation should:
    # 1. Extract cell positions, widths, and heights
    # 2. Compute pairwise overlaps using vectorized operations
    # 3. Return a scalar loss that is zero when no overlaps exist
    #
    # Delete this placeholder and add your implementation:

    # Placeholder - returns a constant loss (REPLACE THIS!)
    return torch.tensor(1.0, requires_grad=True)


def train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=2000,
    lr=0.1,
    lambda_wirelength=1.0,
    lambda_overlap=200.0,
    verbose=True,
    log_interval=100,
):
    """Train the placement optimization using gradient descent.

    Uses a multi-stage pipeline:
    1. Analytical placement (spectral)
    2. Row-based legalization
    3. Gradient optimization with overlap penalty scheduling
    4. Iterative overlap resolution
    5. Detailed placement (slide + swap + slide)

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
    N = cell_features.shape[0]

    # Track loss history
    loss_history = {
        "total_loss": [],
        "wirelength_loss": [],
        "overlap_loss": [],
    }

    widths, heights = cell_features[:, 4], cell_features[:, 5]

    # Fast path for very large designs
    if N > 100000:
        apos = _analytical_place(
            cell_features, pin_features, edge_list, iters=200
        )
        cpos = _legalize_from_analytical(
            cell_features, apos, pin_features, edge_list
        )
        result = cell_features.clone()
        result[:, 2:4] = cpos
        return {
            "final_cell_features": result,
            "initial_cell_features": initial_cell_features,
            "loss_history": loss_history,
        }

    # Stage 1-2: Analytical placement + legalization
    analytical_pos = _analytical_place(
        cell_features, pin_features, edge_list, iters=200
    )
    cell_positions = _legalize_from_analytical(
        cell_features, analytical_pos, pin_features, edge_list
    )
    cell_positions = cell_positions.clone().detach().requires_grad_(True)

    # All hyperparameters smooth functions of log2(N)
    log_n = max(np.log2(max(N, 4)), 2.0)
    sqrt_n = max(np.sqrt(N), 1.0)
    num_epochs = int(np.clip(25000 / log_n ** 1.5, 500, 2500))
    lr = float(np.clip(1.2 / log_n, 0.06, 0.25))
    lambda_overlap = float(np.clip(40 * log_n, 150, 400))
    _ol_start = float(np.clip(0.0002 * log_n, 0.0005, 0.01))
    _ol_end_mult = float(np.clip(0.5 * log_n, 1.5, 8.0))
    _ol_ratio = (_ol_end_mult * lambda_overlap) / _ol_start

    # Create optimizer with warmup + cosine schedule
    warmup = max(10, num_epochs // 12)
    optimizer = optim.Adam([cell_positions], lr=lr, betas=(0.9, 0.999))

    def lr_sched(epoch):
        if epoch < warmup:
            return 0.1 + 0.9 * (epoch / warmup)
        p = (epoch - warmup) / max(num_epochs - warmup - 1, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + np.cos(np.pi * p))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    # Precompute indices for loss computation
    _ci = pin_features[:, 0].long()
    _prx, _pry = pin_features[:, 1], pin_features[:, 2]
    _ne = edge_list.shape[0]
    _he = _ne > 0
    if _he:
        _src, _tgt = edge_list[:, 0].long(), edge_list[:, 1].long()
    _alpha, _inv = 0.1, 10.0

    # Precompute overlap structures based on design size
    _pw = N <= 500
    if _pw:
        _hw = (widths.unsqueeze(1) + widths.unsqueeze(0)) / 2
        _hh = (heights.unsqueeze(1) + heights.unsqueeze(0)) / 2
        _tr, _tc = torch.triu_indices(N, N, offset=1)
    else:
        _im = heights > 1.5
        _mi = torch.where(_im)[0]
        _si = torch.where(~_im)[0]
        _Nm, _Ns = _mi.shape[0], _si.shape[0]

        _hmm = _Nm > 1
        if _hmm:
            mw, mh = widths[_mi], heights[_mi]
            _mmhw = (mw.unsqueeze(1) + mw.unsqueeze(0)) / 2
            _mmhh = (mh.unsqueeze(1) + mh.unsqueeze(0)) / 2
            _mmr, _mmc = torch.triu_indices(_Nm, _Nm, offset=1)

        _hms = _Nm > 0 and _Ns > 0
        if _hms:
            _mshw = (
                (widths[_mi].unsqueeze(1) + widths[_si].unsqueeze(0)) / 2
            )
            _mshh = (
                (heights[_mi].unsqueeze(1) + heights[_si].unsqueeze(0)) / 2
            )

        _hss = _Ns > 1
        if _hss:
            _sw, _sh = widths[_si], heights[_si]
            _msw = _sw.max().item()
            _K = min(max(30, int(np.sqrt(_Ns) * 1.2)), _Ns - 1)

    inv_N = 1.0 / N
    _bw, _bp, _zs = float('inf'), None, 0

    # Stage 3: Gradient optimization with overlap penalty scheduling
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Wirelength loss
        if _he:
            pax = cell_positions[_ci, 0] + _prx
            pay = cell_positions[_ci, 1] + _pry
            dx = torch.abs(pax[_src] - pax[_tgt])
            dy = torch.abs(pay[_src] - pay[_tgt])
            wl = (
                _alpha * torch.logaddexp(dx * _inv, dy * _inv).sum()
            ) / _ne
        else:
            wl = torch.tensor(0.0, requires_grad=True)

        # Overlap loss
        if _pw:
            pdx = torch.abs(
                cell_positions[:, 0].unsqueeze(1)
                - cell_positions[:, 0].unsqueeze(0)
            )
            pdy = torch.abs(
                cell_positions[:, 1].unsqueeze(1)
                - cell_positions[:, 1].unsqueeze(0)
            )
            ov = (
                torch.relu(_hw - pdx) * torch.relu(_hh - pdy)
            )[_tr, _tc]
            ol = (ov.sum() + ov.pow(3).sum()) * inv_N
        else:
            ol = torch.tensor(0.0)

            # Macro-macro overlaps
            if _hmm:
                mp = cell_positions[_mi]
                mdx = torch.abs(
                    mp[:, 0].unsqueeze(1) - mp[:, 0].unsqueeze(0)
                )
                mdy = torch.abs(
                    mp[:, 1].unsqueeze(1) - mp[:, 1].unsqueeze(0)
                )
                ov = (
                    torch.relu(_mmhw - mdx) * torch.relu(_mmhh - mdy)
                )[_mmr, _mmc]
                ol = ol + ov.sum() + ov.pow(3).sum()

            # Macro-standard cell overlaps
            if _hms:
                mp, sp = cell_positions[_mi], cell_positions[_si]
                oms = (
                    torch.relu(
                        _mshw
                        - torch.abs(
                            mp[:, 0].unsqueeze(1) - sp[:, 0].unsqueeze(0)
                        )
                    )
                    * torch.relu(
                        _mshh
                        - torch.abs(
                            mp[:, 1].unsqueeze(1) - sp[:, 1].unsqueeze(0)
                        )
                    )
                )
                ol = ol + oms.sum() + oms.pow(3).sum()

            # Standard cell-standard cell overlaps (sweep-based)
            if _hss:
                sp = cell_positions[_si]
                with torch.no_grad():
                    order = torch.argsort(sp[:, 0])
                sps, sws, shs = sp[order], _sw[order], _sh[order]
                ss_s = ss_c = torch.tensor(0.0)
                for k in range(1, _K + 1):
                    n = _Ns - k
                    dxk = sps[k:, 0] - sps[:n, 0]
                    if dxk.detach().min().item() > _msw:
                        break
                    dyk = torch.abs(sps[k:, 1] - sps[:n, 1])
                    ov = (
                        torch.relu((sws[:n] + sws[k:]) * 0.5 - dxk)
                        * torch.relu((shs[:n] + shs[k:]) * 0.5 - dyk)
                    )
                    ss_s = ss_s + ov.sum()
                    ss_c = ss_c + ov.pow(3).sum()
                ol = ol + ss_s + ss_c

            ol = ol * inv_N

        # Combined loss with scheduled weights
        prog = epoch / max(num_epochs - 1, 1)
        c_ol = _ol_start * (_ol_ratio ** prog)
        c_wl = lambda_wirelength * (1.0 - 0.3 * prog)
        loss = c_wl * wl + c_ol * ol

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent extreme updates
        torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=10.0)

        # Update positions
        optimizer.step()
        scheduler.step()

        # Record losses
        loss_history["total_loss"].append(loss.item())
        loss_history["wirelength_loss"].append(wl.item())
        loss_history["overlap_loss"].append(ol.item())

        # Track best zero-overlap solution
        ov_v, wl_v = ol.item(), wl.item()
        if ov_v < 1e-4 and wl_v < _bw:
            _bw = wl_v
            _bp = cell_positions.detach().clone()
        if ov_v < 1e-6:
            _zs += 1
        else:
            _zs = 0

        # Early stopping if converged
        if _zs >= 100 and epoch > num_epochs * 2 // 3:
            wh = loss_history["wirelength_loss"]
            if (
                abs(np.mean(wh[-100:-50]) - np.mean(wh[-50:]))
                / max(abs(np.mean(wh[-100:-50])), 1e-10)
                < 0.001
            ):
                break

        # Log progress
        if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            print(
                f"Epoch {epoch}/{num_epochs}:"
                f" Total={loss.item():.6f}"
                f" WL={wl.item():.6f}"
                f" OL={ol.item():.6f}"
            )

    # Create final cell features
    final = cell_features.clone()
    final[:, 2:4] = (
        _bp if _bp is not None and _bw < wl.item()
        else cell_positions.detach()
    )

    # Stage 4: Overlap resolution
    final = _resolve_overlaps(
        final, max_iters=int(np.clip(800 + N * 0.3, 800, 2000))
    )

    # Stage 5: Detailed placement - slide/swap/slide with smooth pass counts
    slide_passes = max(6, min(25, int(500 / sqrt_n)))
    swap_passes = max(1, min(3, int(50 / sqrt_n)))
    final = _slide_refine(
        final, pin_features, edge_list, max_passes=slide_passes
    )
    final = _swap_refine(
        final, pin_features, edge_list, max_passes=swap_passes
    )
    final = _slide_refine(
        final, pin_features, edge_list, max_passes=max(3, slide_passes // 3)
    )

    return {
        "final_cell_features": final,
        "initial_cell_features": initial_cell_features,
        "loss_history": loss_history,
    }


# ======= FINAL EVALUATION CODE (Don't edit this part) =======

def _sweep_overlap_pairs(positions, widths, heights):
    """Sweep-line algorithm for finding all overlapping cell pairs.

    Sorts cells by x-coordinate and checks nearby pairs for overlap,
    providing O(N log N + K) performance where K is the number of overlaps.

    Args:
        positions: [N, 2] array of cell positions
        widths: [N] array of cell widths
        heights: [N] array of cell heights

    Returns:
        Tuple of (pairs, areas):
            - pairs: list of (i, j) tuples of overlapping cell indices
            - areas: list of overlap areas for each pair
    """
    N = len(positions)
    si = np.argsort(positions[:, 0])
    sp, sw, sh = positions[si], widths[si], heights[si]
    md = max(widths.max(), heights.max())
    pairs, areas = [], []

    for k in range(1, N):
        n = N - k
        dx = sp[k:, 0] - sp[:n, 0]
        if dx.min() > md:
            break
        ox = (sw[:n] + sw[k:]) / 2 - dx
        xc = ox > 0
        if not xc.any():
            continue
        ady = np.abs(sp[:n, 1] - sp[k:, 1])
        oy = (sh[:n] + sh[k:]) / 2 - ady
        hit = xc & (oy > 0)
        if not hit.any():
            continue
        for idx in np.where(hit)[0]:
            i, j = int(si[idx]), int(si[idx + k])
            if i > j:
                i, j = j, i
            pairs.append((i, j))
            areas.append(float(ox[idx] * oy[idx]))

    return pairs, areas


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

    pos = cell_features[:, 2:4].detach().numpy()
    w = cell_features[:, 4].detach().numpy()
    h = cell_features[:, 5].detach().numpy()
    a = cell_features[:, 0].detach().numpy()

    pairs, ov = _sweep_overlap_pairs(pos, w, h)

    return {
        "overlap_count": len(pairs),
        "total_overlap_area": sum(ov) if ov else 0.0,
        "max_overlap_area": max(ov) if ov else 0.0,
        "overlap_percentage": (
            (len(pairs) / N * 100) if sum(a) > 0 else 0.0
        ),
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

    pairs, _ = _sweep_overlap_pairs(
        cell_features[:, 2:4].detach().numpy(),
        cell_features[:, 4].detach().numpy(),
        cell_features[:, 5].detach().numpy(),
    )

    cells = set()
    for i, j in pairs:
        cells.add(i)
        cells.add(j)

    return cells


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
    num_ov = len(calculate_cells_with_overlaps(cell_features))

    if edge_list.shape[0] == 0:
        return {
            "overlap_ratio": num_ov / N if N > 0 else 0.0,
            "normalized_wl": 0.0,
            "num_cells_with_overlaps": num_ov,
            "total_cells": N,
            "num_nets": 0,
        }

    # Calculate wirelength metric: (wirelength / num nets) / sqrt(total area)
    wl = wirelength_attraction_loss(cell_features, pin_features, edge_list)
    ta = cell_features[:, 0].sum().item()
    ne = edge_list.shape[0]

    # Normalize: (wirelength / net) / sqrt(area)
    # This gives a dimensionless quality metric independent of design size
    normalized_wl = (
        (wl.item() * ne / ne) / (ta ** 0.5) if ta > 0 else 0.0
    )

    return {
        "overlap_ratio": num_ov / N if N > 0 else 0.0,
        "normalized_wl": normalized_wl,
        "num_cells_with_overlaps": num_ov,
        "total_cells": N,
        "num_nets": ne,
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
    print(
        f"Overlap Ratio: {normalized_metrics['overlap_ratio']:.4f} "
        f"({normalized_metrics['num_cells_with_overlaps']}"
        f"/{normalized_metrics['total_cells']} cells)"
    )
    print(f"Normalized Wirelength: {normalized_metrics['normalized_wl']:.4f}")

    # Success check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA")
    print("=" * 70)
    if normalized_metrics["num_cells_with_overlaps"] == 0:
        print("✓ PASS: No overlapping cells!")
        print("✓ PASS: Overlap ratio is 0.0")
        print(
            "\nCongratulations! Your implementation successfully"
            " eliminated all overlaps."
        )
        print(
            f"Your normalized wirelength:"
            f" {normalized_metrics['normalized_wl']:.4f}"
        )
    else:
        print("✗ FAIL: Overlaps still exist")
        print(
            f"  Need to eliminate overlaps in"
            f" {normalized_metrics['num_cells_with_overlaps']} cells"
        )
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
