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

# Above this cell count the exact N x N overlap loss is too slow per epoch
# (~1.2s at N=10,010) and too large in RAM, so training switches to the
# chunked Differentiable Sparse Pairs path instead.
EXACT_OVERLAP_MAX_N = 3000

# Above this cell count even a one-time dense O(N^2) legalization pass is too
# expensive, so the final safety net uses spatial windowing instead.
DENSE_LEGALIZE_MAX_N = 20000


def wirelength_attraction_loss(cell_features, pin_features, edge_list):
    """Calculate loss based on total wirelength to minimize routing.

    The loss computes the smooth Manhattan distance between connected pins and minimizes
    the total wirelength across all edges.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information
        edge_list: [E, 2] tensor with edges

    Returns:
        Scalar loss value
    """
    if edge_list.shape[0] == 0:
        return torch.tensor(0.0, device=cell_features.device, requires_grad=True)

    # Extract cell center positions (X, Y)
    cell_positions = cell_features[:, 2:4]  # [N, 2]
    cell_indices = pin_features[:, 0].long()

    # Calculate absolute pin positions = cell_center + relative_pin_offset
    pin_absolute_x = cell_positions[cell_indices, 0] + pin_features[:, 1]
    pin_absolute_y = cell_positions[cell_indices, 1] + pin_features[:, 2]

    # Get source and target pin positions for each edge
    src_pins = edge_list[:, 0].long()
    tgt_pins = edge_list[:, 1].long()

    src_x = pin_absolute_x[src_pins]
    src_y = pin_absolute_y[src_pins]
    tgt_x = pin_absolute_x[tgt_pins]
    tgt_y = pin_absolute_y[tgt_pins]

    # Smooth L1 approximation of Manhattan distance (|dx| + |dy|)
    eps = 1e-4
    smooth_dx = torch.sqrt((src_x - tgt_x) ** 2 + eps)
    smooth_dy = torch.sqrt((src_y - tgt_y) ** 2 + eps)
    smooth_manhattan = smooth_dx + smooth_dy

    # Return average wirelength per edge
    return torch.sum(smooth_manhattan) / edge_list.shape[0]


def _spatial_hash_sort(positions, widths, heights, num_bins, col_major=False):
    """Sort cell indices into spatially-local order via grid binning.

    Nearby cells end up close together in the sorted order, so a small sliding
    window over it captures most potentially-overlapping pairs without ever
    forming the full N x N matrix.

    Args:
        positions: [N, 2] tensor of cell centers
        widths: [N] tensor of cell widths
        heights: [N] tensor of cell heights
        num_bins: Grid resolution per axis
        col_major: Bin by column-then-row instead of row-then-column

    Returns:
        [N] tensor of cell indices in spatially-sorted order

    Note: each ordering keeps one axis's neighbors contiguous and can miss the
    other's under uneven density, so callers alternate between them.
    """
    with torch.no_grad():
        min_xy = torch.min(positions - torch.stack([widths, heights], dim=1) / 2.0, dim=0)[0]
        max_xy = torch.max(positions + torch.stack([widths, heights], dim=1) / 2.0, dim=0)[0]
        span_xy = torch.clamp(max_xy - min_xy, min=10.0)
        bin_size = span_xy / float(num_bins)

        norm_pos = positions - min_xy
        bx = torch.clamp((norm_pos[:, 0] / bin_size[0]).long(), 0, num_bins - 1)
        by = torch.clamp((norm_pos[:, 1] / bin_size[1]).long(), 0, num_bins - 1)
        bin_id = (bx * num_bins + by) if col_major else (by * num_bins + bx)

        _, sorted_indices = torch.sort(bin_id)
    return sorted_indices


def _dsp_overlap_backward(cell_positions, widths, heights, areas, margin, lambda_overlap, num_bins, block, col_major=False):
    """Differentiable Sparse Pairs overlap loss via spatial hashing, for large N.

    Only compares cells within a sliding window of the spatially-sorted order,
    so the pairwise tensors stay block x block no matter how large N gets.

    Args:
        cell_positions: [N, 2] leaf tensor of cell centers (requires_grad)
        widths: [N] tensor of cell widths
        heights: [N] tensor of cell heights
        areas: [N] tensor of cell areas
        margin: Extra separation margin
        lambda_overlap: Overlap penalty weight
        num_bins: Grid resolution per axis
        block: Window size in cells
        col_major: Bin ordering to use this pass

    Returns:
        Tuple of (total_loss_value, has_overlap) -- a float for logging, and
        whether any window still overlapped

    Note: each window is backpropagated immediately rather than summed into one
    loss, since autograd keeps every window's tensors alive until backward()
    runs (30+ GB at N=100,000). Gradients still accumulate across windows.
    """
    N = cell_positions.shape[0]
    sorted_indices = _spatial_hash_sort(cell_positions.detach(), widths, heights, num_bins, col_major=col_major)
    step = max(1, block // 2)

    total_loss_value = 0.0
    has_overlap = False
    i = 0
    while i < N:
        j = min(N, i + block)
        idx = sorted_indices[i:j]
        n_sub = j - i

        sub_pos = cell_positions[idx]
        sub_w = widths[idx]
        sub_h = heights[idx]
        sub_area = areas[idx]

        dx = torch.abs(sub_pos[:, 0].unsqueeze(1) - sub_pos[:, 0].unsqueeze(0))
        dy = torch.abs(sub_pos[:, 1].unsqueeze(1) - sub_pos[:, 1].unsqueeze(0))
        min_sep_x = 0.5 * (sub_w.unsqueeze(1) + sub_w.unsqueeze(0)) + margin
        min_sep_y = 0.5 * (sub_h.unsqueeze(1) + sub_h.unsqueeze(0)) + margin
        overlap_x = torch.relu(min_sep_x - dx)
        overlap_y = torch.relu(min_sep_y - dy)
        overlap_area = overlap_x * overlap_y

        diag_mask = ~torch.eye(n_sub, dtype=torch.bool, device=cell_positions.device)
        active_mask = (overlap_area > 0) & diag_mask

        if active_mask.any():
            has_overlap = True
            min_w = torch.minimum(sub_w.unsqueeze(1), sub_w.unsqueeze(0))
            min_h = torch.minimum(sub_h.unsqueeze(1), sub_h.unsqueeze(0))
            rel_overlap = (overlap_x / min_w) * (overlap_y / min_h)
            area_weights = torch.sqrt(sub_area.unsqueeze(1) * sub_area.unsqueeze(0))

            loss_matrix = (5.0 * overlap_area + overlap_area ** 2.0 + 10.0 * rel_overlap) * area_weights
            chunk_loss = torch.sum(loss_matrix[active_mask]) / 2.0
            scaled_loss = (lambda_overlap / 20.0) * chunk_loss

            scaled_loss.backward()
            total_loss_value += scaled_loss.item()

        if j >= N:
            break
        i += step

    return total_loss_value, has_overlap


def _chunked_overlap_count(positions, widths, heights, num_bins, block, col_major=False):
    """Count overlapping cell pairs (no_grad) via spatial hashing.

    Args:
        positions: [N, 2] tensor of cell centers
        widths: [N] tensor of cell widths
        heights: [N] tensor of cell heights
        num_bins: Grid resolution per axis
        block: Window size in cells
        col_major: Bin ordering to use

    Returns:
        Number of overlapping pairs found (window-local, so approximate unless
        block >= N; callers needing certainty check both orderings)
    """
    N = positions.shape[0]
    sorted_indices = _spatial_hash_sort(positions, widths, heights, num_bins, col_major=col_major)
    step = max(1, block // 2)
    count = 0
    i = 0
    with torch.no_grad():
        while i < N:
            j = min(N, i + block)
            idx = sorted_indices[i:j]
            p = positions[idx]
            w = widths[idx]
            h = heights[idx]
            dx = torch.abs(p[:, 0].unsqueeze(1) - p[:, 0].unsqueeze(0))
            dy = torch.abs(p[:, 1].unsqueeze(1) - p[:, 1].unsqueeze(0))
            min_x = 0.5 * (w.unsqueeze(1) + w.unsqueeze(0))
            min_y = 0.5 * (h.unsqueeze(1) + h.unsqueeze(0))
            ov = (dx < min_x) & (dy < min_y)
            ov.fill_diagonal_(False)
            count += ov.sum().item()
            if j >= N:
                break
            i += step
    return count


def _legalize_large_placement(positions, widths, heights, num_bins, block, max_iters=40, deadline=None):
    """Chunked, memory-safe version of the exact push-apart legalizer for large N.

    Nudges overlapping pairs apart along their shorter overlap axis and
    iterates, computed in spatially-local windows so it stays cheap at N=100k.

    Args:
        positions: [N, 2] tensor of cell centers
        widths: [N] tensor of cell widths
        heights: [N] tensor of cell heights
        num_bins: Grid resolution per axis
        block: Window size in cells
        max_iters: Maximum push-apart iterations
        deadline: Optional time.time() value at which to stop early

    Returns:
        [N, 2] tensor of adjusted cell centers

    Note: bin ordering alternates each iteration, since a fixed one can miss
    pairs under uneven density (a row-major-only pass left ~5% of Test 11's
    cells overlapping).
    """
    import time

    positions = positions.clone()
    step = max(1, block // 2)
    N = positions.shape[0]

    with torch.no_grad():
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            sorted_indices = _spatial_hash_sort(positions, widths, heights, num_bins, col_major=(it % 2 == 1))
            delta = torch.zeros_like(positions)
            any_overlap = False
            i = 0
            while i < N:
                j = min(N, i + block)
                idx = sorted_indices[i:j]
                p = positions[idx]
                w = widths[idx]
                h = heights[idx]

                dx_mat = p[:, 0].unsqueeze(1) - p[:, 0].unsqueeze(0)
                dy_mat = p[:, 1].unsqueeze(1) - p[:, 1].unsqueeze(0)
                abs_dx = torch.abs(dx_mat)
                abs_dy = torch.abs(dy_mat)
                min_dx = 0.5 * (w.unsqueeze(1) + w.unsqueeze(0)) + 0.01
                min_dy = 0.5 * (h.unsqueeze(1) + h.unsqueeze(0)) + 0.01

                ov_x = torch.clamp(min_dx - abs_dx, min=0.0)
                ov_y = torch.clamp(min_dy - abs_dy, min=0.0)
                ov_mask = (ov_x > 0) & (ov_y > 0)
                ov_mask.fill_diagonal_(False)

                if ov_mask.any():
                    any_overlap = True
                    push_mask_x = ov_x <= ov_y
                    push_mask_y = ~push_mask_x
                    push_x = torch.sign(dx_mat) * ov_x * push_mask_x.float()
                    push_y = torch.sign(dy_mat) * ov_y * push_mask_y.float()

                    zero_dist = (abs_dx == 0) & (abs_dy == 0) & ov_mask
                    if zero_dist.any():
                        push_x[zero_dist] = 0.1
                        push_y[zero_dist] = 0.1

                    # Average the push over the cells this one overlaps rather
                    # than summing: a raw sum explodes when a cell is piled up
                    # with many neighbors at once (measured sending Test 10
                    # from 0.04 to 4.13 normalized WL)
                    overlap_count = ov_mask.sum(dim=1).clamp(min=1).float()
                    d_x = (push_x.sum(dim=1) / overlap_count) * 0.65
                    d_y = (push_y.sum(dim=1) / overlap_count) * 0.65
                    delta[:, 0].index_add_(0, idx, d_x)
                    delta[:, 1].index_add_(0, idx, d_y)

                if j >= N:
                    break
                i += step

            if not any_overlap:
                break
            positions = positions + delta

    return positions


def _analytic_wirelength_solve(N, pin_features, edge_list, device, num_iters=15, ridge=0.05, eps=1e-4):
    """Solve the unconstrained (overlap-ignored) wirelength optimum in closed form.

    The loss is a sum of smooth-L1 terms and separable in x and y, so IRLS
    applies: each iteration fixes the current per-edge distances and solves the
    weighted least-squares problem that majorizes them, which is one linear
    solve against the graph Laplacian. The ridge term keeps the (otherwise
    singular) system well-conditioned.

    Args:
        N: Number of cells
        pin_features: [P, 7] tensor with pin information
        edge_list: [E, 2] tensor with edges
        device: Torch device to build tensors on
        num_iters: IRLS iterations
        ridge: Damping weight toward the previous iterate
        eps: Smoothing constant matching the wirelength loss

    Returns:
        [N, 2] tensor of optimal cell centers, ignoring overlap

    Note: reaches the same optimum as Adam (0.39 on Test 1) in ~3ms rather than
    ~1s. The objective is convex, so this is the global minimum.
    """
    cell_indices = pin_features[:, 0].long()
    src_pins = edge_list[:, 0].long()
    tgt_pins = edge_list[:, 1].long()
    a = cell_indices[src_pins]
    b = cell_indices[tgt_pins]

    # Same-cell edges are a placement-independent constant, so drop them
    valid = a != b
    a = a[valid]
    b = b[valid]
    kx = pin_features[src_pins, 1][valid] - pin_features[tgt_pins, 1][valid]
    ky = pin_features[src_pins, 2][valid] - pin_features[tgt_pins, 2][valid]

    x = torch.zeros(N, device=device)
    y = torch.zeros(N, device=device)
    eye_ridge = ridge * torch.eye(N, device=device)

    if a.shape[0] == 0:
        return torch.stack([x, y], dim=1)

    for _ in range(num_iters):
        dx = x[a] - x[b] + kx
        dy = y[a] - y[b] + ky
        w_x = 1.0 / (2.0 * torch.sqrt(dx ** 2 + eps))
        w_y = 1.0 / (2.0 * torch.sqrt(dy ** 2 + eps))

        for coord, w, k, cur in (("x", w_x, kx, x), ("y", w_y, ky, y)):
            L = torch.zeros(N, N, device=device)
            L.index_put_((a, a), w, accumulate=True)
            L.index_put_((b, b), w, accumulate=True)
            L.index_put_((a, b), -w, accumulate=True)
            L.index_put_((b, a), -w, accumulate=True)
            rhs = torch.zeros(N, device=device)
            rhs.index_add_(0, a, -w * k)
            rhs.index_add_(0, b, w * k)
            solved = torch.linalg.solve(L + eye_ridge, rhs + ridge * cur)
            if coord == "x":
                x = solved
            else:
                y = solved

    return torch.stack([x, y], dim=1)


def _pack_std_cluster(cell_features, order_key, std_idx, aspect_w=1.0, gap=0.02):
    """Row-pack standard cells into one tight, ~square, overlap-free cluster.

    Standard cells hold most of the pins but almost none of the area (on Test
    10: 6.7% of area, 89% of pins, 80% of edges), so packing them into one
    small block directly shortens the majority of nets. Rows are assigned by
    `order_key` y and filled by `order_key` x to keep connected cells together.

    Args:
        cell_features: [N, 6] tensor with cell properties
        order_key: [N, 2] tensor of desired positions used for ordering
        std_idx: Indices of the standard cells to pack
        aspect_w: Width multiplier controlling the block's aspect ratio
        gap: Separation between adjacent cells

    Returns:
        Tuple of ((indices, xs, ys), block_width, block_height), with the block
        centered on the origin by its true bounding box
    """
    device = cell_features.device
    w_all, h_all = cell_features[:, 4], cell_features[:, 5]
    if std_idx.numel() == 0:
        return None, 0.0, 0.0

    sw, sh = w_all[std_idx], h_all[std_idx]
    target_w = ((sw * sh).sum().item() ** 0.5) * aspect_w

    ky, kx = order_key[std_idx, 1], order_key[std_idx, 0]
    order = torch.argsort(ky)
    wg_ordered = sw[order] + gap
    row_of = (torch.cumsum(wg_ordered, dim=0) / max(target_w, 1e-6)).floor().long()
    fin = torch.argsort(row_of.float() * 1e6 + kx[order])

    idx_sorted = std_idx[order][fin]
    row_sorted = row_of[fin]
    widths = w_all[idx_sorted]

    # Vectorized per-row x cursor: running width sum, rebased at each row start
    wg = widths + gap
    left_global = torch.cumsum(wg, dim=0) - wg
    first_in_row = torch.ones_like(row_sorted, dtype=torch.bool)
    if row_sorted.numel() > 1:
        first_in_row[1:] = row_sorted[1:] != row_sorted[:-1]
    row_ordinal = torch.cumsum(first_in_row.long(), dim=0) - 1
    row_base = left_global[torch.where(first_in_row)[0]][row_ordinal]
    xs = (left_global - row_base) + widths / 2.0
    ys = row_sorted.float() * (sh.max().item() + gap)

    heights = h_all[idx_sorted]
    x0, x1 = (xs - widths / 2).min(), (xs + widths / 2).max()
    y0, y1 = (ys - heights / 2).min(), (ys + heights / 2).max()
    xs = xs - (x0 + x1) / 2
    ys = ys - (y0 + y1) / 2
    return (idx_sorted, xs, ys), (x1 - x0).item(), (y1 - y0).item()


def _compact_construct(cell_features, pin_features, edge_list, order_key,
                       macro_mask, std_mask, aspect_w=1.0, gap=0.02, macro_order=None):
    """Construct a complete, overlap-free placement directly (no gradient steps).

    Standard cells go into one tight central cluster, then macros are placed
    one at a time at the flush-contact position minimizing their own incident
    wirelength against everything placed so far.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin information
        edge_list: [E, 2] tensor with edges
        order_key: [N, 2] tensor of desired positions used for ordering
        macro_mask: [N] bool tensor selecting macros
        std_mask: [N] bool tensor selecting standard cells
        aspect_w: Width multiplier for the standard-cell cluster
        gap: Separation between adjacent cells
        macro_order: Optional explicit macro placement sequence

    Returns:
        [N, 2] tensor of overlap-free cell centers

    Note: the greedy is order-sensitive, so callers search over `macro_order`
    (worth 0.380 -> 0.334 on Test 6, 0.370 -> 0.328 on Test 8).
    """
    device = cell_features.device
    pos = torch.zeros_like(order_key)
    w_all, h_all = cell_features[:, 4], cell_features[:, 5]
    std_idx = torch.where(std_mask)[0]
    macro_idx = torch.where(macro_mask)[0]

    cluster, cw, ch = _pack_std_cluster(cell_features, order_key, std_idx, aspect_w, gap)
    if cluster is not None:
        pos[cluster[0], 0] = cluster[1]
        pos[cluster[0], 1] = cluster[2]

    placed_rects = []
    is_placed = torch.zeros(cell_features.shape[0], dtype=torch.bool, device=device)
    if cw > 0:
        placed_rects.append((0.0, 0.0, cw + gap, ch + gap))
        is_placed[std_idx] = True

    cidx = pin_features[:, 0].long()
    s_pin, t_pin = edge_list[:, 0].long(), edge_list[:, 1].long()
    s_cell, t_cell = cidx[s_pin], cidx[t_pin]

    if macro_order is None:
        placement_seq = macro_idx[torch.argsort(cell_features[macro_idx, 1], descending=True)].tolist()
    else:
        placement_seq = [int(i) for i in macro_order]
    for m in placement_seq:
        mw, mh = w_all[m].item(), h_all[m].item()

        incident = (s_cell == m) ^ (t_cell == m)
        own_pin = oth_pin = oth_cell = None
        if incident.any():
            sm = (s_cell == m)[incident]
            a_pin, b_pin = s_pin[incident], t_pin[incident]
            own_pin = torch.where(sm, a_pin, b_pin)
            oth_pin = torch.where(sm, b_pin, a_pin)
            oth_cell = cidx[oth_pin]
            keep = is_placed[oth_cell]
            own_pin, oth_pin, oth_cell = own_pin[keep], oth_pin[keep], oth_cell[keep]
            if own_pin.numel() == 0:
                own_pin = None

        if own_pin is not None:
            own_dx, own_dy = pin_features[own_pin, 1], pin_features[own_pin, 2]
            oth_x = pos[oth_cell, 0] + pin_features[oth_pin, 1]
            oth_y = pos[oth_cell, 1] + pin_features[oth_pin, 2]

        if not placed_rects:
            pos[m, 0] = pos[m, 1] = 0.0
            placed_rects.append((0.0, 0.0, mw, mh))
            is_placed[m] = True
            continue

        cands = []
        for (px, py, pw, ph) in placed_rects:
            for sx in (1, -1):
                cx = px + sx * ((pw + mw) / 2 + gap)
                for cy in (py + (ph - mh) / 2, py, py - (ph - mh) / 2):
                    cands.append((cx, cy))
            for sy in (1, -1):
                cy = py + sy * ((ph + mh) / 2 + gap)
                for cx in (px + (pw - mw) / 2, px, px - (pw - mw) / 2):
                    cands.append((cx, cy))

        best, best_cost = None, float("inf")
        for (cx, cy) in cands:
            blocked = False
            for (px, py, pw, ph) in placed_rects:
                if abs(cx - px) < (mw + pw) / 2 - 1e-6 and abs(cy - py) < (mh + ph) / 2 - 1e-6:
                    blocked = True
                    break
            if blocked:
                continue
            if own_pin is not None:
                cost = ((cx + own_dx - oth_x).abs() + (cy + own_dy - oth_y).abs()).sum().item()
            else:
                cost = (cx * cx + cy * cy) ** 0.5
            if cost < best_cost:
                best_cost, best = cost, (cx, cy)

        if best is None:
            # No flush spot fit; fall back to clearly outside everything placed.
            far = max(abs(px) + pw for (px, py, pw, ph) in placed_rects)
            best = (far + mw, 0.0)
        pos[m, 0], pos[m, 1] = best
        placed_rects.append((best[0], best[1], mw, mh))
        is_placed[m] = True

    return pos


def overlap_repulsion_loss(cell_features, pin_features, edge_list, margin=0.05, epoch=None, num_epochs=None):
    """Calculate loss to prevent cell overlaps using direct 2D pairwise tensors.

    Exact O(N^2) implementation, used up to EXACT_OVERLAP_MAX_N cells. Above
    that an N x N tensor would need tens of GB, so train_placement calls
    `_dsp_overlap_backward` for gradients instead and this returns a no-grad
    chunked count for inspection only.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information
        edge_list: [E, 2] tensor with edges
        margin: Extra separation margin to ensure zero overlap
        epoch: Current optimization epoch index
        num_epochs: Total number of optimization epochs

    Returns:
        Scalar loss value (0 when no overlaps exist)
    """
    N = cell_features.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=cell_features.device, requires_grad=True)

    if N > EXACT_OVERLAP_MAX_N:
        # Informational only; train_placement uses _dsp_overlap_backward here
        positions = cell_features[:, 2:4]
        widths = cell_features[:, 4]
        heights = cell_features[:, 5]
        num_bins = max(32, int((N / 8.0) ** 0.5))
        block = max(512, int(2.5 * N / num_bins))
        count = _chunked_overlap_count(positions.detach(), widths, heights, num_bins, block)
        return torch.tensor(float(count), device=cell_features.device, requires_grad=True)

    # Extract cell center positions (X, Y), widths, heights, and areas
    positions = cell_features[:, 2:4]  # [N, 2]
    widths = cell_features[:, 4]       # [N]
    heights = cell_features[:, 5]      # [N]
    areas = cell_features[:, 0]        # [N]

    # Step 1: Pairwise X and Y center distances using 2D Tensors [N, N]
    dx = torch.abs(positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0))  # [N, N]
    dy = torch.abs(positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0))  # [N, N]

    # Step 2: Minimum required separation distance along X and Y axes
    min_sep_x = (widths.unsqueeze(1) + widths.unsqueeze(0)) / 2.0 + margin  # [N, N]
    min_sep_y = (heights.unsqueeze(1) + heights.unsqueeze(0)) / 2.0 + margin  # [N, N]

    # Step 3: Compute positive overlap amounts along X and Y axes using ReLU
    overlap_x = torch.relu(min_sep_x - dx)  # [N, N]
    overlap_y = torch.relu(min_sep_y - dy)  # [N, N]

    # Step 4: Compute physical 2D overlap area
    overlap_area = overlap_x * overlap_y  # [N, N]

    # Step 5: Compute relative boundary overlap ratio
    min_w = torch.minimum(widths.unsqueeze(1), widths.unsqueeze(0))
    min_h = torch.minimum(heights.unsqueeze(1), heights.unsqueeze(0))
    rel_overlap = (overlap_x / min_w) * (overlap_y / min_h)  # [N, N]

    # Step 6: Exclude self-overlap along the diagonal
    diagonal_mask = ~torch.eye(N, dtype=torch.bool, device=cell_features.device)
    active_overlap_mask = (overlap_area > 0) & diagonal_mask

    if not active_overlap_mask.any():
        return torch.tensor(0.0, device=cell_features.device, requires_grad=True)

    # Step 7: Area weighting
    area_weights = torch.sqrt(areas.unsqueeze(1) * areas.unsqueeze(0))

    # Step 8: Combine linear + quadratic area overlap penalty with relative ejection push
    loss_matrix = (5.0 * overlap_area + overlap_area ** 2.0 + 10.0 * rel_overlap) * area_weights
    total_loss = torch.sum(loss_matrix[active_overlap_mask])

    return total_loss / 20.0


def _optimize_epochs(
    cell_positions, cell_features, pin_features, edge_list,
    widths_const, heights_const, areas_const,
    use_dsp, dsp_num_bins, dsp_block,
    effective_lr, effective_margin, effective_lambda_wl, lambda_overlap, design_scale,
    max_epochs, deadline=None, verbose=False, log_interval=100,
):
    """Run the gradient-descent refinement loop from a starting placement.

    Args:
        cell_positions: [N, 2] leaf tensor of cell centers (requires_grad)
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin information
        edge_list: [E, 2] tensor with edges
        widths_const, heights_const, areas_const: [N] cell size/area tensors
        use_dsp: Use the chunked sparse-pairs overlap loss instead of the exact one
        dsp_num_bins, dsp_block: Spatial hashing parameters for that path
        effective_lr: Adam learning rate
        effective_margin: Extra separation margin in the overlap loss
        effective_lambda_wl: Wirelength loss weight
        lambda_overlap: Base overlap penalty weight
        design_scale: Size-based multiplier on the overlap penalty
        max_epochs: Epoch cap (may be a large run-until-deadline sentinel)
        deadline: Optional time.time() value at which to stop
        verbose: Print per-epoch losses
        log_interval: Epochs between prints

    Returns:
        Dictionary with best_positions/best_wl (lowest-wirelength overlap-free
        epoch seen), last_positions, achieved_zero_overlap, loss_history and
        epochs_run
    """
    import time

    optimizer = optim.Adam([cell_positions], lr=effective_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0.001)

    loss_history = {"total_loss": [], "wirelength_loss": [], "overlap_loss": []}
    best_positions = cell_positions.detach().clone()
    best_wl = float("inf")
    lambda_overlap_frozen = None

    # With a run-until-deadline sentinel the schedules, keyed off
    # epoch/max_epochs, would barely move. Measure the real per-epoch cost over
    # a short window, then re-target max_epochs so the anneal spans the run.
    calibrated = deadline is None
    loop_start_time = time.time() if not calibrated else None
    calibration_epochs = 30

    epoch = 0
    for epoch in range(max_epochs):
        if deadline is not None and epoch > 0 and epoch % 3 == 0 and time.time() > deadline:
            break

        if not calibrated and epoch == calibration_epochs:
            elapsed = time.time() - loop_start_time
            per_epoch_cost = elapsed / calibration_epochs
            remaining_time = max(0.0, deadline - time.time())
            estimated_remaining = max(50, int(remaining_time / max(per_epoch_cost, 1e-6) * 0.85))
            max_epochs = epoch + estimated_remaining
            for group in optimizer.param_groups:
                group["lr"] = effective_lr
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=estimated_remaining, eta_min=0.001
            )
            calibrated = True

        optimizer.zero_grad()

        progress = epoch / max_epochs
        if lambda_overlap_frozen is not None:
            current_lambda_overlap = lambda_overlap_frozen
        else:
            current_lambda_overlap = lambda_overlap * design_scale * (1.0 + 120.0 * (progress ** 1.8))

        cell_features_current = cell_features.clone()
        cell_features_current[:, 2:4] = cell_positions

        wl_loss = wirelength_attraction_loss(
            cell_features_current, pin_features, edge_list
        )
        scaled_wl_loss = effective_lambda_wl * wl_loss

        if use_dsp:
            scaled_wl_loss.backward()
            # Alternate bin ordering by epoch parity to cover both blind spots
            overlap_loss_value, has_overlap = _dsp_overlap_backward(
                cell_positions, widths_const, heights_const, areas_const,
                margin=effective_margin, lambda_overlap=current_lambda_overlap,
                num_bins=dsp_num_bins, block=dsp_block, col_major=(epoch % 2 == 1),
            )
            total_loss_value = scaled_wl_loss.item() + overlap_loss_value
        else:
            overlap_loss = overlap_repulsion_loss(
                cell_features_current, pin_features, edge_list, margin=effective_margin
            )
            total_loss = scaled_wl_loss + current_lambda_overlap * overlap_loss
            total_loss.backward()
            total_loss_value = total_loss.item()
            has_overlap = overlap_loss.item() > 0.0
            overlap_loss_value = overlap_loss.item()

        torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=15.0)

        optimizer.step()
        scheduler.step()

        loss_history["total_loss"].append(total_loss_value)
        loss_history["wirelength_loss"].append(wl_loss.item())
        loss_history["overlap_loss"].append(overlap_loss_value)

        if not has_overlap:
            if lambda_overlap_frozen is None:
                # Relax the penalty once overlap is resolved, so the remaining
                # epochs pull wirelength down instead of pushing cells apart
                lambda_overlap_frozen = max(current_lambda_overlap * 0.2, lambda_overlap * design_scale)
                # Restart the LR schedule over the remaining epochs, since the
                # original would already be decayed by this point
                remaining = max(1, max_epochs - epoch)
                for group in optimizer.param_groups:
                    group["lr"] = effective_lr
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=remaining, eta_min=0.0005
                )
            current_wl = wl_loss.item()
            if current_wl < best_wl:
                best_wl = current_wl
                best_positions = cell_positions.detach().clone()

        if verbose and (epoch % log_interval == 0 or epoch == max_epochs - 1):
            print(f"Epoch {epoch}/{max_epochs}:")
            print(f"  Total Loss: {total_loss_value:.6f}")
            print(f"  Wirelength Loss: {wl_loss.item():.6f}")
            print(f"  Overlap Loss: {overlap_loss_value:.6f}")

    return {
        "best_positions": best_positions,
        "best_wl": best_wl,
        "last_positions": cell_positions.detach().clone(),
        "achieved_zero_overlap": best_wl < float("inf"),
        "loss_history": loss_history,
        "epochs_run": epoch + 1,
    }


def train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=600,
    lr=0.25,
    lambda_wirelength=1.0,
    lambda_overlap=2.0,
    verbose=True,
    log_interval=100,
):
    """Train placement optimization by constructing a legal layout, then refining it.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information
        edge_list: [E, 2] tensor with edges
        num_epochs: Baseline epoch budget (adapted per design size)
        lr: Baseline learning rate (adapted per design size)
        lambda_wirelength: Wirelength loss weight
        lambda_overlap: Base overlap penalty weight
        verbose: Print per-epoch losses
        log_interval: Epochs between prints

    Returns:
        Dictionary with final_cell_features, initial_cell_features and loss_history

    Steps:
        1. Contract connected cells together for a connectivity-aware layout
        2. Solve the unconstrained wirelength optimum to order cells by demand
        3. Construct a complete overlap-free placement, searching macro orders
        4. Refine with gradient descent, keeping the better of the two results
        5. Legalize as a safety net, guaranteeing exact zero overlap
    """
    torch.set_num_threads(8)

    cell_features = cell_features.clone()
    initial_cell_features = cell_features.clone()

    N = cell_features.shape[0]
    widths_const = cell_features[:, 4]
    heights_const = cell_features[:, 5]
    areas_const = cell_features[:, 0]
    macro_mask = cell_features[:, 0] >= MIN_MACRO_AREA
    std_mask = ~macro_mask

    # Initialize positions from topology, then construct a legal layout
    cell_positions = cell_features[:, 2:4].clone().detach()

    # Step 1: Centroid contraction pass to pull connected cells together
    if edge_list.shape[0] > 0:
        cell_indices = pin_features[:, 0].long()
        src_pins = edge_list[:, 0].long()
        tgt_pins = edge_list[:, 1].long()
        src_cells = cell_indices[src_pins]
        tgt_cells = cell_indices[tgt_pins]

        valid_mask = src_cells != tgt_cells
        s_c = src_cells[valid_mask]
        t_c = tgt_cells[valid_mask]

        for _ in range(8):
            neighbor_sum = torch.zeros_like(cell_positions)
            degree = torch.zeros(N, 1, device=cell_positions.device)
            neighbor_sum.index_add_(0, s_c, cell_positions[t_c])
            degree.index_add_(0, s_c, torch.ones(s_c.shape[0], 1, device=cell_positions.device))

            mask_has_deg = (degree > 0).squeeze()
            centroid = neighbor_sum[mask_has_deg] / degree[mask_has_deg]
            cell_positions[mask_has_deg] = 0.5 * cell_positions[mask_has_deg] + 0.5 * centroid

    total_area = cell_features[:, 0].sum().item()
    use_dsp = N > EXACT_OVERLAP_MAX_N
    import time
    train_start_time = time.time()

    # Step 2: Order key -- where each cell wants to sit. Use the exact
    # unconstrained optimum where affordable; above IRLS_MAX_N its dense N x N
    # solve is not (N=100k would need ~40 GB), so use the contracted positions.
    IRLS_MAX_N = 3000

    if N <= IRLS_MAX_N:
        order_key = _analytic_wirelength_solve(
            N, pin_features, edge_list, device=cell_positions.device,
            num_iters=15, ridge=0.05,
        )
    else:
        order_key = cell_positions.detach().clone()

    # Step 3: Randomized restart search over (macro order, cluster aspect). A
    # construction costs milliseconds, and the arrangement matters (searching
    # orders beat the fixed pin-count order by 11-12% on Tests 6 and 8). Trial
    # 0 is the deterministic default, so the search can only improve on it.
    aspect_candidates = (0.7, 1.0, 1.4) if N > 20000 else (0.5, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7, 2.0)

    if N <= DENSE_LEGALIZE_MAX_N:
        legality_bins, legality_block = 1, N
    else:
        legality_bins = max(16, int((N / 6.0) ** 0.5))
        legality_block = max(512, int(4.0 * N / legality_bins))

    macro_positions = torch.where(macro_mask)[0]
    num_macros = int(macro_positions.numel())
    rng = torch.Generator(device="cpu")
    rng.manual_seed(12345)

    # Bounded by trial count, not wall clock, so results are reproducible on a
    # loaded machine; the time cap is only a safety net. Counts are sized to
    # per-trial cost, which grows with N.
    if N <= 300:
        max_trials = 600
    elif N <= DENSE_LEGALIZE_MAX_N:
        max_trials = 250
    else:
        max_trials = 20
    search_deadline = train_start_time + (4.0 if N > 20000 else 2.0)
    best_construct, best_construct_wl = None, float("inf")
    trial = 0
    while True:
        if trial == 0 or num_macros < 2:
            order, aspect_w = None, 1.0
        else:
            order = macro_positions[torch.randperm(num_macros, generator=rng)]
            aspect_w = aspect_candidates[
                int(torch.randint(len(aspect_candidates), (1,), generator=rng))
            ]

        cand = _compact_construct(
            cell_features, pin_features, edge_list, order_key,
            macro_mask, std_mask, aspect_w=aspect_w, macro_order=order,
        )
        probe = cell_features.clone()
        probe[:, 2:4] = cand
        with torch.no_grad():
            cand_wl = wirelength_attraction_loss(probe, pin_features, edge_list).item()
        if cand_wl < best_construct_wl and _chunked_overlap_count(
            cand, widths_const, heights_const, legality_bins, legality_block
        ) == 0:
            best_construct_wl, best_construct = cand_wl, cand

        trial += 1
        if best_construct is not None and (
            trial >= max_trials or num_macros < 2 or time.time() > search_deadline
        ):
            break

    constructed_positions = best_construct.detach().clone()
    cell_positions = constructed_positions.clone().requires_grad_(True)

    # Step 4: Size-adaptive learning rate and epoch budget
    if N <= 500:
        effective_lr = 0.22
        effective_epochs = max(num_epochs, 1200)
    elif N <= EXACT_OVERLAP_MAX_N:
        effective_lr = 0.3
        effective_epochs = max(num_epochs, 800)
    elif N <= 20000:
        # Test 11. LR stays small relative to the construction's tight pitch:
        # Adam's first step is ~lr regardless of gradient size, and a large one
        # shoves cells through each other, spending the budget re-resolving
        # overlap instead of refining wirelength.
        effective_lr = 0.06
        effective_epochs = 150
    else:
        # Test 12. Same small-LR reasoning as Test 11.
        effective_lr = 0.03
        # Refinement is skipped entirely here: it cost 60s+ and the portfolio
        # below discarded its output every time, giving a bit-identical result
        # with 20 epochs, 12, or none.
        effective_epochs = 0

    # Must stay below the gap `_compact_construct` packs at (0.02), or the loss
    # reads the whole valid layout as overlapping and blows it apart
    effective_margin = 0.01
    effective_lambda_wl = 3.5 if N <= 500 else lambda_wirelength

    dsp_num_bins = max(16, int((N / 6.0) ** 0.5))
    dsp_block = min(N, max(256, int(2.5 * N / dsp_num_bins)))

    design_scale = (N / 20.0) ** 0.5

    optimize_kwargs = dict(
        widths_const=widths_const, heights_const=heights_const, areas_const=areas_const,
        use_dsp=use_dsp, dsp_num_bins=dsp_num_bins, dsp_block=dsp_block,
        effective_lr=effective_lr, effective_margin=effective_margin,
        effective_lambda_wl=effective_lambda_wl, lambda_overlap=lambda_overlap,
        design_scale=design_scale,
    )

    # One deep run rather than many short restarts (measured worse for the
    # same budget), bounded by wall clock since per-epoch cost varies widely
    # with N (0.3ms at N=22 vs 3.2ms at N=208).
    if not use_dsp:
        # A bounded polish on an already-legal construction: worth a few
        # percent on the smallest designs, roughly nothing on the larger ones
        deadline = train_start_time + 2.6
        max_epochs = 200000  # effectively "run until the deadline"
    else:
        # DSP path keeps its fixed, already-tuned epoch count
        deadline = None
        max_epochs = effective_epochs

    refine_result = _optimize_epochs(
        cell_positions, cell_features, pin_features, edge_list,
        max_epochs=max_epochs, deadline=deadline,
        verbose=verbose, log_interval=log_interval,
        **optimize_kwargs,
    )

    best_positions = refine_result["best_positions"] if refine_result["achieved_zero_overlap"] else refine_result["last_positions"]
    best_wl = refine_result["best_wl"]
    loss_history = refine_result["loss_history"]

    # Step 5: Keep whichever of the construction and the refinement scores
    # better, so refinement can only ever help
    def _wl_of(p):
        probe = cell_features.clone()
        probe[:, 2:4] = p
        with torch.no_grad():
            return wirelength_attraction_loss(probe, pin_features, edge_list).item()

    def _is_legal(p):
        if N <= DENSE_LEGALIZE_MAX_N:
            bins, block = 1, N
        else:
            bins = max(16, int((N / 6.0) ** 0.5))
            block = max(512, int(4.0 * N / bins))
        return (_chunked_overlap_count(p, widths_const, heights_const, bins, block, col_major=False) == 0
                and _chunked_overlap_count(p, widths_const, heights_const, bins, block, col_major=True) == 0)

    # Legality gates the comparison: overlapping cells have shorter wires, so
    # picking on wirelength alone would reward an illegal refinement
    # Skip the comparison when no refinement ran -- the legality probe is
    # expensive at N=100k and the construction is already legal
    if max_epochs > 0:
        if not _is_legal(best_positions) or _wl_of(constructed_positions) < _wl_of(best_positions):
            best_positions = constructed_positions
    else:
        best_positions = constructed_positions

    final_cell_features = cell_features.clone()

    # Step 6: Legalization safety net, guaranteeing exact zero overlap. Passing
    # block == N degenerates the windowed helpers to an exact dense sweep,
    # matching the ground-truth check in calculate_cells_with_overlaps. Only
    # Test 12 is too large for that and uses real windowing.
    chosen_positions = best_positions

    if N <= DENSE_LEGALIZE_MAX_N:
        check_bins, check_block = 1, N
    else:
        check_bins = max(16, int((N / 6.0) ** 0.5))
        check_block = max(512, int(4.0 * N / check_bins))

    # Check and legalize under both bin orderings, repeating a few times since
    # each pass can shift cells into new configurations. Deadline-bounded so a
    # slow machine cannot add uncapped time on top of the refinement budget.
    legalize_safety_deadline = train_start_time + 9.0
    for _ in range(4):
        overlap_row = _chunked_overlap_count(
            chosen_positions, widths_const, heights_const, check_bins, check_block, col_major=False
        )
        overlap_col = _chunked_overlap_count(
            chosen_positions, widths_const, heights_const, check_bins, check_block, col_major=True
        )
        if (overlap_row == 0 and overlap_col == 0) or time.time() > legalize_safety_deadline:
            break
        chosen_positions = _legalize_large_placement(
            chosen_positions, widths_const, heights_const, check_bins, check_block, max_iters=150,
            deadline=legalize_safety_deadline,
        )

    final_cell_features[:, 2:4] = chosen_positions

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
