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

import math
import os
from collections import defaultdict
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
    num_pins_per_cell = torch.zeros(total_cells, dtype=torch.long)

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
    total_pins = int(num_pins_per_cell.sum().item())
    pin_features = torch.zeros(total_pins, 7)

    # Fixed pin size for all pins (square pins)
    PIN_SIZE = 0.1  # All pins are 0.1 x 0.1

    # Use vectorized pin generation for scalability on very large designs.
    pin_to_cell = torch.repeat_interleave(
        torch.arange(total_cells, dtype=torch.long), num_pins_per_cell
    )
    pin_features[:, PinFeatureIdx.CELL_IDX] = pin_to_cell.float()

    margin = PIN_SIZE / 2.0
    widths_for_pin = cell_widths[pin_to_cell]
    heights_for_pin = cell_heights[pin_to_cell]

    x_span = torch.clamp(widths_for_pin - 2.0 * margin, min=0.0)
    y_span = torch.clamp(heights_for_pin - 2.0 * margin, min=0.0)

    pin_x = torch.rand(total_pins) * x_span + margin
    pin_y = torch.rand(total_pins) * y_span + margin

    # For tiny cells where span is 0, center pins on that axis.
    tiny_x = x_span == 0.0
    tiny_y = y_span == 0.0
    pin_x[tiny_x] = widths_for_pin[tiny_x] * 0.5
    pin_y[tiny_y] = heights_for_pin[tiny_y] * 0.5

    pin_features[:, PinFeatureIdx.PIN_X] = pin_x
    pin_features[:, PinFeatureIdx.PIN_Y] = pin_y
    pin_features[:, PinFeatureIdx.X] = pin_x
    pin_features[:, PinFeatureIdx.Y] = pin_y
    pin_features[:, PinFeatureIdx.WIDTH] = PIN_SIZE
    pin_features[:, PinFeatureIdx.HEIGHT] = PIN_SIZE

    # Step 7: Generate edges with simple random connectivity
    # Each pin connects to 1-3 random pins (preferring different cells)
    edge_list = []
    avg_edges_per_pin = 2.0

    if total_pins > 1:
        # Generate 1-3 candidate edges per pin in a vectorized way, then deduplicate.
        num_connections = torch.randint(1, 4, (total_pins,), dtype=torch.long)
        src = torch.repeat_interleave(torch.arange(total_pins, dtype=torch.long), num_connections)
        tgt = torch.randint(0, total_pins, (src.shape[0],), dtype=torch.long)

        valid = src != tgt
        src = src[valid]
        tgt = tgt[valid]

        # Prefer inter-cell connectivity.
        inter_cell = pin_to_cell[src] != pin_to_cell[tgt]
        src = src[inter_cell]
        tgt = tgt[inter_cell]

        edge_u = torch.minimum(src, tgt)
        edge_v = torch.maximum(src, tgt)
        not_self = edge_u != edge_v

        if torch.any(not_self):
            edge_list = torch.stack([edge_u[not_self], edge_v[not_self]], dim=1)
            edge_list = torch.unique(edge_list, dim=0)
        else:
            edge_list = torch.zeros((0, 2), dtype=torch.long)
    else:
        edge_list = torch.zeros((0, 2), dtype=torch.long)

    print(f"\nGenerated placement data:")
    print(f"  Total cells: {total_cells}")
    print(f"  Total pins: {total_pins}")
    print(f"  Total edges: {len(edge_list)}")
    print(f"  Average edges per pin: {2 * len(edge_list) / total_pins:.2f}")

    return cell_features, pin_features, edge_list

# ======= OPTIMIZATION CODE (edit this part) =======

def wirelength_attraction_loss(
    cell_features_or_positions,
    pin_features_or_cell_indices,
    edge_list_or_pin_offsets,
    maybe_edge_list=None,
):
    if maybe_edge_list is None:
        cell_features = cell_features_or_positions
        pin_features = pin_features_or_cell_indices
        edge_list = edge_list_or_pin_offsets
        cell_positions = cell_features[:, 2:4]
        pin_cell_indices = pin_features[:, 0].long()
        pin_offsets = pin_features[:, 1:3]
    else:
        cell_positions = cell_features_or_positions
        pin_cell_indices = pin_features_or_cell_indices.long()
        pin_offsets = edge_list_or_pin_offsets
        edge_list = maybe_edge_list

    if edge_list.shape[0] == 0:
        return cell_positions.sum() * 0.0

    pin_absolute = cell_positions[pin_cell_indices] + pin_offsets

    src_pins = edge_list[:, 0].long()
    tgt_pins = edge_list[:, 1].long()

    dx = torch.abs(pin_absolute[src_pins, 0] - pin_absolute[tgt_pins, 0])
    dy = torch.abs(pin_absolute[src_pins, 1] - pin_absolute[tgt_pins, 1])

    alpha = 0.1
    smooth_manhattan = alpha * torch.logaddexp(dx / alpha, dy / alpha)
    return smooth_manhattan.mean()


def overlap_repulsion_loss(
    cell_features_or_x,
    pin_features_or_y=None,
    edge_list_or_w=None,
    maybe_h=None,
):
    if maybe_h is None:
        cell_features = cell_features_or_x
        x = cell_features[:, CellFeatureIdx.X]
        y = cell_features[:, CellFeatureIdx.Y]
        w = cell_features[:, CellFeatureIdx.WIDTH]
        h = cell_features[:, CellFeatureIdx.HEIGHT]
    else:
        x = cell_features_or_x
        y = pin_features_or_y
        w = edge_list_or_w
        h = maybe_h
    
    N = x.shape[0]
    if N <= 1:
        return x.sum() * 0.0

    # Small designs: exact all-pairs overlap
    if N <= 2000:
        dx = torch.abs(x.unsqueeze(1) - x.unsqueeze(0))
        dy = torch.abs(y.unsqueeze(1) - y.unsqueeze(0))
        min_sep_x = 0.5 * (w.unsqueeze(1) + w.unsqueeze(0))
        min_sep_y = 0.5 * (h.unsqueeze(1) + h.unsqueeze(0))

        overlap_x = torch.relu(min_sep_x - dx)
        overlap_y = torch.relu(min_sep_y - dy)
        overlap_area = overlap_x * overlap_y

        mask = torch.triu(
            torch.ones((N, N), dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        ov_x = overlap_x[mask]
        ov_y = overlap_y[mask]
        ov_area = overlap_area[mask]
        if ov_area.numel() == 0:
            return x.sum() * 0.0

        active = (ov_x > 0) & (ov_y > 0)
        if not torch.any(active):
            return x.sum() * 0.0

        # Penetration term drives separation along the easier axis
        # area term will pinalise deep overlaps
        penetration = torch.minimum(ov_x[active], ov_y[active])
        return torch.mean(penetration**2 + 0.5 * ov_area[active])

    if N <= 20000:
        k = 48
    elif N <= 50000:
        k = 24
    else:
        k = 12
    k = min(k, N - 1)

    order = torch.argsort(x)
    x_s, y_s, w_s, h_s = x[order], y[order], w[order], h[order]

    base = torch.arange(N - k, device=x.device).unsqueeze(1)
    offs = torch.arange(1, k + 1, device=x.device).unsqueeze(0)
    i_s = base.expand(-1, k).reshape(-1)
    j_s = (base + offs).reshape(-1)

    dx = torch.abs(x_s[i_s] - x_s[j_s])
    dy = torch.abs(y_s[i_s] - y_s[j_s])

    min_sep_x = 0.5 * (w_s[i_s] + w_s[j_s])
    min_sep_y = 0.5 * (h_s[i_s] + h_s[j_s])

    overlap_x = torch.relu(min_sep_x - dx)
    overlap_y = torch.relu(min_sep_y - dy)
    ov_area = overlap_x * overlap_y
    active = (overlap_x > 0) & (overlap_y > 0)
    if not torch.any(active):
        return x.sum() * 0.0
    penetration = torch.minimum(overlap_x[active], overlap_y[active])
    return torch.mean(penetration**2 + 0.5 * ov_area[active])

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

    # Make only cell positions require gradients
    cell_positions = cell_features[:, 2:4].clone().detach()
    cell_positions.requires_grad_(True)

    N = cell_features.shape[0]
    E = edge_list.shape[0]

    widths = cell_features[:, CellFeatureIdx.WIDTH]
    heights = cell_features[:, CellFeatureIdx.HEIGHT]
    pin_cell_indices = pin_features[:, PinFeatureIdx.CELL_IDX].long()
    pin_offsets = pin_features[:, 1:3]

    # Scale defaults by design size to keep runtime practical.
    if num_epochs == 1000:
        if N <= 500:
            num_epochs = 800
        elif N <= 5000:
            num_epochs = 420
        elif N <= 20000:
            num_epochs = 180
        elif N <= 50000:
            num_epochs = 120
        else:
            num_epochs = 70

    if lr == 0.01:
        if N <= 5000:
            lr = 0.02
        elif N <= 20000:
            lr = 0.015
        elif N <= 50000:
            lr = 0.012
        else:
            lr = 0.01

    # Wirelength mini-batch for huge edge counts.
    if E <= 200000:
        edge_batch_size = E
    elif E <= 1000000:
        edge_batch_size = 150000
    else:
        edge_batch_size = 100000

    if N <= 20000:
        overlap_eval_interval = 1
    elif N <= 50000:
        overlap_eval_interval = 2
    else:
        overlap_eval_interval = 4

    # Create optimizer
    optimizer = optim.Adam([cell_positions], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(num_epochs, 1), eta_min=lr * 0.2
    )

    # Track loss history
    loss_history = {
        "total_loss": [],
        "wirelength_loss": [],
        "overlap_loss": [],
    }

    edge_perm = None
    edge_ptr = 0
    if E > edge_batch_size and edge_batch_size > 0:
        edge_perm = torch.randperm(E, device=edge_list.device)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Wirelength on full or sampled edge batch.
        if E > edge_batch_size and edge_batch_size > 0:
            if edge_ptr + edge_batch_size > E:
                edge_perm = torch.randperm(E, device=edge_list.device)
                edge_ptr = 0
            sampled = edge_perm[edge_ptr : edge_ptr + edge_batch_size]
            edge_ptr += edge_batch_size
            edge_batch = edge_list[sampled]
        else:
            edge_batch = edge_list

        wl_loss = wirelength_attraction_loss(
            cell_positions, pin_cell_indices, pin_offsets, edge_batch
        )

        if epoch % overlap_eval_interval == 0 or epoch == num_epochs - 1:
            overlap_loss = overlap_repulsion_loss(
                cell_positions[:, 0],
                cell_positions[:, 1],
                widths,
                heights,
            )
        else:
            overlap_loss = cell_positions.sum() * 0.0

        # Two-phase weighting: first kill overlap aggressively, then rebalance.
        progress = epoch / max(1, num_epochs - 1)
        if N <= 20000:
            if progress < 0.70:
                overlap_weight = lambda_overlap * 60.0
                wirelength_weight = lambda_wirelength * 0.10
            else:
                overlap_weight = lambda_overlap * 25.0
                wirelength_weight = lambda_wirelength * 0.40
        else:
            if progress < 0.40:
                overlap_weight = lambda_overlap * 25.0
                wirelength_weight = lambda_wirelength * 0.40
            elif progress < 0.80:
                overlap_weight = lambda_overlap * 15.0
                wirelength_weight = lambda_wirelength * 0.80
            else:
                overlap_weight = lambda_overlap * 8.0
                wirelength_weight = lambda_wirelength * 1.20

        # Combined loss
        total_loss = wirelength_weight * wl_loss + overlap_weight * overlap_loss

        # Backward pass
        total_loss.backward()

        # Gradient clipping to prevent extreme updates
        torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=5.0)

        # Update positions
        optimizer.step()
        scheduler.step()

        # Record losses
        loss_history["total_loss"].append(total_loss.item())
        loss_history["wirelength_loss"].append(wl_loss.item())
        loss_history["overlap_loss"].append(overlap_loss.item())

        # Log progress
        if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch}/{num_epochs}:")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  Wirelength Loss: {wl_loss.item():.6f}")
            print(f"  Overlap Loss: {overlap_loss.item():.6f}")

        # Early stopping for extremely large designs once wirelength converges.
        if N > 50000 and epoch >= 30 and epoch % 10 == 0:
            recent = loss_history["wirelength_loss"][-10:]
            prev = loss_history["wirelength_loss"][-20:-10]
            if prev:
                prev_mean = sum(prev) / len(prev)
                recent_mean = sum(recent) / len(recent)
                improvement = (prev_mean - recent_mean) / max(abs(prev_mean), 1e-9)
                if improvement < 0.002:
                    break

    # Final legalization pass prioritizing overlap-free placement.
    legalizer_gap = 0.01 if N > 5000 else 0.05
    legalized_positions = legalize_by_rows(
        cell_features, cell_positions.detach(), row_gap=legalizer_gap
    )

    # Create final cell features
    final_cell_features = cell_features.clone()
    final_cell_features[:, 2:4] = legalized_positions

    return {
        "final_cell_features": final_cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": loss_history,
    }


def legalize_by_rows(cell_features, cell_positions, row_gap=0.05):
    """Deterministic row-based legalizer that removes overlaps.

    The legalizer preserves x-order from the optimized placement to retain some
    wirelength structure, then packs cells into non-overlapping rows.
    """
    N = cell_features.shape[0]
    if N <= 1:
        return cell_positions.detach().clone()

    widths_t = cell_features[:, CellFeatureIdx.WIDTH]
    heights_t = cell_features[:, CellFeatureIdx.HEIGHT]
    x_orig_t = cell_positions[:, 0].detach()
    y_orig_t = cell_positions[:, 1].detach()

    widths = widths_t.tolist()
    heights = heights_t.tolist()
    x_orig = x_orig_t.tolist()
    y_orig = y_orig_t.tolist()

    # Compact-ish row width target from total area; larger factor gives fewer rows.
    total_area = float(torch.sum(widths_t * heights_t).item())
    max_width = float(torch.max(widths_t).item())
    target_row_width = max((total_area ** 0.5) * 2.0, max_width * 2.5)

    # Build rows by y-order first to preserve vertical locality,
    # then sort cells by x within each row.
    order_by_y = sorted(range(N), key=lambda idx: y_orig[idx])
    rows = []
    row_cells = []
    row_width = 0.0
    row_height = 0.0
    row_y_sum = 0.0

    for idx in order_by_y:
        wi = widths[idx]
        hi = heights[idx]

        proposed_width = wi if not row_cells else row_width + row_gap + wi
        if row_cells and proposed_width > target_row_width:
            rows.append((row_cells, row_height, row_y_sum / len(row_cells)))
            row_cells = [idx]
            row_width = wi
            row_height = hi
            row_y_sum = y_orig[idx]
        else:
            row_cells.append(idx)
            row_width = proposed_width
            row_height = max(row_height, hi)
            row_y_sum += y_orig[idx]

    if row_cells:
        rows.append((row_cells, row_height, row_y_sum / len(row_cells)))

    rows.sort(key=lambda item: item[2])

    x_new = [0.0] * N
    y_new = [0.0] * N
    row_base_y = 0.0

    for row_cells, row_height, _ in rows:
        row_cells.sort(key=lambda idx: x_orig[idx])
        used_width = sum(widths[idx] for idx in row_cells)
        if len(row_cells) > 1:
            used_width += row_gap * (len(row_cells) - 1)

        row_center_x = sum(x_orig[idx] for idx in row_cells) / len(row_cells)
        cursor_x = row_center_x - used_width / 2.0

        for idx in row_cells:
            wi = widths[idx]
            hi = heights[idx]
            x_new[idx] = cursor_x + wi / 2.0
            y_new[idx] = row_base_y + hi / 2.0
            cursor_x += wi + row_gap

        row_base_y += row_height + row_gap

    x_new = torch.tensor(x_new, dtype=cell_positions.dtype, device=cell_positions.device)
    y_new = torch.tensor(y_new, dtype=cell_positions.dtype, device=cell_positions.device)

    # Recenter around optimized centroid (translation-invariant for wirelength).
    x_new = x_new - x_new.mean() + x_orig_t.mean()
    y_new = y_new - y_new.mean() + y_orig_t.mean()

    return torch.stack([x_new, y_new], dim=1)


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

    # Spatial hashing gives exact overlap detection with near-linear behavior
    # on sparse/legalized placements.
    x = cell_features[:, CellFeatureIdx.X].detach().tolist()
    y = cell_features[:, CellFeatureIdx.Y].detach().tolist()
    widths = cell_features[:, CellFeatureIdx.WIDTH].detach().tolist()
    heights = cell_features[:, CellFeatureIdx.HEIGHT].detach().tolist()

    left = [x[i] - 0.5 * widths[i] for i in range(N)]
    right = [x[i] + 0.5 * widths[i] for i in range(N)]
    bottom = [y[i] - 0.5 * heights[i] for i in range(N)]
    top = [y[i] + 0.5 * heights[i] for i in range(N)]

    max_dims = [max(widths[i], heights[i]) for i in range(N)]
    median_dim = float(torch.median(torch.tensor(max_dims)).item()) if N > 0 else 1.0
    bin_size = max(1.0, median_dim * 1.5)

    grid = defaultdict(list)
    cells_with_overlaps = set()

    for i in range(N):
        gx0 = int(math.floor(left[i] / bin_size))
        gx1 = int(math.floor(right[i] / bin_size))
        gy0 = int(math.floor(bottom[i] / bin_size))
        gy1 = int(math.floor(top[i] / bin_size))

        candidates = set()
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                bucket = grid.get((gx, gy))
                if bucket:
                    candidates.update(bucket)

        for j in candidates:
            dx = abs(x[i] - x[j])
            dy = abs(y[i] - y[j])
            min_sep_x = 0.5 * (widths[i] + widths[j])
            min_sep_y = 0.5 * (heights[i] + heights[j])

            if dx < min_sep_x and dy < min_sep_y:
                cells_with_overlaps.add(i)
                cells_with_overlaps.add(j)

        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                grid[(gx, gy)].append(i)

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
