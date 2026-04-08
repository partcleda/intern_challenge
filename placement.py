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

_WL_ALPHA = 0.03


def wirelength_attraction_loss(cell_features, pin_features, edge_list):
    """Smooth Manhattan wirelength loss. Alpha controlled by _WL_ALPHA."""
    if edge_list.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True)

    cell_positions = cell_features[:, 2:4]
    cell_indices = pin_features[:, 0].long()

    pin_absolute_x = cell_positions[cell_indices, 0] + pin_features[:, 1]
    pin_absolute_y = cell_positions[cell_indices, 1] + pin_features[:, 2]

    src_pins = edge_list[:, 0].long()
    tgt_pins = edge_list[:, 1].long()

    dx = torch.abs(pin_absolute_x[src_pins] - pin_absolute_x[tgt_pins])
    dy = torch.abs(pin_absolute_y[src_pins] - pin_absolute_y[tgt_pins])

    a = _WL_ALPHA
    smooth_manhattan = a * torch.logsumexp(
        torch.stack([dx / a, dy / a], dim=0), dim=0
    )

    return torch.sum(smooth_manhattan) / edge_list.shape[0]


def overlap_repulsion_loss(cell_features, pin_features, edge_list):
    """Linear overlap area penalty — strong gradients at all overlap magnitudes.

    Gradient of overlap_area w.r.t. position is proportional to the perpendicular
    overlap dimension, giving consistent push-apart force for both large macro overlaps
    and small std-cell overlaps. Normalized by N (not N^2) to keep per-pair gradient
    strong when only a few overlaps remain.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor (unused)
        edge_list: [E, 2] tensor (unused)

    Returns:
        Scalar loss value, zero when no cells overlap.
    """
    N = cell_features.shape[0]
    if N <= 1:
        return torch.tensor(0.0, requires_grad=True)

    positions = cell_features[:, 2:4]
    widths = cell_features[:, 4]
    heights = cell_features[:, 5]

    dx = torch.abs(positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0))
    dy = torch.abs(positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0))

    half_w_sum = (widths.unsqueeze(1) + widths.unsqueeze(0)) * 0.5
    half_h_sum = (heights.unsqueeze(1) + heights.unsqueeze(0)) * 0.5

    overlap_x = torch.relu(half_w_sum - dx)
    overlap_y = torch.relu(half_h_sum - dy)
    overlap_area = overlap_x * overlap_y

    mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=cell_features.device), diagonal=1)
    masked_overlaps = overlap_area[mask]

    return masked_overlaps.sum() / N


def _spectral_initial_placement(cell_features, pin_features, edge_list, scale_exp=0.5):
    """Compute connectivity-aware initial positions via graph Laplacian eigenvectors."""
    N = cell_features.shape[0]
    if edge_list.shape[0] == 0 or N <= 2 or N > 500:
        return None

    pin_to_cell = pin_features[:, 0].long()
    src_cells = pin_to_cell[edge_list[:, 0].long()]
    tgt_cells = pin_to_cell[edge_list[:, 1].long()]

    valid = src_cells != tgt_cells
    src_cells = src_cells[valid]
    tgt_cells = tgt_cells[valid]
    if src_cells.shape[0] == 0:
        return None

    A = torch.zeros(N, N)
    ones = torch.ones(src_cells.shape[0])
    A.view(-1).scatter_add_(0, src_cells * N + tgt_cells, ones)
    A.view(-1).scatter_add_(0, tgt_cells * N + src_cells, ones)

    L = torch.diag(A.sum(dim=1)) - A

    try:
        eigvals, eigvecs = torch.linalg.eigh(L)
    except Exception:
        return None

    x_raw = eigvecs[:, 1]
    y_raw = eigvecs[:, 2]

    x_range = x_raw.max() - x_raw.min()
    y_range = y_raw.max() - y_raw.min()
    if x_range < 1e-10 or y_range < 1e-10:
        return None

    x_norm = (x_raw - x_raw.min()) / x_range - 0.5
    y_norm = (y_raw - y_raw.min()) / y_range - 0.5

    scale = cell_features[:, 0].sum().item() ** scale_exp
    return torch.stack([x_norm * scale, y_norm * scale], dim=1)


def _count_discrete_overlaps(positions, widths, heights):
    """Vectorized count of overlapping cell pairs (no Python loops)."""
    N = positions.shape[0]
    dx = torch.abs(positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0))
    dy = torch.abs(positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0))
    half_w = (widths.unsqueeze(1) + widths.unsqueeze(0)) * 0.5
    half_h = (heights.unsqueeze(1) + heights.unsqueeze(0)) * 0.5
    ovlp = torch.relu(half_w - dx) * torch.relu(half_h - dy)
    tri = torch.triu(torch.ones(N, N, dtype=torch.bool, device=positions.device), diagonal=1)
    return (ovlp[tri] > 1e-10).sum().item()


def _legalize_placement(positions, widths, heights, verbose=False):
    """Greedy cell-by-cell legalization guaranteeing zero overlap.

    Processes cells largest-first. For each conflicting cell, generates exact
    displacement candidates from overlapping neighbors (4 per neighbor: the
    minimum shift in each cardinal direction to separate), picks the closest
    valid one. Falls back to spiral search if needed.
    """
    import numpy as np

    N = positions.shape[0]
    pos = positions.detach().clone()
    px = pos[:, 0].numpy().copy()
    py = pos[:, 1].numpy().copy()
    w = widths.detach().numpy().copy()
    h = heights.detach().numpy().copy()

    areas = w * h
    order = np.argsort(-areas)

    placed_x = np.empty(N, dtype=np.float64)
    placed_y = np.empty(N, dtype=np.float64)
    placed_w = np.empty(N, dtype=np.float64)
    placed_h = np.empty(N, dtype=np.float64)
    n_placed = 0

    for idx in order:
        cx, cy = float(px[idx]), float(py[idx])
        cw, ch = float(w[idx]), float(h[idx])

        if n_placed == 0:
            placed_x[0], placed_y[0] = cx, cy
            placed_w[0], placed_h[0] = cw, ch
            n_placed = 1
            continue

        def _has_any_overlap(tx, ty):
            adx = np.abs(tx - placed_x[:n_placed])
            ady = np.abs(ty - placed_y[:n_placed])
            min_sx = (cw + placed_w[:n_placed]) * 0.5
            min_sy = (ch + placed_h[:n_placed]) * 0.5
            return np.any((adx < min_sx) & (ady < min_sy))

        if not _has_any_overlap(cx, cy):
            placed_x[n_placed], placed_y[n_placed] = cx, cy
            placed_w[n_placed], placed_h[n_placed] = cw, ch
            n_placed += 1
            continue

        adx = np.abs(cx - placed_x[:n_placed])
        ady = np.abs(cy - placed_y[:n_placed])
        min_sx = (cw + placed_w[:n_placed]) * 0.5
        min_sy = (ch + placed_h[:n_placed]) * 0.5
        conflicts = np.where((adx < min_sx) & (ady < min_sy))[0]

        candidates = []
        margin = 1e-3
        for k in conflicts:
            sep_x = min_sx[k] + margin
            sep_y = min_sy[k] + margin
            candidates.append((placed_x[k] + sep_x, cy))
            candidates.append((placed_x[k] - sep_x, cy))
            candidates.append((cx, placed_y[k] + sep_y))
            candidates.append((cx, placed_y[k] - sep_y))

        best_pos = None
        best_dist = float('inf')
        for tx, ty in candidates:
            d = (tx - cx) ** 2 + (ty - cy) ** 2
            if d < best_dist and not _has_any_overlap(tx, ty):
                best_dist = d
                best_pos = (tx, ty)

        if best_pos is None:
            for radius_step in range(1, 2000):
                step = 0.5 * radius_step
                found = False
                for sx in [-step, 0, step]:
                    for sy in [-step, 0, step]:
                        if sx == 0 and sy == 0:
                            continue
                        tx, ty = cx + sx, cy + sy
                        if not _has_any_overlap(tx, ty):
                            best_pos = (tx, ty)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        if best_pos is not None:
            px[idx], py[idx] = best_pos
        placed_x[n_placed] = px[idx]
        placed_y[n_placed] = py[idx]
        placed_w[n_placed] = cw
        placed_h[n_placed] = ch
        n_placed += 1

    result = pos.clone()
    result[:, 0] = torch.tensor(px, dtype=pos.dtype)
    result[:, 1] = torch.tensor(py, dtype=pos.dtype)
    return result


def _get_lr(epoch, warmup_epochs, total_epochs, peak_lr, min_lr=1e-4):
    """Cosine annealing with linear warmup."""
    if epoch < warmup_epochs:
        return peak_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


_DEFAULT_HP = {
    "wl_alpha": 0.03,
    "spectral_scale_exp": 0.4,
    "lam_ol_scale": 283.56,
    "lam_ol_maint_scale": 31.16,
    "lam_wl_full": 0.938,
    "lam_wl_sub": 0.123,
    "drift_weight": 0.0008,
    "annealing_threshold": 214,
    "grad_norm_base": 3.597,
    "grad_norm_exp": 0.557,
    "lr_floor_mult": 0.278,
    "anneal_ol_floor": 0.062,
    "anneal_ol_decay": 1.529,
    "anneal_wl_ramp": 1.512,
    "anneal_ol_gate": 0.042,
    "anneal_wl_ungated": 0.143,
    "p2_epochs": 3000,
    "p2_lam_mult": 3.0,
    "p4_epoch_base": 5000,
    "p4_epoch_nref": 50.0,
    "p4_lr": 0.0291,
    "p4_wl_guard": 24.827,
    "p4_guard_mult": 1.5,
    "p4_guard_max": 80.0,
    "p4_clip": 2.218,
    "p5_epochs": 500,
    "p5_lr": 0.00286,
    "p5_ol_weight": 10.895,
    "p5_clip": 1.6,
}


def _single_train_run(cell_features, pin_features, edge_list, use_spectral,
                      num_epochs, lr, verbose, log_interval, hp=None):
    """Run one full optimization pass: phases 1-5. Returns (final_cell_features, final_wl)."""
    global _WL_ALPHA
    hp = {**_DEFAULT_HP, **(hp or {})}
    _WL_ALPHA = hp["wl_alpha"]
    cell_features = cell_features.clone()
    N = cell_features.shape[0]
    widths = cell_features[:, CellFeatureIdx.WIDTH]
    heights = cell_features[:, CellFeatureIdx.HEIGHT]

    if use_spectral:
        spectral_pos = _spectral_initial_placement(
            cell_features, pin_features, edge_list,
            scale_exp=hp["spectral_scale_exp"],
        )
        if spectral_pos is not None:
            cell_features[:, 2:4] = spectral_pos

    cell_positions = cell_features[:, 2:4].clone().detach().requires_grad_(True)

    n_ref = 50.0
    scale = max(1.0, N / n_ref)
    lam_ol_active = hp["lam_ol_scale"] * scale
    lam_ol_maint = hp["lam_ol_maint_scale"] * scale
    lam_wl_full = hp["lam_wl_full"]
    lam_wl_sub = hp["lam_wl_sub"]
    drift_weight = hp["drift_weight"]
    USE_ANNEALING = N >= hp["annealing_threshold"]
    max_grad_norm = max(hp["grad_norm_base"], hp["grad_norm_base"] * scale ** hp["grad_norm_exp"])

    optimizer = optim.Adam([cell_positions], lr=lr)
    warmup_epochs = max(int(num_epochs * 0.05), 10)

    overlap_resolved = False
    cur_lam_ol = lam_ol_active
    cur_lam_wl = lam_wl_sub

    # ---- Phase 1: Main training loop ----
    for epoch in range(num_epochs):
        current_lr = _get_lr(epoch, warmup_epochs, num_epochs, lr)
        if not USE_ANNEALING and not overlap_resolved:
            current_lr = max(current_lr, lr * hp["lr_floor_mult"])
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad()
        cf_cur = cell_features.clone()
        cf_cur[:, 2:4] = cell_positions

        wl_loss = wirelength_attraction_loss(cf_cur, pin_features, edge_list)
        ol_loss = overlap_repulsion_loss(cf_cur, pin_features, edge_list)

        if USE_ANNEALING:
            progress = epoch / num_epochs
            lam_ol = lam_ol_active * max(hp["anneal_ol_floor"], 1.0 - progress * hp["anneal_ol_decay"])
            ol_cleared = ol_loss.item() < hp["anneal_ol_gate"]
            lam_wl = lam_wl_full * min(1.0, progress * hp["anneal_wl_ramp"]) if ol_cleared else lam_wl_full * hp["anneal_wl_ungated"]
        else:
            lam_ol = cur_lam_ol
            lam_wl = cur_lam_wl

        total_loss = lam_wl * wl_loss + lam_ol * ol_loss + drift_weight * ((cell_positions - cell_positions.mean(dim=0, keepdim=True)) ** 2).sum() / N
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=max_grad_norm)
        optimizer.step()

        if not USE_ANNEALING:
            with torch.no_grad():
                discrete_count = _count_discrete_overlaps(cell_positions, widths, heights)
            if not overlap_resolved and discrete_count == 0:
                overlap_resolved = True
                cur_lam_ol = lam_ol_maint
                cur_lam_wl = lam_wl_full
            elif overlap_resolved and discrete_count > 0:
                overlap_resolved = False
                cur_lam_ol = lam_ol_active
                cur_lam_wl = lam_wl_sub

        if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            if USE_ANNEALING:
                with torch.no_grad():
                    discrete_count = _count_discrete_overlaps(cell_positions, widths, heights)
            print(f"Epoch {epoch}/{num_epochs} overlaps={discrete_count}:"
                  f"  WL={wl_loss.item():.6f}  OL={ol_loss.item():.6f}  "
                  f"lam_ol={lam_ol:.1f}  lam_wl={lam_wl:.3f}  lr={current_lr:.6f}")

    # ---- Phase 2: Pure overlap reduction ----
    with torch.no_grad():
        pre_ol_count = _count_discrete_overlaps(cell_positions, widths, heights)
    if pre_ol_count > 0:
        ol_opt = optim.Adam([cell_positions], lr=lr)
        for ol_epoch in range(hp["p2_epochs"]):
            ol_opt.zero_grad()
            cf_ol = cell_features.clone()
            cf_ol[:, 2:4] = cell_positions
            ol_loss = overlap_repulsion_loss(cf_ol, pin_features, edge_list)
            (lam_ol_active * hp["p2_lam_mult"] * ol_loss).backward()
            ol_opt.step()
            if ol_epoch % 500 == 0:
                with torch.no_grad():
                    c = _count_discrete_overlaps(cell_positions, widths, heights)
                if verbose:
                    print(f"  OL-phase {ol_epoch}: overlaps={c}  loss={ol_loss.item():.6f}")
                if c == 0:
                    break

    # ---- Phase 3: Greedy legalization ----
    with torch.no_grad():
        legalized_pos = _legalize_placement(cell_positions, widths, heights)
        post_count = _count_discrete_overlaps(legalized_pos, widths, heights)
    cell_positions = legalized_pos.detach().clone().requires_grad_(True)

    # ---- Phase 4: Wirelength optimization ----
    if post_count == 0:
        ft_epochs = max(300, int(hp["p4_epoch_base"] * (hp["p4_epoch_nref"] / N) ** 0.5))
        ft_opt = optim.Adam([cell_positions], lr=hp["p4_lr"])
        wl_guard = hp["p4_wl_guard"]

        for ft_epoch in range(ft_epochs):
            ft_opt.zero_grad()
            cf_ft = cell_features.clone()
            cf_ft[:, 2:4] = cell_positions
            wl_loss = wirelength_attraction_loss(cf_ft, pin_features, edge_list)
            ol_loss = overlap_repulsion_loss(cf_ft, pin_features, edge_list)
            (wl_loss + wl_guard * ol_loss).backward()
            torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=hp["p4_clip"])
            ft_opt.step()

            if ft_epoch % 500 == 0 and ft_epoch > 0:
                with torch.no_grad():
                    c = _count_discrete_overlaps(cell_positions, widths, heights)
                if c > 10:
                    wl_guard = min(wl_guard * hp["p4_guard_mult"], hp["p4_guard_max"])

        with torch.no_grad():
            legalized_pos = _legalize_placement(cell_positions, widths, heights)
            post_count = _count_discrete_overlaps(legalized_pos, widths, heights)
        cell_positions = legalized_pos.detach().clone().requires_grad_(True)

        # ---- Phase 5: Post-legalization wirelength cleanup ----
        if post_count == 0:
            final_opt = optim.Adam([cell_positions], lr=hp["p5_lr"])
            for _ in range(hp["p5_epochs"]):
                final_opt.zero_grad()
                cf_final = cell_features.clone()
                cf_final[:, 2:4] = cell_positions
                wl_loss = wirelength_attraction_loss(cf_final, pin_features, edge_list)
                ol_loss = overlap_repulsion_loss(cf_final, pin_features, edge_list)
                (wl_loss + hp["p5_ol_weight"] * ol_loss).backward()
                torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=hp["p5_clip"])
                final_opt.step()

            with torch.no_grad():
                legalized_pos = _legalize_placement(cell_positions, widths, heights)
            cell_positions = legalized_pos.detach().clone()

    final_cf = cell_features.clone()
    final_cf[:, 2:4] = cell_positions.detach()
    with torch.no_grad():
        final_wl = wirelength_attraction_loss(final_cf, pin_features, edge_list).item()
    return final_cf, final_wl


def train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=5000,
    lr=0.05,
    lambda_wirelength=1.0,
    lambda_overlap=10.0,
    verbose=True,
    log_interval=100,
    hp=None,
):
    """Placement optimizer with multi-restart for small N, annealing for large N.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity
        num_epochs: Max optimization iterations per restart
        lr: Peak learning rate for Adam
        lambda_wirelength: Base weight for wirelength loss (unused, kept for API compat)
        lambda_overlap: Base weight for overlap loss (unused, kept for API compat)
        verbose: Whether to print progress
        log_interval: How often to print progress
        hp: Optional hyperparameter dict (merged with _DEFAULT_HP)

    Returns:
        Dictionary with final_cell_features, initial_cell_features, loss_history.
    """
    initial_cell_features = cell_features.clone()
    N = cell_features.shape[0]
    n_restarts = max(1, int(400 / N)) if N < 200 else 1

    best_cf = None
    best_wl = float('inf')

    for restart in range(n_restarts):
        cf_run = cell_features.clone()

        if restart > 0:
            total_area = cf_run[:, 0].sum().item()
            spread = (total_area ** 0.5) * 0.6
            angles = torch.rand(N) * 2 * 3.14159
            radii = torch.rand(N) * spread
            cf_run[:, 2] = radii * torch.cos(angles)
            cf_run[:, 3] = radii * torch.sin(angles)

        use_spectral = (restart == 0)
        run_verbose = verbose and (restart == 0)

        final_cf, final_wl = _single_train_run(
            cf_run, pin_features, edge_list, use_spectral,
            num_epochs, lr, run_verbose, log_interval, hp=hp,
        )

        overlap_count = len(calculate_cells_with_overlaps(final_cf))
        if overlap_count == 0 and final_wl < best_wl:
            best_wl = final_wl
            best_cf = final_cf

        if verbose:
            tag = "*best*" if final_wl <= best_wl and overlap_count == 0 else ""
            print(f"Restart {restart+1}/{n_restarts}: wl={final_wl:.4f}  "
                  f"overlaps={overlap_count}  {tag}")

    if best_cf is None:
        best_cf = final_cf

    return {
        "final_cell_features": best_cf,
        "initial_cell_features": initial_cell_features,
        "loss_history": {},
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
