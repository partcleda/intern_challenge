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


# Helper: greedy deterministic untangler (module-level)
def greedy_untangle(cell_feats, max_iter=200, step=1.0):
    cf = cell_feats.clone()
    N = cf.shape[0]
    for it in range(max_iter):
        moved = False
        positions = cf[:, 2:4].detach().numpy()
        widths = cf[:, 4].detach().numpy()
        heights = cf[:, 5].detach().numpy()

        # Check all pairs
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                min_sep_x = (widths[i] + widths[j]) / 2.0
                min_sep_y = (heights[i] + heights[j]) / 2.0
                overlap_x = max(0.0, min_sep_x - abs(dx))
                overlap_y = max(0.0, min_sep_y - abs(dy))

                if overlap_x > 0 and overlap_y > 0:
                    # Move each cell half the required separation along the larger axis
                    if overlap_x > overlap_y:
                        dir = 1.0 if dx >= 0 else -1.0
                        shift = (overlap_x / 2.0 + 1e-6) * dir * step
                        positions[i, 0] += shift
                        positions[j, 0] -= shift
                    else:
                        dir = 1.0 if dy >= 0 else -1.0
                        shift = (overlap_y / 2.0 + 1e-6) * dir * step
                        positions[i, 1] += shift
                        positions[j, 1] -= shift
                    moved = True

        cf[:, 2:4] = torch.from_numpy(positions)
        if not moved:
            break

    return cf


# Helper: local cluster untangler (module-level)
def local_cluster_untangle(cell_feats, max_iter=200, step=1.0, size_threshold=6):
    cf = cell_feats.clone()
    N = cf.shape[0]

    def find_overlapping_pairs(cf_local):
        pos = cf_local[:, 2:4].detach().numpy()
        w = cf_local[:, 4].detach().numpy()
        h = cf_local[:, 5].detach().numpy()
        pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = abs(pos[i,0] - pos[j,0])
                dy = abs(pos[i,1] - pos[j,1])
                minx = (w[i] + w[j]) / 2.0
                miny = (h[i] + h[j]) / 2.0
                ox = max(0.0, minx - dx)
                oy = max(0.0, miny - dy)
                if ox > 0 and oy > 0:
                    pairs.append((i,j, ox*oy))
        return pairs

    pairs = find_overlapping_pairs(cf)
    if not pairs:
        return cf

    # Union-find
    parent = list(range(N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for i,j,_ in pairs:
        union(i,j)
    clusters = {}
    for i in range(N):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    for cl in clusters.values():
        if len(cl) < size_threshold:
            continue

        # Local deterministic pushes inside cluster
        for it in range(max_iter):
            moved = False
            pos = cf[:, 2:4].detach().numpy()
            w = cf[:, 4].detach().numpy()
            h = cf[:, 5].detach().numpy()
            for a in cl:
                for b in cl:
                    if a >= b:
                        continue
                    dx = pos[a,0] - pos[b,0]
                    dy = pos[a,1] - pos[b,1]
                    minx = (w[a] + w[b]) / 2.0
                    miny = (h[a] + h[b]) / 2.0
                    ox = max(0.0, minx - abs(dx))
                    oy = max(0.0, miny - abs(dy))
                    if ox > 0 and oy > 0:
                        if ox > oy:
                            dir = 1.0 if dx >= 0 else -1.0
                            shift = (ox/2.0 + 1e-6) * dir * step
                            pos[a,0] += shift
                            pos[b,0] -= shift
                        else:
                            dir = 1.0 if dy >= 0 else -1.0
                            shift = (oy/2.0 + 1e-6) * dir * step
                            pos[a,1] += shift
                            pos[b,1] -= shift
                        moved = True
            cf[:, 2:4] = torch.from_numpy(pos)
            if not moved:
                break

        # Small local gradient-based overlap-only refine on cluster members
        local_pos = cf[:, 2:4].clone().detach()
        local_pos.requires_grad_(True)
        local_optimizer = optim.Adam([local_pos], lr=0.01)
        for _ in range(100):
            local_optimizer.zero_grad()
            tmp_cf = cf.clone()
            tmp_cf[:, 2:4] = local_pos
            loss = 0.0
            for i in range(len(cl)):
                for j in range(i+1, len(cl)):
                    ii = cl[i]
                    jj = cl[j]
                    xi = local_pos[ii,0]
                    xj = local_pos[jj,0]
                    yi = local_pos[ii,1]
                    yj = local_pos[jj,1]
                    wi = cf[ii,4]
                    wj = cf[jj,4]
                    hi = cf[ii,5]
                    hj = cf[jj,5]
                    dx = torch.abs(xi - xj)
                    dy = torch.abs(yi - yj)
                    minx = (wi + wj) / 2.0
                    miny = (hi + hj) / 2.0
                    ox = torch.relu(minx - dx)
                    oy = torch.relu(miny - dy)
                    loss = loss + ox * oy
            loss.backward()
            torch.nn.utils.clip_grad_norm_([local_pos], max_norm=5.0)
            local_optimizer.step()
        cf[:, 2:4] = local_pos.detach()

    return cf

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

    src_x = pin_absolute_x[src_pins]
    src_y = pin_absolute_y[src_pins]
    tgt_x = pin_absolute_x[tgt_pins]
    tgt_y = pin_absolute_y[tgt_pins]

    # Calculate smooth approximation of Manhattan distance
    # Using log-sum-exp approximation for differentiability
    alpha = 0.1  # Smoothing parameter
    dx = torch.abs(src_x - tgt_x)
    dy = torch.abs(src_y - tgt_y)

    # Smooth L1 distance with numerical stability
    smooth_manhattan = alpha * torch.logsumexp(
        torch.stack([dx / alpha, dy / alpha], dim=0), dim=0
    )

    # Total wirelength
    total_wirelength = torch.sum(smooth_manhattan)

    return total_wirelength / edge_list.shape[0]  # Normalize by number of edges


def overlap_repulsion_loss(cell_features, pin_features, edge_list):
    """
    Calculate loss to prevent cell overlaps using a quadratic penalty.
    This provides a much stronger repulsion force for larger overlaps.
    """
    N = cell_features.shape[0]
    if N <= 1:
        return torch.tensor(0.0, requires_grad=True)

    # Extract positions and sizes
    positions = cell_features[:, CellFeatureIdx.X:CellFeatureIdx.Y+1]  # [N, 2]
    widths = cell_features[:, CellFeatureIdx.WIDTH]      # [N]
    heights = cell_features[:, CellFeatureIdx.HEIGHT]    # [N]

    # Use broadcasting to get all pairwise differences and separations
    # This is highly efficient.
    dx = torch.abs(positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0))
    dy = torch.abs(positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0))

    min_sep_x = (widths.unsqueeze(1) + widths.unsqueeze(0)) / 2.0
    min_sep_y = (heights.unsqueeze(1) + heights.unsqueeze(0)) / 2.0

    # Calculate overlap in each dimension
    overlap_x = torch.relu(min_sep_x - dx)
    overlap_y = torch.relu(min_sep_y - dy)

    # Calculate overlap area for each pair
    overlap_area = overlap_x * overlap_y

    # --- KEY CHANGE ---
    # Use a quadratic penalty for overlap area. This aggressively penalizes
    # larger overlaps and creates a smoother gradient for the optimizer to follow.
    # We also add a small linear term to ensure small overlaps are still penalized.
    quadratic_penalty = overlap_area.pow(2)
    linear_penalty = overlap_area
    
    # Total loss is the sum of penalties over all pairs (avoiding double counting)
    # The triu mask ensures we only count pair (i, j) where i < j
    mask = torch.triu(torch.ones_like(overlap_area), diagonal=1)
    
    # We normalize by the number of pairs to make the loss independent of cell count
    num_pairs = N * (N - 1) / 2.0
    
    # Combine and normalize
    # The 10.0 is a weight to make the quadratic term dominant
    total_loss = torch.sum((10.0 * quadratic_penalty + linear_penalty) * mask) / (num_pairs + 1e-8)

    return total_loss

def train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=1500, # Increased epochs for the single loop
    lr=0.01,         # Slightly higher learning rate can work well with Adam
    lambda_wirelength=1.0,
    # Overlap lambda will now be controlled by a schedule
    lambda_overlap_initial=1.0,
    lambda_overlap_final=500.0, # End with a very high penalty
    verbose=True,
    log_interval=100,
):
    """
    Train the placement using a simplified loop with a dynamic overlap penalty.
    """
    # Clone features and set up positions for gradient descent
    initial_cell_features = cell_features.clone()
    cell_positions = cell_features[:, 2:4].clone().detach()
    cell_positions.requires_grad_(True)

    # Optimizer
    optimizer = optim.Adam([cell_positions], lr=lr)
    
    # --- DYNAMIC LAMBDA SCHEDULE ---
    # This is a key improvement. We start with a low overlap penalty to allow
    # cells to move freely based on wirelength, then gradually increase it
    # to "squeeze out" all overlaps at the end.
    lambda_schedule = torch.linspace(lambda_overlap_initial, lambda_overlap_final, steps=num_epochs)

    loss_history = []

    # --- SIMPLIFIED TRAINING LOOP ---
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Update cell features with current positions
        cell_features_current = cell_features.clone()
        cell_features_current[:, 2:4] = cell_positions

        # Calculate losses using the new overlap function
        wl_loss = wirelength_attraction_loss(
            cell_features_current, pin_features, edge_list
        )
        overlap_loss = overlap_repulsion_loss(
            cell_features_current, pin_features, edge_list
        )

        # Get the current overlap weight from our schedule
        current_lambda_overlap = lambda_schedule[epoch]

        # Combined loss
        total_loss = (lambda_wirelength * wl_loss) + (current_lambda_overlap * overlap_loss)

        # Backward pass and optimization step
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=5.0) # Gradient clipping is good practice
        optimizer.step()

        loss_history.append(total_loss.item())

        # Log progress
        if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch}/{num_epochs} | Loss: {total_loss.item():.4f} | "
                  f"WL Loss: {wl_loss.item():.4f} | Overlap Loss: {overlap_loss.item():.4f} | "
                  f"Lambda_O: {current_lambda_overlap:.2f}")

    # Create final cell features from the optimized positions
    final_cell_features = cell_features.clone()
    final_cell_features[:, 2:4] = cell_positions.detach()
    
    # Optional: A single, final greedy untangle pass to clean up any tiny residual overlaps
    final_cell_features = greedy_untangle(final_cell_features, max_iter=500, step=1.0)


    return {
        "final_cell_features": final_cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": loss_history, # Simplified history
    }

# Deterministic cluster packer: pack cluster members in a grid inside bounding box
def pack_clusters(final_cell_features, size_threshold=2, max_attempts=5):
    cf = final_cell_features.clone()
    N = cf.shape[0]

    # Build overlapping pairs
    positions = cf[:, 2:4].detach().numpy()
    widths = cf[:, 4].detach().numpy()
    heights = cf[:, 5].detach().numpy()

    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            dx = abs(positions[i,0] - positions[j,0])
            dy = abs(positions[i,1] - positions[j,1])
            minx = (widths[i] + widths[j]) / 2.0
            miny = (heights[i] + heights[j]) / 2.0
            ox = max(0.0, minx - dx)
            oy = max(0.0, miny - dy)
            if ox > 0 and oy > 0:
                pairs.append((i,j))

    if not pairs:
        return cf

    # Union-find build clusters
    parent = list(range(N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for i,j in pairs:
        union(i,j)

    clusters = {}
    for i in range(N):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    # Pack clusters deterministically
    for cl in clusters.values():
        if len(cl) < size_threshold:
            continue

        # compute bounding box of cluster
        xs = positions[cl,0]
        ys = positions[cl,1]
        ws = widths[cl]
        hs = heights[cl]
        min_x = float(xs.min() - ws.max())
        max_x = float(xs.max() + ws.max())
        min_y = float(ys.min() - hs.max())
        max_y = float(ys.max() + hs.max())

        # try packing into a grid; increase grid spacing if overlaps persist
        for attempt in range(max_attempts):
            cols = int(max(1, round(len(cl) ** 0.5)))
            rows = int((len(cl) + cols - 1) // cols)
            cell_w = max(ws.max(), 1e-3)
            cell_h = max(hs.max(), 1e-3)
            grid_w = cols * cell_w * (1.1 + 0.2 * attempt)
            grid_h = rows * cell_h * (1.1 + 0.2 * attempt)

            # center grid in bounding box
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            start_x = cx - grid_w / 2.0 + cell_w / 2.0
            start_y = cy - grid_h / 2.0 + cell_h / 2.0

            new_pos = []
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx >= len(cl):
                        break
                    x = start_x + c * (grid_w / cols)
                    y = start_y + r * (grid_h / rows)
                    new_pos.append((x, y))
                    idx += 1
                if idx >= len(cl):
                    break

            # apply positions and test for overlaps
            for ii, (x,y) in zip(cl, new_pos):
                cf[ii, 2] = float(x)
                cf[ii, 3] = float(y)

            # check for overlaps in this cluster
            ok = True
            pos2 = cf[:, 2:4].detach().numpy()
            for a in cl:
                for b in cl:
                    if a >= b:
                        continue
                    dx = abs(pos2[a,0] - pos2[b,0])
                    dy = abs(pos2[a,1] - pos2[b,1])
                    minx = (widths[a] + widths[b]) / 2.0
                    miny = (heights[a] + heights[b]) / 2.0
                    if dx < minx and dy < miny:
                        ok = False
                        break
                if not ok:
                    break

            if ok:
                break

    return cf


# Aggressive deterministic overlap elimination: repeated untangle + packer
def aggressive_overlap_elimination(cell_features, pin_features, edge_list, max_rounds=5):
    """Repeated deterministic passes to eliminate overlaps before gradient steps.

    This applies progressively stronger untangling and packing passes until
    no overlapping cells remain or `max_rounds` is reached.
    """
    cf = cell_features.clone()
    for rnd in range(max_rounds):
        # quick check
        cells_with_overlaps = calculate_cells_with_overlaps(cf)
        if len(cells_with_overlaps) == 0:
            break

        # 1) Strong greedy untangle with bigger steps and more iterations
        cf = greedy_untangle(cf, max_iter=500, step=2.0)

        # 2) Local cluster untangler with lower size threshold to hit small clusters
        cf = local_cluster_untangle(cf, max_iter=300, step=1.5, size_threshold=2)

        # 3) Aggressive packer: more attempts and spacing to resolve dense clusters
        cf = pack_clusters(cf, size_threshold=2, max_attempts=12)

        # 4) Final lightweight gradient refine for overlaps only (short)
        cells_with_overlaps = calculate_cells_with_overlaps(cf)
        if len(cells_with_overlaps) == 0:
            break

        if len(cells_with_overlaps) > 0:
            # Build a small optimizer over positions and minimize overlap loss only
            pos = cf[:, 2:4].clone().detach()
            pos.requires_grad_(True)
            opt = optim.Adam([pos], lr=0.02)
            for _ in range(200):
                opt.zero_grad()
                tmp = cf.clone()
                tmp[:, 2:4] = pos
                ov = overlap_repulsion_loss(tmp, pin_features, edge_list)
                ov.backward()
                torch.nn.utils.clip_grad_norm_([pos], max_norm=5.0)
                opt.step()
            cf[:, 2:4] = pos.detach()

    return cf


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
