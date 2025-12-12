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
import math
import random
import numpy as np


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


def overlap_repulsion_loss(cell_features, pin_features, edge_list, inflation_factor=1.0):
    """
    Calculates overlap loss by checking every cell against every other cell.
    
    Strategy:
    1. Check all pairs (N*N).
    2. If they overlap, penalize the square of the overlap area.
    3. This pushes cells apart.
    """
    N = cell_features.shape[0]
    device = cell_features.device
    if N <= 1:
        return torch.tensor(0.0, requires_grad=True, device=device)

    positions = cell_features[:, 2:4]
    widths = cell_features[:, 4] * inflation_factor
    heights = cell_features[:, 5] * inflation_factor
    
    # Calculate pairwise distances
    # Since N is small (<2500), we can afford to check every single pair (O(N^2)).
    # This gives the most accurate gradient.
    
    # x_diff[i, j] = x[i] - x[j]
    x = positions[:, 0]
    y = positions[:, 1]
    
    x_diff = x.unsqueeze(1) - x.unsqueeze(0)
    y_diff = y.unsqueeze(1) - y.unsqueeze(0)
    
    x_dist = torch.abs(x_diff)
    y_dist = torch.abs(y_diff)
    
    # Minimum separation required to avoid overlap
    w_sum = widths.unsqueeze(1) + widths.unsqueeze(0)
    h_sum = heights.unsqueeze(1) + heights.unsqueeze(0)
    
    min_sep_x = w_sum / 2.0
    min_sep_y = h_sum / 2.0
    
    # Overlap amount (positive if overlapping, 0 otherwise)
    overlap_w = torch.relu(min_sep_x - x_dist)
    overlap_h = torch.relu(min_sep_y - y_dist)
    
    # Overlap area
    overlap_area = overlap_w * overlap_h
    
    # Penalize overlap strongly
    # Using squared overlap area creates stronger gradients for large overlaps
    # Adding a linear term ensures gradients for small overlaps don't vanish
    loss_matrix = overlap_area ** 2 + overlap_area * 100.0
    
    # only count upper triangle to avoid double counting, and exclude diagonal
    mask = torch.triu(torch.ones(N, N, device=device), diagonal=1)
    
    total_loss = torch.sum(loss_matrix * mask)
    
    return total_loss


def grid_density_loss(cell_features, grid_resolution=64):
    """Calculate density overflow loss using Separable Axes (X and Y projections)."""
    N = cell_features.shape[0]
    if N == 0:
        return torch.tensor(0.0, requires_grad=True, device=cell_features.device)

    # Fixed Canvas
    CANVAS_SIZE = 1000.0
    bin_size = CANVAS_SIZE / grid_resolution
    
    # 1D Projection Data
    cx = cell_features[:, 2].unsqueeze(1)
    cy = cell_features[:, 3].unsqueeze(1)
    w = cell_features[:, 4].unsqueeze(1)
    h = cell_features[:, 5].unsqueeze(1)
    
    # Bin Centers [1, R]
    bin_centers = torch.linspace(bin_size/2, CANVAS_SIZE - bin_size/2, grid_resolution, device=cell_features.device).unsqueeze(0)
    bin_L = bin_centers - bin_size/2
    bin_R = bin_centers + bin_size/2
    
    # X Density
    cell_L = cx - w/2
    cell_R = cx + w/2
    overlap_x = torch.relu(torch.min(cell_R, bin_R) - torch.max(cell_L, bin_L))
    density_x = torch.sum(overlap_x, dim=0)

    # Y Density
    cell_B = cy - h/2
    cell_T = cy + h/2
    overlap_y = torch.relu(torch.min(cell_T, bin_R) - torch.max(cell_B, bin_L))
    density_y = torch.sum(overlap_y, dim=0)
    
    # Target (Physical Capacity)
    # Allow density up to the physical width of the bin (100% utilization)
    # This allows clustering (good for WL) while preventing overflow
    target_x = bin_size * 1.0
    target_y = bin_size * 1.0
    
    overflow_x = torch.relu(density_x - target_x)
    overflow_y = torch.relu(density_y - target_y)
    
    return torch.mean(overflow_x**2) + torch.mean(overflow_y**2)


def legalize_placement(cell_features):
    """
    Projects cells to legal non-overlapping positions on a discrete grid.

    This function implements a Hybrid Legalization Strategy:
    1.  **Sorting**: Processes Macros (large cells) first, then Standard Cells.
    2.  **Vectorized Spiral Search**: Checks thousands of local candidate positions efficiently using NumPy vectorization.
    3.  **SAT Fallback**: If local search fails, uses a Summed Area Table (Integral Image) to find the nearest valid spot globally in O(1) per region check.

    Args:
        cell_features (torch.Tensor): [N, 6] tensor containing cell attributes.
                                      Columns 2 and 3 (x, y) are updated in-place.

    Returns:
        torch.Tensor: The updated cell_features tensor with legal coordinates.
    """
    N = cell_features.shape[0]
    device = cell_features.device
    
    # Working copies
    pos = cell_features[:, 2:4].clone().cpu().numpy()
    widths = cell_features[:, 4].clone().cpu().numpy()
    heights = cell_features[:, 5].clone().cpu().numpy()
    
    # Canvas parameters
    CANVAS_W, CANVAS_H = 1000.0, 1000.0
    
    # Simple Occupancy Grid (Approximate) to speed up search
    # 1000x1000 grid is 1M entries. manageable.
    occupied = torch.zeros((1000, 1000), dtype=torch.bool)
    # Use Numpy for grid (Faster CPU access and vectorization)
    occupied = np.zeros((1000, 1000), dtype=bool)
    
    # Sort indices by Area (Descending) - Macros first
    # Idea: Place Big blocks first (harder to fit), then fill gaps with small cells.
    areas = widths * heights
    indices = torch.argsort(torch.tensor(areas), descending=True).numpy()
    
    for idx in indices:
        w = widths[idx]
        h = heights[idx]
        x = pos[idx, 0]
        y = pos[idx, 1]
        
        # Snap to grid dimensions
        w_grid = int(math.ceil(w + 0.05))
        h_grid = int(math.ceil(h + 0.05))
        w_grid = max(1, min(1000, w_grid))
        h_grid = max(1, min(1000, h_grid))
        
        # Initial candidate (top-left)
        tl_x = int(round(x - w/2))
        tl_y = int(round(y - h/2))
        
    # Precompute spiral offsets once (up to radius 50)
    # This optimization checks nearby spots first. I realized checking one by one was too slow in Python.
    max_spiral_radius = 50
    sq_radius = max_spiral_radius * 2 + 1
    x_range = np.arange(-max_spiral_radius, max_spiral_radius + 1)
    y_range = np.arange(-max_spiral_radius, max_spiral_radius + 1)
    xx, yy = np.meshgrid(x_range, y_range)
    dists = xx**2 + yy**2
    # Sort by distance to ensure closest spots are checked first
    flat_indices = np.argsort(dists.ravel())
    spiral_dx = xx.ravel()[flat_indices]
    spiral_dy = yy.ravel()[flat_indices]
    
    # Filter out 0,0 since we check it manually/implicitly or duplicate doesn't matter much
    # The first element is 0,0.
    
    # print(f"DEBUG: Legalizing {len(indices)} cells with Vectorized Spiral...", flush=True)

    for idx in indices:
        w = widths[idx]
        h = heights[idx]
        x = pos[idx, 0]
        y = pos[idx, 1]
        
        # Snap to grid dimensions
        w_grid = int(math.ceil(w + 0.05))
        h_grid = int(math.ceil(h + 0.05))
        w_grid = max(1, min(1000, w_grid))
        h_grid = max(1, min(1000, h_grid))
        
        # Initial candidate (top-left)
        tl_x = int(round(x - w/2))
        tl_y = int(round(y - h/2))
        
        found = False
        
        # OPTIMIZATION: Vectorized Spiral Search
        # Initially, I used a loop here, but it was extremely slow for 100k cells.
        # Moved to NumPy vectorization to check thousands of candidates at once.
        
        # 1. Generate candidate coordinates
        cand_x = tl_x + spiral_dx
        cand_y = tl_y + spiral_dy
        
        # 2. Filter bounds
        valid_bounds = (cand_x >= 0) & (cand_y >= 0) & (cand_x + w_grid <= 1000) & (cand_y + h_grid <= 1000)
        cand_x = cand_x[valid_bounds]
        cand_y = cand_y[valid_bounds]
        
        if len(cand_x) > 0:
            # 3. Check occupancy
            # Fast check for standard cells (Height=1 usually)
            if h_grid == 1 and w_grid <= 3:
                # Optimized for known standard cell sizes
                is_occupied = occupied[cand_x, cand_y]
                if w_grid > 1:
                    is_occupied |= occupied[cand_x + 1, cand_y]
                if w_grid > 2:
                    is_occupied |= occupied[cand_x + 2, cand_y]
                    
                # Find first free spot
                free_indices = np.flatnonzero(~is_occupied)
                if len(free_indices) > 0:
                    best_k = free_indices[0]
                    ctx, cty = cand_x[best_k], cand_y[best_k]
                    
                    occupied[ctx:ctx+w_grid, cty:cty+h_grid] = True
                    pos[idx, 0] = ctx + w/2
                    pos[idx, 1] = cty + h/2
                    found = True
            else:
                # General case (iterative but still faster than python spiral generation)
                # Check first few candidates?
                # For macros, we can just iterate the sorted candidates
                for k in range(min(5000, len(cand_x))): # Check first 5000 valid spots
                    ctx, cty = cand_x[k], cand_y[k]
                    if not occupied[ctx:ctx+w_grid, cty:cty+h_grid].any():
                        occupied[ctx:ctx+w_grid, cty:cty+h_grid] = True
                        pos[idx, 0] = ctx + w/2
                        pos[idx, 1] = cty + h/2
                        found = True
                        break

        # 2. FALLBACK STAGE: Integral Image 
        # If the local search (spiral) fails, we need to scan the whole board.
        # Checking every pixel is O(W*H), which is too slow.
        # I used a Summed Area Table (SAT) to query range sums in O(1).
        # This helps place the last few 'impossible' cells.
        if not found:
            # Construct Summed Area Table
            sat = occupied.astype(int).cumsum(axis=0).cumsum(axis=1)
            pad_sat = np.pad(sat, ((1,0), (1,0)), mode='constant')
            
            # Helper to get region sum in O(1)
            rows = 1000 - w_grid + 1
            cols = 1000 - h_grid + 1
            
            if rows > 0 and cols > 0:
                region_sums = (pad_sat[w_grid:w_grid+rows, h_grid:h_grid+cols] 
                             - pad_sat[0:rows, h_grid:h_grid+cols] 
                             - pad_sat[w_grid:w_grid+rows, 0:cols] 
                             + pad_sat[0:rows, 0:cols])
                
                valid_indices = np.argwhere(region_sums == 0)
                
                if len(valid_indices) > 0:
                    # Choose the one closest to original (x,y)?
                    # To optimize WL, we should find the closest available spot.
                    # calculating distances to (tl_x, tl_y) for all candidates is vectorizable.
                    
                    cand_x = valid_indices[:, 0]
                    cand_y = valid_indices[:, 1]
                    
                    dists = (cand_x - tl_x)**2 + (cand_y - tl_y)**2
                    best_k = np.argmin(dists)
                    
                    ctx = cand_x[best_k]
                    cty = cand_y[best_k]
                    
                    occupied[ctx:ctx+w_grid, cty:cty+h_grid] = True
                    pos[idx, 0] = ctx + w/2
                    pos[idx, 1] = cty + h/2
                    found = True
            
        if not found:
             print(f"CRITICAL: Could not legalize cell {idx}. Canvas full?")

    # Update tensor
    cell_features[:, 2:4] = torch.tensor(pos, device=device)
    return cell_features

def train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=300,
    lr=0.5,
    lambda_wirelength=0.001,
    lambda_overlap=10000.0,
    verbose=True,
    log_interval=50,
    save_visuals=False,
    visual_dir="visuals",
):
    """Run the placement optimization loop."""
    if verbose:
        print(f"Starting placement optimization...")
        print(f"  Device: {cell_features.device}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning Rate: {lr}")

    N = cell_features.shape[0]
    initial_cell_features = cell_features.clone()
    
    # Initialize with grid if mostly zero (fix input ambiguity)
    cell_positions = cell_features[:, 2:4].clone().detach()
    
    # Initialize with a spread out grid to minimize initial overlaps
    # Only do this if strictly necessary (all zeros). 
    # Forcing it overrides potentially good initial state.
    if torch.all(cell_positions.abs() < 1e-6): 
        if verbose: print("Initializing with spread grid...")
        rows = int(math.ceil(math.sqrt(N)))
        cols = int(math.ceil(N / rows))
        canvas_width = 1000.0
        canvas_height = 1000.0
        
        spacing_x = canvas_width / (cols + 1)
        spacing_y = canvas_height / (rows + 1)
        
        for i in range(N):
            r = i // cols
            c = i % cols
            cell_positions[i, 0] = (c + 1) * spacing_x
            cell_positions[i, 1] = (r + 1) * spacing_y
            
        # Add noise
        cell_positions += torch.randn_like(cell_positions) * (spacing_x * 0.1)
        
    cell_positions.requires_grad_(True)
    
    # History
    loss_history = {"total_loss": [], "wirelength_loss": [], "overlap_loss": []}
    
    # Optimizer (Adam for speed)
    optimizer = optim.Adam([cell_positions], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)

    consecutive_zero = 0
    target_zero_epochs = 20
    
    # Large N Handling
    use_exact_overlap = (N <= 2500)
    if not use_exact_overlap:
        pass # Large N optimization applied silently
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        current_features = cell_features.clone()
        current_features[:, 2:4] = cell_positions
        
        progress = epoch / num_epochs
        # Inflation: 1.2 -> 1.0 quickly
        inflation = max(1.0, 1.2 - progress * 4.0)
        
        # Calculate Losses
        wl_loss = wirelength_attraction_loss(current_features, pin_features, edge_list)
        
        if use_exact_overlap:
            # Pure Repulsion Focus for small/medium designs
            ov_loss = overlap_repulsion_loss(current_features, pin_features, edge_list, inflation_factor=inflation)
            # Ramp up overlap penalty
            # Reduced ramp from 50.0 to 5.0 to prioritize WL
            curr_lambda_ov = lambda_overlap * (1.0 + progress * 5.0)
            
            # Remove density loss for small N (it conflicts with precise packing)
            # d_loss = grid_density_loss(current_features)
            # total_loss = lambda_wirelength * wl_loss + curr_lambda_ov * ov_loss + d_loss * 100.0
            total_loss = lambda_wirelength * wl_loss + curr_lambda_ov * ov_loss
            
        else:
            # Scalable approach for Large N (>2500)
            d_loss = grid_density_loss(current_features)
            
            # Reduced density weight: Trust Legalizer to fix overlaps
            # Prioritize Wirelength
            lambda_density = 10000.0 * (1.0 + progress * 1.0) 
            
            ov_loss = torch.tensor(0.0)
            
            total_loss = lambda_wirelength * wl_loss + lambda_density * d_loss

            
        # Boundary constraints
        bound_violation = torch.relu(-cell_positions) + torch.relu(cell_positions - 1000.0)
        total_loss += torch.sum(bound_violation ** 2) * 500.0
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=20.0)
        
        optimizer.step()
        scheduler.step()
        
        # Tracking & Early Stopping
        ov_val = ov_loss.item()
        
        if use_exact_overlap:
            if ov_val <= 1e-5 and inflation <= 1.001:
                consecutive_zero += 1
            else:
                consecutive_zero = 0
        else:
             # Run full epochs for large N to optimize WL
            consecutive_zero = 0 
            
        loss_history["total_loss"].append(total_loss.item())
        loss_history["wirelength_loss"].append(wl_loss.item())
        loss_history["overlap_loss"].append(ov_val)
        
        if verbose and epoch % log_interval == 0:
            print(f"Epoch {epoch}: Tot={total_loss.item():.1f} OvLoss={ov_val:.6f} Infl={inflation:.2f}")

        if consecutive_zero >= target_zero_epochs and epoch > 50:
            if verbose: print(f"Converged at epoch {epoch} with zero overlap.")
            break

    # Final Polish: Legalization (Outside loop!)
    if verbose: print("Running legalization step...")
    cell_positions.data.clamp_(0, 1000.0)
    final_cell_features = cell_features.clone()
    final_cell_features[:, 2:4] = cell_positions.detach()
    
    # Enforce strict non-overlap
    final_cell_features = legalize_placement(final_cell_features)
    
    return {
        "final_cell_features": final_cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": loss_history
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
