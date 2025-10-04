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

    # Calculate Manhattan distance (L1 norm)
    dx = torch.abs(src_x - tgt_x)
    dy = torch.abs(src_y - tgt_y)

    # Use simple L1 distance (Manhattan) which is differentiable
    # Adding small epsilon for numerical stability
    epsilon = 1e-8
    manhattan_dist = dx + dy + epsilon

    # Total wirelength
    total_wirelength = torch.sum(manhattan_dist)

    return total_wirelength / edge_list.shape[0]  # Normalize by number of edges


def overlap_repulsion_loss(cell_features, pin_features, edge_list):
    """Calculate loss to prevent cell overlaps with exponential penalties.

    This function implements a highly aggressive overlap detection and penalty system
    using vectorized PyTorch operations. It uses exponential penalties to strongly
    discourage any overlaps.

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

    # Extract cell properties
    positions = cell_features[:, 2:4]  # [N, 2] - x, y coordinates
    widths = cell_features[:, 4]       # [N] - cell widths
    heights = cell_features[:, 5]      # [N] - cell heights
    areas = cell_features[:, 0]        # [N] - cell areas
    
    # Create pairwise distance matrices using broadcasting
    positions_i = positions.unsqueeze(1)  # [N, 1, 2]
    positions_j = positions.unsqueeze(0)  # [1, N, 2]
    
    # Calculate center-to-center distances for all pairs
    dx = torch.abs(positions_i[:, :, 0] - positions_j[:, :, 0])  # [N, N]
    dy = torch.abs(positions_i[:, :, 1] - positions_j[:, :, 1])  # [N, N]
    
    # Calculate minimum separation distances for all pairs
    widths_i = widths.unsqueeze(1)   # [N, 1]
    widths_j = widths.unsqueeze(0)   # [1, N]
    heights_i = heights.unsqueeze(1) # [N, 1]
    heights_j = heights.unsqueeze(0) # [1, N]
    
    # Minimum separation to avoid overlap (add small margin for safety)
    margin = 0.01
    min_sep_x = (widths_i + widths_j) / 2.0 + margin   # [N, N]
    min_sep_y = (heights_i + heights_j) / 2.0 + margin # [N, N]
    
    # Calculate overlap amounts using ReLU for differentiability
    overlap_x = torch.relu(min_sep_x - dx)  # [N, N]
    overlap_y = torch.relu(min_sep_y - dy)  # [N, N]
    
    # Total overlap area for each pair
    overlap_area = overlap_x * overlap_y  # [N, N]
    
    # Create upper triangular mask to avoid double counting and self-comparison
    upper_tri_mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    
    # Apply mask to get only unique pairs
    masked_overlap_area = overlap_area * upper_tri_mask.float()
    
    # Count number of overlapping pairs
    has_overlap = (masked_overlap_area > 1e-6).float()
    num_overlaps = torch.sum(has_overlap)
    
    # Sum all overlap areas
    total_overlap = torch.sum(masked_overlap_area)
    
    # Exponential penalty: penalize overlaps MUCH more strongly
    # Use L2 (squared) penalty for stronger gradients
    squared_overlap = torch.sum(masked_overlap_area ** 2)
    
    # Combine multiple penalty terms
    # 1. Linear overlap area
    # 2. Squared overlap area (L2 penalty)
    # 3. Exponential penalty for any overlap existence
    # 4. Count penalty (penalize number of overlapping pairs)
    
    num_pairs = N * (N - 1) // 2
    
    # Multi-component loss with aggressive penalties
    linear_loss = total_overlap / max(num_pairs, 1)
    quadratic_loss = squared_overlap / max(num_pairs, 1)
    count_penalty = num_overlaps / max(num_pairs, 1)
    
    # Cubic penalty for even stronger gradients on larger overlaps
    cubic_overlap = torch.sum(masked_overlap_area ** 3)
    cubic_loss = cubic_overlap / max(num_pairs, 1)
    
    # Exponential penalty for overlap existence (with safe clamping)
    overlap_normalized = torch.clamp(total_overlap / 5.0, max=30.0)
    exp_penalty = torch.exp(overlap_normalized) - 1.0
    
    # Combine all penalties with ULTRA aggressive weights
    final_loss = (
        50.0 * linear_loss +      # Basic overlap area (boosted)
        200.0 * quadratic_loss +  # Squared penalty (stronger gradients)
        100.0 * count_penalty +   # Penalty for number of overlaps (boosted)
        500.0 * cubic_loss +      # Cubic penalty (EXTREME for large overlaps)
        20.0 * exp_penalty        # Exponential penalty (boosted)
    )
    
    return final_loss


def train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=5000,
    lr=0.1,
    lambda_wirelength=1.0,
    lambda_overlap=1500.0,
    verbose=True,
    log_interval=100,
):
    """Train the placement optimization using multi-phase adaptive gradient descent.

    This implementation uses:
    - Multi-phase optimization (overlap elimination -> balanced -> wirelength)
    - Early stopping when zero overlaps achieved
    - Adaptive learning rate scheduling
    - Aggressive overlap penalties with exponential decay

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity
        num_epochs: Maximum number of optimization iterations
        lr: Initial learning rate for Adam optimizer
        lambda_wirelength: Weight for wirelength loss
        lambda_overlap: Weight for overlap loss
        verbose: Whether to print progress
        log_interval: How often to print progress

    Returns:
        Dictionary with:
            - final_cell_features: Optimized cell positions
            - initial_cell_features: Original cell positions (for comparison)
            - loss_history: Loss values over time
            - stopped_early: Whether early stopping was triggered
    """
    # Clone features and create learnable positions
    cell_features = cell_features.clone()
    initial_cell_features = cell_features.clone()

    # Make only cell positions require gradients
    cell_positions = cell_features[:, 2:4].clone().detach()
    cell_positions.requires_grad_(True)

    # Create optimizer with momentum
    optimizer = optim.Adam([cell_positions], lr=lr, betas=(0.9, 0.999))
    
        # Learning rate scheduler for adaptive learning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=150, min_lr=1e-5
    )

    # Track loss history
    loss_history = {
        "total_loss": [],
        "wirelength_loss": [],
        "overlap_loss": [],
    }

    # Early stopping parameters - adaptive based on problem size
    N = cell_features.shape[0]
    if N < 50:
        early_stop_patience = 600
    elif N < 150:
        early_stop_patience = 1000
    elif N < 500:
        early_stop_patience = 1500
    else:
        # VERY large problems need extreme patience
        early_stop_patience = 3000
        
    early_stop_counter = 0
    best_overlap_loss = float('inf')
    zero_overlap_epochs = 0
    stopped_early = False

    # Training loop with multi-phase adaptive strategies
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Create cell_features with current positions
        cell_features_current = cell_features.clone()
        cell_features_current[:, 2:4] = cell_positions

        # Calculate losses
        wl_loss = wirelength_attraction_loss(
            cell_features_current, pin_features, edge_list
        )
        overlap_loss = overlap_repulsion_loss(
            cell_features_current, pin_features, edge_list
        )

        # MULTI-PHASE OPTIMIZATION with adaptive weighting
        epoch_tensor = torch.tensor(float(epoch))
        
        # Phase 1 (0-30%): Aggressive overlap elimination
        # Phase 2 (30-70%): Balanced optimization  
        # Phase 3 (70-100%): Wirelength focus (if no overlaps)
        phase1_end = int(num_epochs * 0.3)
        phase2_end = int(num_epochs * 0.7)
        
        if epoch < phase1_end:
            # Phase 1: VERY aggressive overlap penalty
            overlap_weight = lambda_overlap * (1.0 + 15.0 * torch.exp(-epoch_tensor / 400.0))
            wl_weight = lambda_wirelength * 0.3  # Reduce wirelength importance
        elif epoch < phase2_end:
            # Phase 2: Balanced with decay
            overlap_weight = lambda_overlap * (1.0 + 5.0 * torch.exp(-(epoch_tensor - phase1_end) / 600.0))
            wl_weight = lambda_wirelength * 0.7
        else:
            # Phase 3: Focus on wirelength if overlaps are gone
            if overlap_loss.item() < 1e-4:
                overlap_weight = lambda_overlap * 0.5
                wl_weight = lambda_wirelength * 2.0
            else:
                # Still have overlaps, keep fighting them
                overlap_weight = lambda_overlap
                wl_weight = lambda_wirelength
        
        total_loss = wl_weight * wl_loss + overlap_weight * overlap_loss

        # Backward pass
        total_loss.backward()

        # Adaptive gradient clipping - more conservative to prevent NaN
        if epoch < phase1_end:
            max_grad_norm = 3.0  # More conservative in phase 1
        else:
            max_grad_norm = 1.5 + 1.0 * torch.exp(-torch.tensor(total_loss.item()) / 50.0)
        torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=max_grad_norm)

        # Update positions
        optimizer.step()
        
        # Check for NaN in positions after update
        if torch.isnan(cell_positions).any():
            if verbose:
                print(f"\n⚠️  NaN detected in cell positions at epoch {epoch}! Stopping.")
            stopped_early = True
            # Reset to last valid state by going back one step
            optimizer.zero_grad()
            break
        
        # Update learning rate based on overlap loss
        scheduler.step(overlap_loss)

        # Record losses
        loss_history["total_loss"].append(total_loss.item())
        loss_history["wirelength_loss"].append(wl_loss.item())
        loss_history["overlap_loss"].append(overlap_loss.item())

        # Early stopping logic - track zero overlap achievement
        if overlap_loss.item() < 1e-6:  # Practical zero threshold
            zero_overlap_epochs += 1
            # If we've had zero overlaps for 100+ epochs, we can stop
            if zero_overlap_epochs >= 100 and epoch > 1000:
                if verbose:
                    print(f"\n🎉 Early stopping at epoch {epoch}: Zero overlaps achieved and stable!")
                stopped_early = True
                break
        else:
            zero_overlap_epochs = 0
        
        # Track best overlap loss for patience-based early stopping
        if overlap_loss.item() < best_overlap_loss * 0.99:  # Require 1% improvement
            best_overlap_loss = overlap_loss.item()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Adaptive minimum epochs based on problem size
        if N < 150:
            min_epochs = 1500
        elif N < 500:
            min_epochs = 2500
        else:
            min_epochs = 5000  # Very large problems need more time
        
        # If no improvement in overlap for a long time and we're past minimum epochs
        # For very large problems (N > 1000), disable patience-based stopping to allow full training
        if N <= 1000 and early_stop_counter >= early_stop_patience and epoch > min_epochs:
            if verbose:
                print(f"\n⚠️  Early stopping at epoch {epoch}: No overlap improvement for {early_stop_patience} epochs")
            stopped_early = True
            break

        # Log progress
        if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            current_lr = optimizer.param_groups[0]['lr']
            phase = "Phase 1 (Overlap)" if epoch < phase1_end else "Phase 2 (Balanced)" if epoch < phase2_end else "Phase 3 (Wirelength)"
            print(f"Epoch {epoch}/{num_epochs} - {phase}:")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  Wirelength Loss: {wl_loss.item():.6f}")
            print(f"  Overlap Loss: {overlap_loss.item():.6f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Overlap Weight: {overlap_weight:.2f}")
            if zero_overlap_epochs > 0:
                print(f"  ✓ Zero overlap epochs: {zero_overlap_epochs}")

    # Create final cell features
    final_cell_features = cell_features.clone()
    final_cell_features[:, 2:4] = cell_positions.detach()

    return {
        "final_cell_features": final_cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": loss_history,
        "stopped_early": stopped_early,
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
        with torch.no_grad():  # Ensure we're not tracking gradients during evaluation
            wl_loss = wirelength_attraction_loss(cell_features.detach(), pin_features.detach(), edge_list)
        
        # Check for NaN in loss
        if torch.isnan(wl_loss):
            print(f"WARNING: NaN detected in wl_loss!")
            normalized_wl = 0.0
            num_nets = edge_list.shape[0]
        else:
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

    # Initialize positions with intelligent spread to minimize initial overlaps
    total_cells = cell_features.shape[0]
    total_area = cell_features[:, 0].sum().item()
    widths = cell_features[:, 4]
    heights = cell_features[:, 5]
    
    # Calculate optimal grid size with generous spacing
    # We want cells to be well-separated initially
    avg_cell_size = (total_area / total_cells) ** 0.5
    grid_size = int(torch.ceil(torch.sqrt(torch.tensor(float(total_cells)))).item())
    
    # Calculate cell spacing to ensure no initial overlaps
    # Use 2x the average cell size as spacing for safety
    cell_spacing = avg_cell_size * 2.5
    
    # Sort cells by size (largest first) for better packing
    areas = cell_features[:, 0]
    sorted_indices = torch.argsort(areas, descending=True)
    
    # Generate positions in a grid pattern with generous spacing
    positions = torch.zeros(total_cells, 2)
    for idx, cell_idx in enumerate(sorted_indices):
        row = idx // grid_size
        col = idx % grid_size
        
        # Base grid position with generous spacing
        base_x = (col - grid_size/2) * cell_spacing
        base_y = (row - grid_size/2) * cell_spacing
        
        # Add small random offset to break symmetry
        offset_x = (torch.rand(1) - 0.5) * cell_spacing * 0.2
        offset_y = (torch.rand(1) - 0.5) * cell_spacing * 0.2
        
        positions[cell_idx, 0] = base_x + offset_x.item()
        positions[cell_idx, 1] = base_y + offset_y.item()
    
    cell_features[:, 2] = positions[:, 0]
    cell_features[:, 3] = positions[:, 1]

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
