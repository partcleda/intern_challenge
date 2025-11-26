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
import math
from enum import IntEnum

import torch
import torch.optim as optim
import torch.nn as nn


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

# ======= SURROGATE GRADIENT FUNCTIONS (SNN-inspired) =======

class FastSigmoid(torch.autograd.Function):
    """Fast sigmoid surrogate gradient for SNN-style optimization.
    
    Forward: step function (hard threshold)
    Backward: smooth gradient using fast sigmoid approximation
    
    The gradient is: 1 / (scale * |x| + 1)^2
    - Lower scale = stronger gradients for large overlaps
    - Higher scale = weaker gradients (more conservative)
    """
    @staticmethod
    def forward(ctx, input_, scale=2.0):
        ctx.save_for_backward(input_)
        ctx.scale = scale
        return (input_ > 0).type(input_.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        scale = ctx.scale
        grad_input = grad_output.clone()
        # Adaptive gradient: stronger for small overlaps, weaker for large
        # But not as weak as original (scale=10) - scale=2 gives better balance
        return grad_input / (scale * torch.abs(input_) + 1.0) ** 2, None


class SmoothStep(torch.autograd.Function):
    """Smooth step surrogate gradient.
    
    Forward: step function (hard threshold)
    Backward: box function (gradient only in [-0.5, 0.5] range)
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).type(x.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_ <= -0.5] = 0
        grad_input[input_ > 0.5] = 0
        return grad_input


class SigmoidStep(torch.autograd.Function):
    """Sigmoid step surrogate gradient.
    
    Forward: step function (hard threshold)
    Backward: sigmoid derivative (smooth gradient)
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).type(x.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        res = torch.sigmoid(input_)
        return res * (1 - res) * grad_output


# Create function instances
def fast_sigmoid(input_, scale=2.0):
    """Fast sigmoid with configurable scale parameter."""
    return FastSigmoid.apply(input_, scale)

smooth_step = SmoothStep.apply
sigmoid_step = SigmoidStep.apply

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
    """Calculate wirelength loss using smooth Euclidean distance approximation.
    
    Uses Euclidean distance (L2 norm) with a smooth approximation for differentiability,
    which naturally encourages shorter, more direct connections compared to Manhattan distance.
    The smooth approximation uses sqrt(x^2 + y^2 + eps) for numerical stability.
    
    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information
        edge_list: [E, 2] tensor with edges

    Returns:
        Scalar loss value (normalized by number of edges)
    """
    if edge_list.shape[0] == 0:
        return torch.tensor(0.0, device=cell_features.device, requires_grad=True)

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

    # Calculate Euclidean distance with smooth approximation
    # Using sqrt(x^2 + y^2 + eps) for numerical stability and differentiability
    dx = src_x - tgt_x
    dy = src_y - tgt_y
    eps = 1e-6  # Small epsilon for numerical stability
    
    # Smooth Euclidean distance: sqrt(dx^2 + dy^2 + eps)
    # This is naturally differentiable and encourages shorter, more direct connections
    # Add check to prevent NaN from invalid positions
    distance_squared = dx * dx + dy * dy
    # Clamp to prevent negative values (shouldn't happen, but safety first)
    distance_squared = torch.clamp(distance_squared, min=0.0)
    smooth_euclidean = torch.sqrt(distance_squared + eps)
    
    # Check for NaN/Inf and replace with safe value
    smooth_euclidean = torch.where(
        torch.isfinite(smooth_euclidean),
        smooth_euclidean,
        torch.zeros_like(smooth_euclidean) + eps
    )

    # Total wirelength
    total_wirelength = torch.sum(smooth_euclidean)

    return total_wirelength / edge_list.shape[0]  # Normalize by number of edges


def overlap_repulsion_loss(cell_features, pin_features, edge_list, epoch_progress=1.0):
    """Calculate overlap loss using FastSigmoid with strong squared penalty.
    
    Simplified and strengthened approach:
    - Uses FastSigmoid surrogate gradient (SNN-inspired) for smooth detection
    - Applies squared penalty to overlap area (stronger than logarithmic)
    - Uses area weighting to prioritize macro overlaps
    - Adaptive scale: stronger gradients early, sharper late
    
    Key improvements:
    1. Squared penalty (overlap_area²) instead of log - provides constant strong gradients
    2. FastSigmoid for smooth detection with adaptive sharpness
    3. Area weighting to focus on macro overlaps
    4. Simple and effective - no complex physics modeling
    
    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information (not used here)
        edge_list: [E, 2] tensor with edges (not used here)
        epoch_progress: Float in [0, 1] for adaptive sharpness (0 = start, 1 = end)

    Returns:
        Scalar loss value (should be 0 when no overlaps exist)
    """
    N = cell_features.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=cell_features.device, requires_grad=True)

    # Extract positions, dimensions, and areas
    x = cell_features[:, CellFeatureIdx.X]  # [N] - x positions
    y = cell_features[:, CellFeatureIdx.Y]  # [N] - y positions
    w = cell_features[:, CellFeatureIdx.WIDTH]  # [N] - widths
    h = cell_features[:, CellFeatureIdx.HEIGHT]  # [N] - heights
    area = cell_features[:, CellFeatureIdx.AREA]  # [N] - areas (for weighting)

    # Create broadcasted versions for pairwise computation
    xi, yi = x.unsqueeze(1), y.unsqueeze(1)  # [N, 1]
    xj, yj = x.unsqueeze(0), y.unsqueeze(0)  # [1, N]
    wi, hi, areai = w.unsqueeze(1), h.unsqueeze(1), area.unsqueeze(1)  # [N, 1]
    wj, hj, areaj = w.unsqueeze(0), h.unsqueeze(0), area.unsqueeze(0)  # [1, N]

    # Compute pairwise distances
    dx_abs = torch.abs(xi - xj)  # [N, N] - absolute x distances
    dy_abs = torch.abs(yi - yj)  # [N, N] - absolute y distances

    # Calculate minimum separation required
    min_sep_x = 0.5 * (wi + wj)  # [N, N]
    min_sep_y = 0.5 * (hi + hj)  # [N, N]
    
    # Add a small margin to encourage actual separation (not just non-overlap)
    # Margin increases as training progresses to ensure clean separation
    # Early: small margin (0.5%) to allow cells to spread
    # Late: larger margin (2%) to ensure clear separation
    min_dim_i = torch.minimum(wi, hi)  # [N, 1]
    min_dim_j = torch.minimum(wj, hj)  # [1, N]
    min_dim_pair = torch.minimum(min_dim_i, min_dim_j)  # [N, N]
    
    # Adaptive margin: 0.5% early, 2% late
    margin_factor = 0.005 + 0.015 * epoch_progress  # 0.005 -> 0.02
    margin = margin_factor * min_dim_pair  # [N, N]
    
    # Required separation = minimum separation + margin
    required_sep_x = min_sep_x + margin  # [N, N]
    required_sep_y = min_sep_y + margin  # [N, N]
    
    # Compute overlap amounts (positive when cells are too close)
    # This now includes the margin, so cells must be separated by margin to avoid penalty
    overlap_x_raw = required_sep_x - dx_abs  # [N, N] - positive when too close in x
    overlap_y_raw = required_sep_y - dy_abs  # [N, N] - positive when too close in y

    # Hybrid approach: Use ReLU for large overlaps (strong constant gradient)
    # and FastSigmoid for small overlaps (smooth detection)
    # This ensures we catch both large and tiny overlaps effectively
    
    # Threshold: use ReLU for overlaps > threshold, FastSigmoid for smaller
    # This threshold is small (0.1 units) to catch tiny overlaps
    threshold = 0.1
    
    # For large overlaps: use ReLU directly (constant gradient = 1.0)
    overlap_x_large = torch.relu(overlap_x_raw - threshold)  # [N, N] - only large overlaps
    overlap_y_large = torch.relu(overlap_y_raw - threshold)  # [N, N]
    
    # For small overlaps: use FastSigmoid (smooth detection)
    # Adaptive scale: lower = stronger gradients for small overlaps
    scale = 0.2 + 0.8 * epoch_progress  # 0.2 -> 1.0 as training progresses
    overlap_x_small_gate = fast_sigmoid(overlap_x_raw, scale=scale)  # [N, N]
    overlap_y_small_gate = fast_sigmoid(overlap_y_raw, scale=scale)  # [N, N]
    
    # Small overlap amount (only when overlap < threshold)
    overlap_x_small = overlap_x_small_gate * torch.clamp(overlap_x_raw, max=threshold)  # [N, N]
    overlap_y_small = overlap_y_small_gate * torch.clamp(overlap_y_raw, max=threshold)  # [N, N]
    
    # Combine: large overlaps (ReLU) + small overlaps (FastSigmoid)
    # This ensures we catch all overlaps with appropriate gradients
    overlap_x = overlap_x_large + overlap_x_small  # [N, N]
    overlap_y = overlap_y_large + overlap_y_small  # [N, N]
    overlap_x = torch.clamp(overlap_x, min=0.0)
    overlap_y = torch.clamp(overlap_y, min=0.0)

    # Compute overlap area
    overlap_area = overlap_x * overlap_y  # [N, N]
    
    # Squared penalty: provides constant strong gradients for all overlaps
    # This is simpler and more effective than logarithmic penalty
    # For overlap_area = k, gradient = 2k (constant multiplier)
    overlap_penalty = overlap_area * overlap_area  # [N, N]

    # Area weighting: prioritize macro overlaps (they're harder to move)
    # Use geometric mean for balanced weighting
    repulsion_strength = torch.sqrt(areai * areaj)  # [N, N]
    weighted_penalty = overlap_penalty * repulsion_strength  # [N, N]

    # Mask upper triangle to avoid double counting (i < j)
    mask_upper = torch.triu(
        torch.ones(N, N, device=cell_features.device, dtype=torch.bool), 
        diagonal=1
    )
    pair_penalties = weighted_penalty * mask_upper  # [N, N]

    # Sum all penalties
    total_penalty = torch.sum(pair_penalties)

    return total_penalty


def train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=1000,
    lr=0.1,
    lambda_wirelength=1.0,
    lambda_overlap=100.0,
    verbose=True,
    log_interval=200,
):
    """Train placement using cosine annealing schedule with balanced loss weighting.
    
    Uses a unique training strategy:
    - Cosine annealing for learning rate (smooth decay from high to low)
    - Balanced loss weighting: both wirelength and overlap from the start
    - Adaptive overlap weight: increases linearly, then plateaus
    - Problem-size adaptive hyperparameters: scales LR and lambda with N
    - No curriculum learning: treats both objectives equally from the beginning
    
    This approach is different from curriculum learning - it optimizes both
    objectives simultaneously with adaptive weighting.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity
        num_epochs: Number of optimization iterations
        lr: Base learning rate for Adam optimizer (will be scaled and cosine annealed)
        lambda_wirelength: Weight for wirelength loss
        lambda_overlap: Base weight for overlap loss (will be scaled and scheduled)
        verbose: Whether to print progress
        log_interval: How often to print progress

    Returns:
        Dictionary with:
            - final_cell_features: Optimized cell positions
            - initial_cell_features: Original cell positions (for comparison)
            - loss_history: Loss values over time
    """
    # Get device from input tensors
    device = cell_features.device
    
    # Adaptive hyperparameters based on problem size
    # Overlap loss scales quadratically with N (N² pairs), so we need to scale accordingly
    N = cell_features.shape[0]
    
    # Scale learning rate: larger problems need slightly higher LR
    # Use logarithmic scaling: lr_scale = 1.0 + 0.1 * log10(N/50)
    # For N=50: scale=1.0, N=500: scale=1.1, N=2000: scale=1.16
    lr_scale = 1.0 + 0.1 * math.log10(max(N / 50.0, 1.0))
    scaled_lr = lr * lr_scale
    
    # Scale overlap penalty: larger problems need stronger penalty
    # Overlap loss has N² pairs, so scale lambda_overlap with N
    # Use square root scaling to avoid excessive penalty: lambda_scale = sqrt(N/50)
    # For N=50: scale=1.0, N=500: scale=3.16, N=2000: scale=6.32
    lambda_scale = math.sqrt(max(N / 50.0, 1.0))
    scaled_lambda_overlap = lambda_overlap * lambda_scale
    
    # Clone features and create learnable positions - ensure on device
    cell_features = cell_features.to(device)
    pin_features = pin_features.to(device)
    edge_list = edge_list.to(device)
    
    cell_features = cell_features.clone()
    initial_cell_features = cell_features.clone()

    # Make only cell positions require gradients - ensure on device
    cell_positions = cell_features[:, 2:4].clone().detach()
    cell_positions = cell_positions.to(device)
    cell_positions.requires_grad_(True)

    # Create optimizer - optimizer state will be on same device as parameters
    # Use scaled learning rate for large problems
    optimizer = optim.Adam([cell_positions], lr=scaled_lr)
    
    # Force optimizer state to be on GPU by doing a dummy step
    if torch.cuda.is_available() and device.type == 'cuda':
        # Create a dummy loss to trigger optimizer state initialization on GPU
        dummy_loss = cell_positions.sum()
        dummy_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

    # Track loss history
    loss_history = {
        "total_loss": [],
        "wirelength_loss": [],
        "overlap_loss": [],
    }

    # Training loop with cosine annealing and adaptive overlap weighting
    for epoch in range(num_epochs):
        epoch_progress = epoch / num_epochs
        
        # Cosine annealing for learning rate: smooth decay from scaled_lr to scaled_lr/2
        # Less aggressive decay to maintain learning capacity throughout training
        # For large problems, keep LR higher for longer
        # Formula: lr_min + (lr_max - lr_min) * (1 + cos(π * progress)) / 2
        lr_min = scaled_lr * 0.5
        lr_max = scaled_lr
        current_lr = lr_min + (lr_max - lr_min) * (1 + math.cos(epoch_progress * math.pi)) / 2
        
        # Adaptive overlap weight: start immediately and ramp up quickly
        # Start overlap penalty immediately to begin separation right away
        # Ramp up to full strength by 30% progress, then maintain
        # Use scaled lambda_overlap for large problems
        if epoch_progress < 0.3:
            # First 30%: linear ramp from 0 to full strength
            ramp_progress = epoch_progress / 0.3  # 0 to 1 over this range
            current_lambda_overlap = scaled_lambda_overlap * ramp_progress
        else:
            # Last 70%: full strength
            current_lambda_overlap = scaled_lambda_overlap
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        optimizer.zero_grad()

        # Create cell_features with current positions - ensure on device
        cell_features_current = cell_features.clone().to(device)
        cell_features_current[:, 2:4] = cell_positions

        # Calculate losses
        wl_loss = wirelength_attraction_loss(
            cell_features_current, pin_features, edge_list
        )
        overlap_loss = overlap_repulsion_loss(
            cell_features_current, pin_features, edge_list, epoch_progress
        )

        # Combined loss with adaptive lambda
        total_loss = lambda_wirelength * wl_loss + current_lambda_overlap * overlap_loss
        
        # Check for NaN/Inf before backward pass
        should_skip_update = False
        if not torch.isfinite(total_loss):
            if verbose and epoch % log_interval == 0:
                print(f"WARNING: NaN/Inf detected in total_loss at epoch {epoch}")
                wl_val = wl_loss.item() if torch.isfinite(wl_loss) else float('nan')
                ol_val = overlap_loss.item() if torch.isfinite(overlap_loss) else float('nan')
                print(f"  wl_loss: {wl_val}, overlap_loss: {ol_val}")
            should_skip_update = True
        else:
            # Backward pass only if loss is finite
            total_loss.backward()
            
            # Check for NaN/Inf in gradients before clipping
            has_nan_grad = False
            if cell_positions.grad is not None:
                if not torch.isfinite(cell_positions.grad).all():
                    has_nan_grad = True
                    if verbose and epoch % log_interval == 0:
                        print(f"WARNING: NaN/Inf gradients detected at epoch {epoch}, skipping update")
            
            if not has_nan_grad:
                # Gradient clipping to prevent extreme updates
                torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=5.0)
                # Update positions
                optimizer.step()
            else:
                should_skip_update = True
                # Zero out gradients to prevent NaN propagation
                if cell_positions.grad is not None:
                    cell_positions.grad.zero_()
        
        if should_skip_update:
            # Still record losses (as NaN) for debugging, but don't update positions
            # Zero gradients to prevent accumulation
            optimizer.zero_grad()

        # Record losses (may be NaN, but that's useful for debugging)
        loss_history["total_loss"].append(total_loss.item() if torch.isfinite(total_loss) else float('nan'))
        loss_history["wirelength_loss"].append(wl_loss.item() if torch.isfinite(wl_loss) else float('nan'))
        loss_history["overlap_loss"].append(overlap_loss.item() if torch.isfinite(overlap_loss) else float('nan'))

        # Log progress
        if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch}/{num_epochs} (progress: {epoch_progress:.2f}):")
            print(f"  LR: {current_lr:.4f}, λ_overlap: {current_lambda_overlap:.2f}")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  Wirelength Loss: {wl_loss.item():.6f}")
            print(f"  Overlap Loss: {overlap_loss.item():.6f}")

    # Create final cell features
    final_cell_features = cell_features.clone()
    final_cell_features = final_cell_features.to(device)
    final_cell_features[:, 2:4] = cell_positions.detach()

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

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate placement problem
    num_macros = 3
    num_std_cells = 50

    print(f"Generating placement problem:")
    print(f"  - {num_macros} macros")
    print(f"  - {num_std_cells} standard cells")

    cell_features, pin_features, edge_list = generate_placement_input(
        num_macros, num_std_cells
    )
    
    # Move tensors to GPU if available
    cell_features = cell_features.to(device)
    pin_features = pin_features.to(device)
    edge_list = edge_list.to(device)

    # Initialize positions with random spread to reduce initial overlaps
    total_cells = cell_features.shape[0]
    spread_radius = 30.0
    # Create random tensors on the same device as cell_features
    angles = torch.rand(total_cells, device=device) * 2 * 3.14159
    radii = torch.rand(total_cells, device=device) * spread_radius

    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)

    # Calculate initial metrics (move to CPU for numpy operations)
    print("\n" + "=" * 70)
    print("INITIAL STATE")
    print("=" * 70)
    cell_features_cpu = cell_features.cpu() if cell_features.is_cuda else cell_features
    initial_metrics = calculate_overlap_metrics(cell_features_cpu)
    print(f"Overlap count: {initial_metrics['overlap_count']}")
    print(f"Total overlap area: {initial_metrics['total_overlap_area']:.2f}")
    print(f"Max overlap area: {initial_metrics['max_overlap_area']:.2f}")
    print(f"Overlap percentage: {initial_metrics['overlap_percentage']:.2f}%")

    # Run optimization
    print("\n" + "=" * 70)
    print("RUNNING OPTIMIZATION")
    print("=" * 70)
    
    # Time the optimization
    import time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)
    start_time = time.time()
    
    result = train_placement(
        cell_features,
        pin_features,
        edge_list,
        num_epochs=1000,
        lr=0.1,  # Maximum LR for cosine annealing (will decay smoothly)
        lambda_wirelength=1.0,
        lambda_overlap=100.0,  # Maximum overlap penalty (increased for stronger push)
        verbose=True,
        log_interval=100,
    )
    
    # Measure elapsed time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print(f"\nOptimization completed in {elapsed_time:.2f} seconds")

    # Calculate final metrics (both detailed and normalized)
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    final_cell_features = result["final_cell_features"]
    
    # Move back to CPU for evaluation (if on GPU)
    final_cell_features_cpu = final_cell_features.cpu() if final_cell_features.is_cuda else final_cell_features
    pin_features_cpu = pin_features.cpu() if pin_features.is_cuda else pin_features
    edge_list_cpu = edge_list.cpu() if edge_list.is_cuda else edge_list

    # Detailed metrics
    final_metrics = calculate_overlap_metrics(final_cell_features_cpu)
    print(f"Overlap count (pairs): {final_metrics['overlap_count']}")
    print(f"Total overlap area: {final_metrics['total_overlap_area']:.2f}")
    print(f"Max overlap area: {final_metrics['max_overlap_area']:.2f}")

    # Normalized metrics (matching test suite)
    print("\n" + "-" * 70)
    print("TEST SUITE METRICS (for leaderboard)")
    print("-" * 70)
    normalized_metrics = calculate_normalized_metrics(
        final_cell_features_cpu, pin_features_cpu, edge_list_cpu
    )
    print(f"Overlap Ratio: {normalized_metrics['overlap_ratio']:.4f} "
          f"({normalized_metrics['num_cells_with_overlaps']}/{normalized_metrics['total_cells']} cells)")
    print(f"Normalized Wirelength: {normalized_metrics['normalized_wl']:.4f}")

    # Success check with validation
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA")
    print("=" * 70)
    
    # Validate metrics for NaN/Inf values and invalid states
    import math
    
    # Check if metrics are valid numbers
    normalized_wl = normalized_metrics['normalized_wl']
    overlap_ratio = normalized_metrics['overlap_ratio']
    num_cells_with_overlaps = normalized_metrics['num_cells_with_overlaps']
    
    # Check for NaN or Inf in metrics
    has_invalid_metrics = (
        not math.isfinite(normalized_wl) or 
        not math.isfinite(overlap_ratio) or
        normalized_wl < 0 or  # Wirelength should be non-negative
        overlap_ratio < 0 or overlap_ratio > 1  # Overlap ratio should be in [0, 1]
    )
    
    # Check if final cell positions are valid (no NaN/Inf)
    final_positions = final_cell_features_cpu[:, 2:4].detach().numpy()
    has_invalid_positions = (
        not math.isfinite(final_positions.min()) or 
        not math.isfinite(final_positions.max())
    )
    
    # Check if loss history indicates broken optimization
    # If all losses are NaN, optimization was completely broken
    loss_history = result.get("loss_history", {})
    total_losses = loss_history.get("total_loss", [])
    has_broken_optimization = False
    nan_count = 0
    if total_losses:
        # Check if more than 50% of losses are NaN (indicating broken optimization)
        nan_count = sum(1 for loss in total_losses if not math.isfinite(loss))
        has_broken_optimization = (nan_count / len(total_losses)) > 0.5
    
    # Determine if solution is valid
    is_valid_solution = (
        not has_invalid_metrics and 
        not has_invalid_positions and 
        not has_broken_optimization
    )
    
    # Only pass if solution is valid AND no overlaps exist
    if is_valid_solution and num_cells_with_overlaps == 0:
        print("✓ PASS: No overlapping cells!")
        print("✓ PASS: Overlap ratio is 0.0")
        print("✓ PASS: All metrics are valid (no NaN/Inf)")
        print("\nCongratulations! Your implementation successfully eliminated all overlaps.")
        print(f"Your normalized wirelength: {normalized_wl:.4f}")
    else:
        # Determine failure reason
        failure_reasons = []
        
        if num_cells_with_overlaps > 0:
            failure_reasons.append(f"Overlaps still exist ({num_cells_with_overlaps} cells)")
        
        if has_invalid_metrics:
            failure_reasons.append("Invalid metrics detected (NaN/Inf values)")
            if not math.isfinite(normalized_wl):
                print(f"  ⚠ WARNING: Normalized wirelength is {normalized_wl}")
            if not math.isfinite(overlap_ratio):
                print(f"  ⚠ WARNING: Overlap ratio is {overlap_ratio}")
        
        if has_invalid_positions:
            failure_reasons.append("Invalid cell positions detected (NaN/Inf)")
            print(f"  ⚠ WARNING: Final cell positions contain NaN or Inf values")
        
        if has_broken_optimization:
            failure_reasons.append("Optimization appears broken (loss values are NaN)")
            print(f"  ⚠ WARNING: {nan_count}/{len(total_losses)} loss values are NaN/Inf")
            print(f"  This indicates numerical instability in your loss function")
        
        print("✗ FAIL: Solution does not meet success criteria")
        for reason in failure_reasons:
            print(f"  - {reason}")
        
        print("\nSuggestions:")
        if has_invalid_metrics or has_broken_optimization:
            print("  1. Check for numerical instability in your loss functions")
            print("     - Avoid operations that can produce NaN/Inf (e.g., exp() of large values)")
            print("     - Add numerical stability checks (clamping, epsilon values)")
            print("     - Verify gradients are finite before optimizer.step()")
        if num_cells_with_overlaps > 0:
            print("  2. Check your overlap_repulsion_loss() implementation")
            print("  3. Try increasing lambda_overlap or adjusting learning rate")

    # Generate visualization (move to CPU if needed)
    initial_cell_features_cpu = result["initial_cell_features"].cpu() if result["initial_cell_features"].is_cuda else result["initial_cell_features"]
    final_cell_features_cpu_viz = final_cell_features.cpu() if final_cell_features.is_cuda else final_cell_features
    plot_placement(
        initial_cell_features_cpu,
        final_cell_features_cpu_viz,
        pin_features_cpu,
        edge_list_cpu,
        filename="placement_result.png",
    )

if __name__ == "__main__":
    main()
