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
import time
from enum import IntEnum

import torch
import torch.optim as optim
import torch.nn as nn

# Profiling / debug flags (enable via environment variables)
PROFILE_PLACEMENT = bool(int(os.environ.get("PROFILE_PLACEMENT", "0")))
PROFILE_OVERLAP = PROFILE_PLACEMENT or bool(int(os.environ.get("PROFILE_OVERLAP", "0")))
DEBUG_CUDA_OVERLAP = bool(int(os.environ.get("CUDA_OVERLAP_DEBUG", "0")))
FORCE_CPU_OVERLAP = bool(int(os.environ.get("FORCE_CPU_OVERLAP", "0")))
LAST_OVERLAP_PROFILE = {}

# Try to import CUDA-accelerated overlap loss (falls back to PyTorch implementation if unavailable)
try:
    from cuda_backend import (
        compute_overlap_loss as cuda_overlap_loss,
        is_available as cuda_overlap_available,
        get_last_stats as cuda_overlap_stats,
    )
    CUDA_OVERLAP_AVAILABLE = cuda_overlap_available()
except ImportError:
    cuda_overlap_loss = None
    cuda_overlap_stats = lambda: {}
    CUDA_OVERLAP_AVAILABLE = False


def overlap_repulsion_loss(cell_features, pin_features, edge_list, epoch_progress=1.0):
    """Main entry point for overlap loss."""
    return overlap_repulsion_loss_original(
        cell_features, pin_features, edge_list, epoch_progress
    )


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
    
    NOTE: This has INVERSE gradient scaling (weak gradients for large overlaps),
    which is why we prefer Softplus for overlap loss.
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


def strong_fast_sigmoid(input_, scale=1.0, alpha=1.0):
    """Wrapper function for StrongFastSigmoid autograd function."""
    return StrongFastSigmoid.apply(input_, scale, alpha)


class StrongFastSigmoid(torch.autograd.Function):
    """Custom surrogate gradient with magnitude-aware scaling for aggressive overlap elimination.
    
    This is a unique implementation inspired by Softplus but with custom gradient characteristics:
    - Forward: smooth activation similar to Softplus but with custom scaling
    - Backward: sigmoid-based gradient that scales with magnitude (like Softplus)
    
    Key innovation: Uses scaled sigmoid gradient: scale * sigmoid(alpha * x) where alpha
    increases with training progress, ensuring stronger gradients for larger overlaps.
    
    Unlike original FastSigmoid (inverse scaling), this gives STRONGER gradients for LARGER overlaps.
    More aggressive than standard Softplus due to adaptive alpha scaling.
    """
    @staticmethod
    def forward(ctx, input_, scale=1.0, alpha=1.0):
        ctx.save_for_backward(input_)
        ctx.scale = scale
        ctx.alpha = alpha
        # Forward: smooth activation similar to Softplus but with custom scaling
        # Use scaled softplus: log(1 + exp(alpha * x)) / alpha
        # This gives smooth, magnitude-preserving output
        # Clamp alpha to avoid division by zero
        alpha_safe = max(float(alpha), 1e-8)
        # Clamp input to avoid overflow in softplus (softplus can overflow for inputs > 50)
        input_clamped = torch.clamp(input_, min=-50.0, max=50.0)
        scaled_input = input_clamped * alpha_safe
        # Clamp scaled_input again to be extra safe
        scaled_input = torch.clamp(scaled_input, min=-50.0, max=50.0)
        result = torch.nn.functional.softplus(scaled_input) / alpha_safe
        # Check for NaN/Inf and replace with zero
        result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        scale = ctx.scale
        alpha = ctx.alpha
        
        # Gradient: scale * sigmoid(alpha * x) * (1 + magnitude_boost)
        # Add magnitude boost for larger overlaps to make loss curve steeper
        # This ensures large overlaps get exponentially stronger gradients
        # Clamp alpha to avoid numerical issues
        alpha_safe = max(alpha, 1e-8)
        # Clamp input to avoid overflow
        input_clamped = torch.clamp(input_, min=-50.0, max=50.0)
        scaled_input = input_clamped * alpha_safe
        sigmoid_grad = torch.sigmoid(scaled_input)  # Base gradient (like Softplus)
        
        # Magnitude boost: for large overlaps, add extra gradient scaling
        # Use tanh to smoothly boost gradients for larger overlaps
        # Clamp input for tanh to avoid overflow
        input_for_tanh = torch.clamp(input_ * 0.5, min=-10.0, max=10.0)
        magnitude_boost = 1.0 + 2.0 * torch.tanh(input_for_tanh)  # 1.0 -> 3.0 for large overlaps
        
        grad_input = grad_output * scale * sigmoid_grad * magnitude_boost
        
        # Check for NaN/Inf and replace with zero
        grad_input = torch.where(torch.isfinite(grad_input), grad_input, torch.zeros_like(grad_input))
        
        return grad_input, None, None


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
    """Calculate wirelength loss using optimized vectorized operations.
    
    Optimized for GPU performance with minimal overhead:
    - Single-pass computation with fused operations
    - Efficient tensor indexing
    - No intermediate tensor allocations

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

    # Calculate absolute pin positions (single indexing operation)
    pin_absolute_x = cell_positions[cell_indices, 0] + pin_features[:, 1]
    pin_absolute_y = cell_positions[cell_indices, 1] + pin_features[:, 2]

    # Get source and target pin positions for each edge (single indexing)
    src_pins = edge_list[:, 0].long()
    tgt_pins = edge_list[:, 1].long()

    src_x = pin_absolute_x[src_pins]
    src_y = pin_absolute_y[src_pins]
    tgt_x = pin_absolute_x[tgt_pins]
    tgt_y = pin_absolute_y[tgt_pins]

    # Calculate Euclidean distance with smooth approximation (fused)
    dx = src_x - tgt_x
    dy = src_y - tgt_y
    eps = 1e-6
    
    # Fused: distance_squared + eps, then sqrt (single kernel)
    distance_squared = dx * dx + dy * dy
    smooth_euclidean = torch.sqrt(distance_squared + eps)

    # Total wirelength (single reduction)
    total_wirelength = torch.sum(smooth_euclidean)

    return total_wirelength / edge_list.shape[0]  # Normalize by number of edges


def overlap_repulsion_loss_original(cell_features, pin_features, edge_list, epoch_progress=1.0):
    """Calculate overlap loss using optimized GPU-friendly vectorized computation.
    
    Uses spatial filtering for large problems to skip distant pairs, but processes
    all valid pairs in fully vectorized operations for optimal GPU utilization.
    No sequential chunking - everything is vectorized for GPU efficiency.
    
    Key optimizations:
    1. Spatial filtering: only checks pairs within reasonable distance (vectorized)
    2. Pure ReLU for most overlaps (threshold=0.001) - strong constant gradient
    3. FastSigmoid only for tiny overlaps (< 0.001) for fine-tuning
    4. Fully vectorized operations - no sequential processing
    5. Modular code to avoid duplication

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

    device = cell_features.device
    global LAST_OVERLAP_PROFILE
    if PROFILE_OVERLAP:
        torch.cuda.synchronize() if device.type == "cuda" else None
        overlap_prof_start = time.perf_counter()
        overlap_host_time = 0.0
        overlap_kernel_time = 0.0
    else:
        overlap_prof_start = None
    
    # Extract positions, dimensions, and areas
    x = cell_features[:, CellFeatureIdx.X]  # [N]
    y = cell_features[:, CellFeatureIdx.Y]  # [N]
    w = cell_features[:, CellFeatureIdx.WIDTH]  # [N]
    h = cell_features[:, CellFeatureIdx.HEIGHT]  # [N]
    area = cell_features[:, CellFeatureIdx.AREA]  # [N]

    # Adaptive margin: more aggressive to ensure complete separation
    # Start at 1% and increase to 3% to force cells apart (not just touching)
    margin_factor = 0.01 + 0.02 * epoch_progress  # 0.01 -> 0.03 (more aggressive)
    
    # Use custom StrongFastSigmoid for ALL overlaps - unique implementation with aggressive gradients
    # This provides magnitude-aware gradients that INCREASE with overlap size
    # Unlike original FastSigmoid (inverse scaling), this gives STRONGER gradients for LARGER overlaps
    # 
    # Key features:
    # - Forward: scaled softplus activation (smooth, magnitude-preserving)
    # - Backward: sigmoid-based gradient that scales with magnitude (like Softplus)
    # - Alpha parameter controls gradient sharpness (higher = stronger gradients)
    # - More aggressive than standard Softplus due to adaptive alpha scaling
    
    # Adaptive scaling: stronger as training progresses and for larger problems
    base_scale = 3.0 + 12.0 * epoch_progress  # 3.0 -> 15.0 (stronger as training progresses)
    if N > 5000:
        # For very large problems, need even stronger gradients
        scale = base_scale * 1.3  # 30% stronger for large problems
    else:
        scale = base_scale
    
    # Alpha parameter: controls gradient sharpness (higher = stronger gradients)
    # Higher alpha = stronger gradients for larger overlaps (more aggressive)
    # Start moderate, increase aggressively over time
    alpha = 3.0 + 12.0 * epoch_progress  # 3.0 -> 15.0 (more aggressive over time)
    
    # Fast CUDA path (only when extension is available and tensors are on GPU)
    if CUDA_OVERLAP_AVAILABLE and device.type == "cuda" and not FORCE_CPU_OVERLAP:
        try:
            if PROFILE_OVERLAP:
                host_prep_end = time.perf_counter()
                overlap_host_time = host_prep_end - overlap_prof_start
                torch.cuda.synchronize()
                kernel_start = time.perf_counter()
                loss = cuda_overlap_loss(
                    cell_features,
                    margin_factor,
                    alpha,
                    scale,
                    epoch_progress,
                )
                torch.cuda.synchronize()
                overlap_kernel_time = time.perf_counter() - kernel_start
                LAST_OVERLAP_PROFILE = {
                    "backend": "cuda",
                    "host": overlap_host_time,
                    "kernel": overlap_kernel_time,
                }
            else:
                loss = cuda_overlap_loss(
                    cell_features,
                    margin_factor,
                    alpha,
                    scale,
                    epoch_progress,
                )
                LAST_OVERLAP_PROFILE = {"backend": "cuda", "host": 0.0, "kernel": 0.0}
            if DEBUG_CUDA_OVERLAP:
                stats = cuda_overlap_stats()
                print(
                    f"[CUDA-OVERLAP] backend used "
                    f"(pairs={stats.get('pairs')}, bin_size={stats.get('bin_size', 0.0):.3f})"
                )
            return loss
        except RuntimeError as exc:
            # Fall back to PyTorch implementation on failure
            if DEBUG_CUDA_OVERLAP:
                print(f"[CUDA-OVERLAP] fallback to CPU due to: {exc}")
            pass
    elif DEBUG_CUDA_OVERLAP and FORCE_CPU_OVERLAP:
        print("[CUDA-OVERLAP] FORCE_CPU_OVERLAP=1 -> using PyTorch implementation")
    
    # Simplified, optimized approach: Minimize overhead, maximize GPU utilization
    # Key strategy: Use larger chunks, simpler operations, aggressive spatial filtering
    # For very large problems (100K+), use spatial hashing to reduce O(N²) to O(N)
    if N > 50000:
        # For 100K+ cells: Use spatial hashing with larger bins (fewer neighbors)
        USE_SPATIAL_HASHING = True
        BIN_SIZE = 100.0  # Larger bins = fewer neighbors to check
        MAX_NEIGHBORS_PER_BIN = 300  # Allow more neighbors per bin
        CHUNK_SIZE_I = 200  # Larger chunks = fewer kernel launches
        CHUNK_SIZE_J = 200
    elif N > 20000:
        USE_SPATIAL_HASHING = False
        CHUNK_SIZE_I = 1000  # Much larger chunks = much fewer kernel launches
        CHUNK_SIZE_J = 2000
        USE_DOUBLE_CHUNKING = False
        CLEAR_CACHE_BETWEEN_CHUNKS = False
    elif N > 5000:
        USE_SPATIAL_HASHING = False
        CHUNK_SIZE_I = 1500  # Very large chunks for medium problems
        CHUNK_SIZE_J = 3000
        USE_DOUBLE_CHUNKING = False
        CLEAR_CACHE_BETWEEN_CHUNKS = False
    else:
        USE_SPATIAL_HASHING = False
        CHUNK_SIZE_I = 2000  # Maximum chunk size for small problems
        CHUNK_SIZE_J = 5000
        USE_DOUBLE_CHUNKING = False
        CLEAR_CACHE_BETWEEN_CHUNKS = False
    
    USE_CHUNKING = N > 2000  # Only chunk for problems > 2000 cells (reduce overhead)
    
    # For very large problems, use spatial hashing: O(N) instead of O(N²)
    if USE_SPATIAL_HASHING:
        # Spatial hashing: bin cells by their positions
        # Only check pairs within the same bin or adjacent bins
        # Use float32 throughout for numerical stability and performance
        total_penalty = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        
        # Compute spatial bounds
        x_min = x.min().item()
        x_max = x.max().item()
        y_min = y.min().item()
        y_max = y.max().item()
        
        # Create spatial bins
        num_bins_x = max(1, int((x_max - x_min) / BIN_SIZE) + 1)
        num_bins_y = max(1, int((y_max - y_min) / BIN_SIZE) + 1)
        
        # Bin cells by their positions
        bin_x = ((x - x_min) / BIN_SIZE).long().clamp(0, num_bins_x - 1)
        bin_y = ((y - y_min) / BIN_SIZE).long().clamp(0, num_bins_y - 1)
        bin_idx = bin_y * num_bins_x + bin_x  # Flattened bin index
        
        # Group cells by bin
        sorted_indices = torch.argsort(bin_idx)
        sorted_bin_idx = bin_idx[sorted_indices]
        
        # Find bin boundaries
        unique_bins, bin_counts = torch.unique_consecutive(sorted_bin_idx, return_counts=True)
        bin_starts = torch.cat([torch.tensor([0], device=device), bin_counts.cumsum(0)[:-1]])
        
        # Process each bin and its neighbors
        for bin_id in unique_bins:
            bin_start = bin_starts[bin_id == unique_bins][0].item()
            bin_count = bin_counts[bin_id == unique_bins][0].item()
            
            if bin_count == 0:
                continue
            
            # Get cells in this bin
            bin_cell_indices = sorted_indices[bin_start:bin_start + bin_count]
            if len(bin_cell_indices) > MAX_NEIGHBORS_PER_BIN:
                # If too many cells in bin, randomly sample
                perm = torch.randperm(len(bin_cell_indices), device=device)[:MAX_NEIGHBORS_PER_BIN]
                bin_cell_indices = bin_cell_indices[perm]
            
            # Get neighbor bins (same bin + 8 adjacent bins)
            bin_x_coord = (bin_id % num_bins_x).item()
            bin_y_coord = (bin_id // num_bins_x).item()
            
            neighbor_bins = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx = bin_x_coord + dx
                    ny = bin_y_coord + dy
                    if 0 <= nx < num_bins_x and 0 <= ny < num_bins_y:
                        neighbor_bins.append(ny * num_bins_x + nx)
            
            # Collect all neighbor cells
            neighbor_indices_list = []
            for nbin_id in neighbor_bins:
                if nbin_id in unique_bins:
                    nbin_pos = (unique_bins == nbin_id).nonzero(as_tuple=True)[0]
                    if len(nbin_pos) > 0:
                        nbin_start = bin_starts[nbin_pos[0]].item()
                        nbin_count = bin_counts[nbin_pos[0]].item()
                        nbin_cells = sorted_indices[nbin_start:nbin_start + nbin_count]
                        if len(nbin_cells) > MAX_NEIGHBORS_PER_BIN:
                            perm = torch.randperm(len(nbin_cells), device=device)[:MAX_NEIGHBORS_PER_BIN]
                            nbin_cells = nbin_cells[perm]
                        neighbor_indices_list.append(nbin_cells)
            
            if not neighbor_indices_list:
                continue
            
            neighbor_indices = torch.cat(neighbor_indices_list)
            neighbor_indices = torch.unique(neighbor_indices)  # Remove duplicates
            
            # Process in chunks to avoid large tensors
            for i_chunk_start in range(0, len(bin_cell_indices), CHUNK_SIZE_I):
                i_chunk_end = min(i_chunk_start + CHUNK_SIZE_I, len(bin_cell_indices))
                i_chunk = bin_cell_indices[i_chunk_start:i_chunk_end]
                
                for j_chunk_start in range(0, len(neighbor_indices), CHUNK_SIZE_J):
                    j_chunk_end = min(j_chunk_start + CHUNK_SIZE_J, len(neighbor_indices))
                    j_chunk = neighbor_indices[j_chunk_start:j_chunk_end]
                    
                    # Ensure j > i (upper triangle)
                    i_broadcast = i_chunk.unsqueeze(1)
                    j_broadcast = j_chunk.unsqueeze(0)
                    valid_mask = j_broadcast > i_broadcast
                    
                    if not valid_mask.any():
                        continue
                    
                    # Get cell data
                    xi = x[i_chunk].unsqueeze(1)
                    yi = y[i_chunk].unsqueeze(1)
                    wi = w[i_chunk].unsqueeze(1)
                    hi = h[i_chunk].unsqueeze(1)
                    areai = area[i_chunk].unsqueeze(1)
                    
                    xj = x[j_chunk].unsqueeze(0)
                    yj = y[j_chunk].unsqueeze(0)
                    wj = w[j_chunk].unsqueeze(0)
                    hj = h[j_chunk].unsqueeze(0)
                    areaj = area[j_chunk].unsqueeze(0)
                    
                    # Compute overlaps
                    dx_abs = torch.abs(xi - xj)
                    dy_abs = torch.abs(yi - yj)
                    
                    min_sep_x = 0.5 * (wi + wj)
                    min_sep_y = 0.5 * (hi + hj)
                    
                    min_dim_i = torch.minimum(wi, hi)
                    min_dim_j = torch.minimum(wj, hj)
                    min_dim_pair = torch.minimum(min_dim_i, min_dim_j)
                    margin = margin_factor * min_dim_pair
                    
                    required_sep_x = min_sep_x + margin
                    required_sep_y = min_sep_y + margin
                    
                    overlap_x_raw = required_sep_x - dx_abs
                    overlap_y_raw = required_sep_y - dy_abs
                    
                    # Use StrongFastSigmoid
                    overlap_x = strong_fast_sigmoid(overlap_x_raw, scale=scale, alpha=alpha)
                    overlap_y = strong_fast_sigmoid(overlap_y_raw, scale=scale, alpha=alpha)
                    
                    overlap_x = torch.clamp(overlap_x, min=0.0)
                    overlap_y = torch.clamp(overlap_y, min=0.0)
                    
                    overlap_area = overlap_x * overlap_y
                    overlap_area_safe = torch.clamp(overlap_area, min=1e-8)
                    overlap_penalty = torch.pow(overlap_area_safe, 2.5)
                    
                    repulsion_strength = torch.sqrt(areai * areaj)
                    weighted_penalty = overlap_penalty * repulsion_strength
                    
                    weighted_penalty = weighted_penalty * valid_mask.float()
                    chunk_penalty = torch.sum(weighted_penalty)
                    # Convert to float32 if needed and check for NaN/Inf
                    if chunk_penalty.dtype != torch.float32:
                        chunk_penalty = chunk_penalty.float()
                    chunk_penalty = torch.where(torch.isfinite(chunk_penalty), chunk_penalty, torch.tensor(0.0, device=device, dtype=torch.float32))
                    total_penalty = total_penalty + chunk_penalty
                    
                    # Don't clear cache - it's expensive and hurts performance
                    # Let PyTorch manage memory automatically
        
        # Final check: ensure result is finite
        if not torch.isfinite(total_penalty):
            total_penalty = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        
        if PROFILE_OVERLAP and overlap_prof_start is not None:
            torch.cuda.synchronize() if device.type == "cuda" else None
            LAST_OVERLAP_PROFILE = {
                "backend": "cpu",
                "host": time.perf_counter() - overlap_prof_start,
                "kernel": 0.0,
            }
        else:
            LAST_OVERLAP_PROFILE = None
        return total_penalty
    
    elif USE_CHUNKING:
        # Chunked processing: process i cells at a time against all j cells
        # Each chunk is fully vectorized (GPU-friendly), but we avoid full N×N matrix
        total_penalty = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Pre-compute max cell size for spatial filtering
        # Adaptive spatial filtering: more aggressive for very large problems to save memory
        max_w = torch.max(w)
        max_h = torch.max(h)
        max_cell_size = torch.max(max_w, max_h)
        
        # Adaptive spatial filter distance based on problem size
        # For very large problems, use more aggressive filtering to skip distant pairs
        if N > 50000:
            # Very aggressive: only check pairs within 3x max cell size (saves memory)
            spatial_filter_dist = 3.0 * max_cell_size + 50.0
        elif N > 20000:
            # Aggressive: check pairs within 4x max cell size
            spatial_filter_dist = 4.0 * max_cell_size + 75.0
        else:
            # Conservative: check pairs within 5x max cell size (catches all overlaps)
            spatial_filter_dist = 5.0 * max_cell_size + 100.0
        
        # Process cells in chunks
        # For each chunk of i cells, we need to process against all j > i
        # To ensure we only process upper triangle (i < j), we need to be careful
        for i_start in range(0, N, CHUNK_SIZE_I):
            i_end = min(i_start + CHUNK_SIZE_I, N)
            i_indices = torch.arange(i_start, i_end, device=device)
            
            # Get chunk data for i cells
            xi_chunk = x[i_indices]  # [chunk_i]
            yi_chunk = y[i_indices]  # [chunk_i]
            wi_chunk = w[i_indices]  # [chunk_i]
            hi_chunk = h[i_indices]  # [chunk_i]
            areai_chunk = area[i_indices]  # [chunk_i]
            
            # For each i in this chunk, we need to process j > i
            j_start = i_start + 1
            if j_start >= N:
                break
            
            # For very large problems, use double chunking (also chunk j dimension)
            if USE_DOUBLE_CHUNKING:
                # Process j in chunks too to avoid creating huge tensors
                for j_chunk_start in range(j_start, N, CHUNK_SIZE_J):
                    j_chunk_end = min(j_chunk_start + CHUNK_SIZE_J, N)
                    j_indices = torch.arange(j_chunk_start, j_chunk_end, device=device)
                    
                    # Get data for j chunk
                    xj_chunk = x[j_indices]  # [chunk_j]
                    yj_chunk = y[j_indices]  # [chunk_j]
                    wj_chunk = w[j_indices]  # [chunk_j]
                    hj_chunk = h[j_indices]  # [chunk_j]
                    areaj_chunk = area[j_indices]  # [chunk_j]
                    
                    # Quick spatial pre-filter: check if any j cell is within range
                    # For very large problems, move to CPU to avoid GPU memory pressure
                    # This avoids creating large tensors for distant cells
                    if CLEAR_CACHE_BETWEEN_CHUNKS:
                        # Use CPU for bounding box checks to save GPU memory
                        xi_min = xi_chunk.min().cpu().item()
                        xi_max = xi_chunk.max().cpu().item()
                        yi_min = yi_chunk.min().cpu().item()
                        yi_max = yi_chunk.max().cpu().item()
                        
                        xj_min = xj_chunk.min().cpu().item()
                        xj_max = xj_chunk.max().cpu().item()
                        yj_min = yj_chunk.min().cpu().item()
                        yj_max = yj_chunk.max().cpu().item()
                    else:
                        # For smaller problems, use GPU (faster)
                        xi_min = xi_chunk.min().item()
                        xi_max = xi_chunk.max().item()
                        yi_min = yi_chunk.min().item()
                        yi_max = yi_chunk.max().item()
                        
                        xj_min = xj_chunk.min().item()
                        xj_max = xj_chunk.max().item()
                        yj_min = yj_chunk.min().item()
                        yj_max = yj_chunk.max().item()
                    
                    # Quick bounding box check (all scalars now, no GPU tensors)
                    if (xj_max < xi_min - spatial_filter_dist.item() or 
                        xj_min > xi_max + spatial_filter_dist.item() or
                        yj_max < yi_min - spatial_filter_dist.item() or
                        yj_min > yi_max + spatial_filter_dist.item()):
                        continue  # Skip this j chunk entirely
                    
                    # Broadcast for pairwise computation: [chunk_i, chunk_j]
                    xi_broadcast = xi_chunk.unsqueeze(1)  # [chunk_i, 1]
                    yi_broadcast = yi_chunk.unsqueeze(1)  # [chunk_i, 1]
                    xj_broadcast = xj_chunk.unsqueeze(0)  # [1, chunk_j]
                    yj_broadcast = yj_chunk.unsqueeze(0)  # [1, chunk_j]
                    
                    # Broadcast dimensions
                    wi_broadcast = wi_chunk.unsqueeze(1)  # [chunk_i, 1]
                    hi_broadcast = hi_chunk.unsqueeze(1)  # [chunk_i, 1]
                    wj_broadcast = wj_chunk.unsqueeze(0)  # [1, chunk_j]
                    hj_broadcast = hj_chunk.unsqueeze(0)  # [1, chunk_j]
                    areai_broadcast = areai_chunk.unsqueeze(1)  # [chunk_i, 1]
                    areaj_broadcast = areaj_chunk.unsqueeze(0)  # [1, chunk_j]
                    
                    # Create upper triangle mask: ensure j > i
                    i_broadcast = i_indices.unsqueeze(1)  # [chunk_i, 1]
                    j_broadcast = j_indices.unsqueeze(0)  # [1, chunk_j]
                    upper_triangle_mask = j_broadcast > i_broadcast  # [chunk_i, chunk_j]
                    
                    # Spatial filtering
                    dist_x = torch.abs(xi_broadcast - xj_broadcast)  # [chunk_i, chunk_j]
                    dist_y = torch.abs(yi_broadcast - yj_broadcast)  # [chunk_i, chunk_j]
                    max_dist = torch.maximum(dist_x, dist_y)  # [chunk_i, chunk_j]
                    spatial_mask = max_dist < spatial_filter_dist  # [chunk_i, chunk_j]
                    
                    valid_mask = upper_triangle_mask & spatial_mask  # [chunk_i, chunk_j]
                    
                    if not valid_mask.any():
                        continue
                    
                    # Compute pairwise distances
                    dx_abs = torch.abs(xi_broadcast - xj_broadcast)  # [chunk_i, chunk_j]
                    dy_abs = torch.abs(yi_broadcast - yj_broadcast)  # [chunk_i, chunk_j]
                    
                    # Calculate minimum separation and margin
                    min_sep_x = 0.5 * (wi_broadcast + wj_broadcast)  # [chunk_i, chunk_j]
                    min_sep_y = 0.5 * (hi_broadcast + hj_broadcast)  # [chunk_i, chunk_j]
                    
                    min_dim_i = torch.minimum(wi_broadcast, hi_broadcast)  # [chunk_i, 1]
                    min_dim_j = torch.minimum(wj_broadcast, hj_broadcast)  # [1, chunk_j]
                    min_dim_pair = torch.minimum(min_dim_i, min_dim_j)  # [chunk_i, chunk_j]
                    margin = margin_factor * min_dim_pair  # [chunk_i, chunk_j]
                    
                    required_sep_x = min_sep_x + margin  # [chunk_i, chunk_j]
                    required_sep_y = min_sep_y + margin  # [chunk_i, chunk_j]
                    
                    # Compute overlap amounts
                    overlap_x_raw = required_sep_x - dx_abs  # [chunk_i, chunk_j]
                    overlap_y_raw = required_sep_y - dy_abs  # [chunk_i, chunk_j]
                    
                    # Use custom StrongFastSigmoid for ALL overlaps
                    overlap_x = strong_fast_sigmoid(overlap_x_raw, scale=scale, alpha=alpha)
                    overlap_y = strong_fast_sigmoid(overlap_y_raw, scale=scale, alpha=alpha)
                    
                    overlap_x = torch.clamp(overlap_x, min=0.0)
                    overlap_y = torch.clamp(overlap_y, min=0.0)
                    
                    # Power-law penalty
                    overlap_area = overlap_x * overlap_y  # [chunk_i, chunk_j]
                    overlap_area_safe = torch.clamp(overlap_area, min=1e-8)
                    overlap_penalty = torch.pow(overlap_area_safe, 2.5)
                    
                    # Area weighting
                    repulsion_strength = torch.sqrt(areai_broadcast * areaj_broadcast)  # [chunk_i, chunk_j]
                    weighted_penalty = overlap_penalty * repulsion_strength  # [chunk_i, chunk_j]
                    
                    # Apply mask
                    weighted_penalty = weighted_penalty * valid_mask.float()
                    chunk_penalty = torch.sum(weighted_penalty)
                    # Check for NaN/Inf before accumulating
                    if torch.isfinite(chunk_penalty):
                        total_penalty = total_penalty + chunk_penalty
                    
                    # Don't clear cache - it's expensive and hurts performance
                    # Let PyTorch manage memory automatically
            else:
                # Single chunking: process all j cells at once (for smaller problems)
                j_indices = torch.arange(j_start, N, device=device)
                
                # Get data for all j cells
                xj_all = x[j_indices]  # [num_j]
                yj_all = y[j_indices]  # [num_j]
                wj_all = w[j_indices]  # [num_j]
                hj_all = h[j_indices]  # [num_j]
                areaj_all = area[j_indices]  # [num_j]
                
                # Broadcast for pairwise computation: [chunk_i, num_j]
                # Do this FIRST before spatial filtering
                xi_broadcast = xi_chunk.unsqueeze(1)  # [chunk_i, 1]
                yi_broadcast = yi_chunk.unsqueeze(1)  # [chunk_i, 1]
                xj_broadcast = xj_all.unsqueeze(0)  # [1, num_j]
                yj_broadcast = yj_all.unsqueeze(0)  # [1, num_j]
                
                # Broadcast dimensions (needed for spatial filtering and overlap computation)
                wi_broadcast = wi_chunk.unsqueeze(1)  # [chunk_i, 1]
                hi_broadcast = hi_chunk.unsqueeze(1)  # [chunk_i, 1]
                wj_broadcast = wj_all.unsqueeze(0)  # [1, num_j]
                hj_broadcast = hj_all.unsqueeze(0)  # [1, num_j]
                areai_broadcast = areai_chunk.unsqueeze(1)  # [chunk_i, 1]
                areaj_broadcast = areaj_all.unsqueeze(0)  # [1, num_j]
                
                # Create upper triangle mask: ensure j > i for each i in the chunk
                i_broadcast = i_indices.unsqueeze(1)  # [chunk_i, 1]
                j_broadcast = j_indices.unsqueeze(0)  # [1, num_j]
                upper_triangle_mask = j_broadcast > i_broadcast  # [chunk_i, num_j]
                
                # Spatial filtering: compute approximate distance (only for valid pairs)
                dist_x = torch.abs(xi_broadcast - xj_broadcast)  # [chunk_i, num_j]
                dist_y = torch.abs(yi_broadcast - yj_broadcast)  # [chunk_i, num_j]
                max_dist = torch.maximum(dist_x, dist_y)  # [chunk_i, num_j]
                
                # Early exit: if all pairs are too far, skip this chunk entirely
                if max_dist.min() > spatial_filter_dist:
                    continue
                
                spatial_mask = max_dist < spatial_filter_dist  # [chunk_i, num_j]
                
                # Combine both masks: upper triangle AND spatial filter
                valid_mask = upper_triangle_mask & spatial_mask  # [chunk_i, num_j]
                
                if not valid_mask.any():
                    continue
                
                # Compute pairwise distances
                dx_abs = torch.abs(xi_broadcast - xj_broadcast)  # [chunk_i, num_j]
                dy_abs = torch.abs(yi_broadcast - yj_broadcast)  # [chunk_i, num_j]
                
                # Calculate minimum separation and margin
                min_sep_x = 0.5 * (wi_broadcast + wj_broadcast)  # [chunk_i, num_j]
                min_sep_y = 0.5 * (hi_broadcast + hj_broadcast)  # [chunk_i, num_j]
                
                min_dim_i = torch.minimum(wi_broadcast, hi_broadcast)  # [chunk_i, 1]
                min_dim_j = torch.minimum(wj_broadcast, hj_broadcast)  # [1, num_j]
                min_dim_pair = torch.minimum(min_dim_i, min_dim_j)  # [chunk_i, num_j]
                margin = margin_factor * min_dim_pair  # [chunk_i, num_j]
                
                required_sep_x = min_sep_x + margin  # [chunk_i, num_j]
                required_sep_y = min_sep_y + margin  # [chunk_i, num_j]
                
                # Compute overlap amounts
                overlap_x_raw = required_sep_x - dx_abs  # [chunk_i, num_j]
                overlap_y_raw = required_sep_y - dy_abs  # [chunk_i, num_j]
                
                # Use custom StrongFastSigmoid for ALL overlaps
                overlap_x = strong_fast_sigmoid(overlap_x_raw, scale=scale, alpha=alpha)
                overlap_y = strong_fast_sigmoid(overlap_y_raw, scale=scale, alpha=alpha)
                
                overlap_x = torch.clamp(overlap_x, min=0.0)
                overlap_y = torch.clamp(overlap_y, min=0.0)
                
                # Power-law penalty
                overlap_area = overlap_x * overlap_y  # [chunk_i, num_j]
                overlap_area_safe = torch.clamp(overlap_area, min=1e-8)
                overlap_penalty = torch.pow(overlap_area_safe, 2.5)
                
                # Area weighting
                repulsion_strength = torch.sqrt(areai_broadcast * areaj_broadcast)  # [chunk_i, num_j]
                weighted_penalty = overlap_penalty * repulsion_strength  # [chunk_i, num_j]
                
                # Apply mask
                weighted_penalty = weighted_penalty * valid_mask.float()
                chunk_penalty = torch.sum(weighted_penalty)
                # Check for NaN/Inf before accumulating
                if torch.isfinite(chunk_penalty):
                    total_penalty = total_penalty + chunk_penalty
        
        # Final check: ensure result is finite
        if not torch.isfinite(total_penalty):
            total_penalty = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        
        if PROFILE_OVERLAP and overlap_prof_start is not None:
            torch.cuda.synchronize() if device.type == "cuda" else None
            LAST_OVERLAP_PROFILE = {
                "backend": "cpu",
                "host": time.perf_counter() - overlap_prof_start,
                "kernel": 0.0,
            }
        else:
            LAST_OVERLAP_PROFILE = None
        return total_penalty
    
    else:
        # For small problems: use same adaptive FastSigmoid approach
        # Broadcast for pairwise computation
        xi, yi = x.unsqueeze(1), y.unsqueeze(1)  # [N, 1]
        xj, yj = x.unsqueeze(0), y.unsqueeze(0)  # [1, N]
        wi, hi, areai = w.unsqueeze(1), h.unsqueeze(1), area.unsqueeze(1)  # [N, 1]
        wj, hj, areaj = w.unsqueeze(0), h.unsqueeze(0), area.unsqueeze(0)  # [1, N]
        
        # Compute pairwise distances
        dx_abs = torch.abs(xi - xj)  # [N, N]
        dy_abs = torch.abs(yi - yj)  # [N, N]
        
        # Calculate minimum separation and margin
        min_sep_x = 0.5 * (wi + wj)  # [N, N]
        min_sep_y = 0.5 * (hi + hj)  # [N, N]
        
        min_dim_i = torch.minimum(wi, hi)  # [N, 1]
        min_dim_j = torch.minimum(wj, hj)  # [1, N]
        min_dim_pair = torch.minimum(min_dim_i, min_dim_j)  # [N, N]
        margin = margin_factor * min_dim_pair  # [N, N]
        
        required_sep_x = min_sep_x + margin  # [N, N]
        required_sep_y = min_sep_y + margin  # [N, N]
        
        # Compute overlap amounts
        overlap_x_raw = required_sep_x - dx_abs  # [N, N]
        overlap_y_raw = required_sep_y - dy_abs  # [N, N]
        
        # Use custom StrongFastSigmoid for ALL overlaps - aggressive magnitude-aware gradients
        overlap_x = strong_fast_sigmoid(overlap_x_raw, scale=scale, alpha=alpha)
        overlap_y = strong_fast_sigmoid(overlap_y_raw, scale=scale, alpha=alpha)
        
        # Ensure non-negative (Softplus already ensures this, but clamp for safety)
        overlap_x = torch.clamp(overlap_x, min=0.0)
        overlap_y = torch.clamp(overlap_y, min=0.0)
        
        # Aggressive power-law penalty: overlap^2.5 for stronger gradients on remaining overlaps
        overlap_area = overlap_x * overlap_y  # [N, N]
        overlap_area_safe = torch.clamp(overlap_area, min=1e-8)  # Avoid numerical issues
        overlap_penalty = torch.pow(overlap_area_safe, 2.5)  # More aggressive than squared
        
        # Area weighting
        repulsion_strength = torch.sqrt(areai * areaj)  # [N, N]
        weighted_penalty = overlap_penalty * repulsion_strength  # [N, N]
        
        # Upper triangle mask
        valid_mask = torch.triu(
            torch.ones(N, N, device=device, dtype=torch.bool), 
            diagonal=1
        )
        weighted_penalty = weighted_penalty * valid_mask.float()
        total_penalty = torch.sum(weighted_penalty)
        
        # Final check: ensure result is finite
        if not torch.isfinite(total_penalty):
            total_penalty = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        
        if PROFILE_OVERLAP and overlap_prof_start is not None:
            torch.cuda.synchronize() if device.type == "cuda" else None
            LAST_OVERLAP_PROFILE = {
                "backend": "cpu",
                "host": time.perf_counter() - overlap_prof_start,
                "kernel": 0.0,
            }
        else:
            LAST_OVERLAP_PROFILE = None
        return total_penalty


# Enable CUDA optimizations globally
# Use torch.compile if available (PyTorch 2.0+), otherwise use JIT
_USE_TORCH_COMPILE = hasattr(torch, 'compile') and torch.cuda.is_available()

def train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=1000,
    lr=0.1,
    lambda_wirelength=1.0,
    lambda_overlap=100.0,
    verbose=True,
    log_interval=50,
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
    
    # Scale learning rate: larger problems need higher LR for faster convergence
    # Use logarithmic scaling: lr_scale = 1.0 + 0.15 * log10(N/50)
    # For N=50: scale=1.0, N=500: scale=1.15, N=2000: scale=1.24, N=10000: scale=1.40
    lr_scale = 1.0 + 0.15 * math.log10(max(N / 50.0, 1.0))
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
    # Use fused Adam optimizer if available (faster CUDA implementation)
    try:
        # Fused Adam is faster but requires CUDA
        if torch.cuda.is_available() and device.type == 'cuda':
            optimizer = optim.Adam([cell_positions], lr=scaled_lr, fused=True)
        else:
            optimizer = optim.Adam([cell_positions], lr=scaled_lr)
    except TypeError:
        # Fused not available in this PyTorch version
        optimizer = optim.Adam([cell_positions], lr=scaled_lr)
    
    # Create GradScaler for mixed precision training (faster, lower memory)
    # Use new torch.amp API to avoid deprecation warnings
    if torch.cuda.is_available() and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda', enabled=True)
    else:
        scaler = torch.amp.GradScaler('cpu', enabled=False)
    
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

    # Adaptive learning rate: track loss to increase LR if it plateaus
    # But be more conservative - only increase if overlap loss plateaus
    best_overlap_loss = float('inf')
    plateau_count = 0
    plateau_threshold = 100  # If overlap loss doesn't improve for 100 epochs, increase LR
    lr_increase_factor = 1.2  # More conservative: 1.2x instead of 1.5x
    max_lr_multiplier = 2.0  # Lower max: 2.0x instead of 3.0x
    current_lr_multiplier = 1.0

    # Training loop with adaptive learning rate and overlap weighting
    for epoch in range(num_epochs):
        epoch_progress = epoch / num_epochs
        if PROFILE_PLACEMENT:
            torch.cuda.synchronize() if device.type == 'cuda' else None
            epoch_start = time.perf_counter()
        
        # Base cosine annealing: smooth decay from scaled_lr to scaled_lr/3
        lr_min = scaled_lr * (0.3 if N > 2000 else 0.5)
        lr_max = scaled_lr
        base_lr = lr_min + (lr_max - lr_min) * (1 + math.cos(epoch_progress * math.pi)) / 2
        
        # Apply adaptive multiplier (increases if loss plateaus)
        current_lr = base_lr * current_lr_multiplier
        
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

        # Calculate losses with CUDA optimizations
        # Use torch.amp for automatic mixed precision (faster, lower memory)
        # Only use mixed precision for large problems to avoid overhead
        use_amp = torch.cuda.is_available() and device.type == 'cuda' and N > 1000
        if PROFILE_PLACEMENT:
            torch.cuda.synchronize() if device.type == 'cuda' else None
            wl_timer_start = time.perf_counter()
        with torch.amp.autocast('cuda', enabled=use_amp):
            wl_loss = wirelength_attraction_loss(
                cell_features_current, pin_features, edge_list
            )
        if PROFILE_PLACEMENT:
            torch.cuda.synchronize() if device.type == 'cuda' else None
            wl_time = time.perf_counter() - wl_timer_start
            torch.cuda.synchronize() if device.type == 'cuda' else None
            overlap_timer_start = time.perf_counter()
        overlap_loss = overlap_repulsion_loss(
            cell_features_current, pin_features, edge_list, epoch_progress
        )
        if PROFILE_PLACEMENT:
            torch.cuda.synchronize() if device.type == 'cuda' else None
            ol_time = time.perf_counter() - overlap_timer_start
            overlap_host = LAST_OVERLAP_PROFILE.get("host", 0.0) if LAST_OVERLAP_PROFILE else 0.0
            overlap_kernel = LAST_OVERLAP_PROFILE.get("kernel", 0.0) if LAST_OVERLAP_PROFILE else 0.0

        # Combined loss with adaptive lambda
        total_loss = lambda_wirelength * wl_loss + current_lambda_overlap * overlap_loss
        
        # Check for NaN/Inf before backward pass
        should_skip_update = False
        backward_time = 0.0
        backward_start = None
        if PROFILE_PLACEMENT:
            torch.cuda.synchronize() if device.type == 'cuda' else None
            backward_start = time.perf_counter()
        if not torch.isfinite(total_loss):
            if verbose and epoch % log_interval == 0:
                print(f"WARNING: NaN/Inf detected in total_loss at epoch {epoch}")
                wl_val = wl_loss.item() if torch.isfinite(wl_loss) else float('nan')
                ol_val = overlap_loss.item() if torch.isfinite(overlap_loss) else float('nan')
                print(f"  wl_loss: {wl_val}, overlap_loss: {ol_val}")
            should_skip_update = True
        else:
            # Backward pass with mixed precision scaling
            scaler.scale(total_loss).backward()
            
            # Check for NaN/Inf in gradients before clipping
            has_nan_grad = False
            if cell_positions.grad is not None:
                if not torch.isfinite(cell_positions.grad).all():
                    has_nan_grad = True
                    if verbose and epoch % log_interval == 0:
                        print(f"WARNING: NaN/Inf gradients detected at epoch {epoch}, skipping update")
            
            if not has_nan_grad:
                # Gradient clipping with scaler (faster fused operation)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=5.0)
                # Update positions with scaler (handles mixed precision)
                scaler.step(optimizer)
                scaler.update()
            else:
                should_skip_update = True
                # Zero out gradients to prevent NaN propagation
                scaler.update()  # Update scaler even on skip
                if cell_positions.grad is not None:
                    cell_positions.grad.zero_()
        
        if should_skip_update:
            # Still record losses (as NaN) for debugging, but don't update positions
            # Zero gradients to prevent accumulation
            optimizer.zero_grad()
        if PROFILE_PLACEMENT and backward_start is not None:
            torch.cuda.synchronize() if device.type == 'cuda' else None
            backward_time = time.perf_counter() - backward_start

        # Record losses (may be NaN, but that's useful for debugging)
        total_loss_val = total_loss.item() if torch.isfinite(total_loss) else float('nan')
        wl_loss_val = wl_loss.item() if torch.isfinite(wl_loss) else float('nan')
        ol_loss_val = overlap_loss.item() if torch.isfinite(overlap_loss) else float('nan')
        
        loss_history["total_loss"].append(total_loss_val)
        loss_history["wirelength_loss"].append(wl_loss_val)
        loss_history["overlap_loss"].append(ol_loss_val)
        
        # Adaptive learning rate: increase if OVERLAP loss plateaus (not total loss)
        # This is more targeted - we care about overlap reduction, not total loss
        # IMPORTANT: Do NOT increase LR if overlap loss is already 0 (success case)
        if torch.isfinite(overlap_loss):
            ol_loss_val = ol_loss_val if torch.isfinite(overlap_loss) else float('inf')
            
            # Only track plateau if loss is not zero (if loss is 0, we've succeeded!)
            if ol_loss_val > 1e-8:  # Only consider non-zero losses for plateau detection
                if ol_loss_val < best_overlap_loss:
                    best_overlap_loss = ol_loss_val
                    plateau_count = 0
                else:
                    plateau_count += 1
                    
                # If overlap loss hasn't improved for plateau_threshold epochs, increase LR
                # But only if we're past the initial ramp-up phase (epoch_progress > 0.3)
                # And only if loss is still non-zero (don't increase LR when we've succeeded)
                if (plateau_count >= plateau_threshold and 
                    current_lr_multiplier < max_lr_multiplier and 
                    epoch_progress > 0.3 and
                    ol_loss_val > 1e-8):
                    current_lr_multiplier = min(current_lr_multiplier * lr_increase_factor, max_lr_multiplier)
                    plateau_count = 0  # Reset counter
                    if verbose:
                        print(f" LR increased to {current_lr:.4f} (multiplier: {current_lr_multiplier:.2f}) due to overlap loss plateau")
            else:
                # Loss is effectively zero - we've succeeded, reset tracking
                best_overlap_loss = ol_loss_val
                plateau_count = 0

        # Log progress / profiling
        if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch}/{num_epochs} (progress: {epoch_progress:.2f}):")
            print(f"  LR: {current_lr:.4f} (×{current_lr_multiplier:.2f}), λ_overlap: {current_lambda_overlap:.2f}")
            print(f"  Total Loss: {total_loss_val:.6f}")
            print(f"  Wirelength Loss: {wl_loss_val:.6f}")
            print(f"  Overlap Loss: {ol_loss_val:.6f}")
        if PROFILE_PLACEMENT:
            torch.cuda.synchronize() if device.type == 'cuda' else None
            epoch_time = time.perf_counter() - epoch_start if 'epoch_start' in locals() else 0.0
            print(
                f"[PROFILE] epoch {epoch}: "
                f"wl={wl_time:.3f}s overlap={ol_time:.3f}s "
                f"(host {overlap_host:.3f}s kernel {overlap_kernel:.3f}s) "
                f"backward={backward_time:.3f}s total={epoch_time:.3f}s"
            )

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

# Compile overlap loss function for maximum performance using torch.compile
# This automatically uses NVIDIA's cuTENSOR library and kernel fusion
# Compile after function definition to avoid issues
if _USE_TORCH_COMPILE:
    try:
        # Compile with reduce-overhead mode: balance between compilation time and runtime speed
        # This uses cuTENSOR and automatic kernel fusion under the hood
        overlap_repulsion_loss = torch.compile(
            overlap_repulsion_loss, 
            mode='reduce-overhead',  # Good balance: faster than default, less compilation time than max-autotune
            fullgraph=False  # Allow partial compilation for flexibility
        )
    except Exception as e:
        # If compilation fails, continue without it
        pass

if __name__ == "__main__":
    main()
