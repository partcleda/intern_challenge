"""
Loss functions for VLSI placement optimization.
"""

import torch
import os
from .surrogate_gradients import strong_fast_sigmoid
from .utils import CellFeatureIdx, DEBUG_CUDA_OVERLAP, FORCE_CPU_OVERLAP

# DISABLED: torch.compile() causes performance degradation with dynamic input shapes
# The function is called with many different chunk sizes (51+ distinct shapes),
# causing CUDA graph recording overhead that kills performance.
# The other optimizations (simplified surrogate gradient, fused operations) provide
# sufficient speedup without the compilation overhead.
USE_TORCH_COMPILE = False
TORCH_COMPILE_AVAILABLE = False

# Enable performance optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Faster for consistent input sizes
    torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32 for Ampere+ GPUs
    torch.backends.cudnn.allow_tf32 = True

# Try to import CUDA-accelerated overlap loss
try:
    from cuda_backend import (
        compute_overlap_loss as cuda_overlap_loss,
        is_available as cuda_overlap_available,
        get_last_stats as cuda_overlap_stats,
    )
    CUDA_OVERLAP_AVAILABLE = cuda_overlap_available()
    if CUDA_OVERLAP_AVAILABLE:
        print(f"[CUDA] CUDA backend is available and will be used for N>=50000 cells")
except ImportError as e:
    cuda_overlap_loss = None
    cuda_overlap_stats = lambda: {}
    CUDA_OVERLAP_AVAILABLE = False
    print(f"[CUDA] CUDA backend not available: {e}")
except Exception as e:
    cuda_overlap_loss = None
    cuda_overlap_stats = lambda: {}
    CUDA_OVERLAP_AVAILABLE = False
    print(f"[CUDA] CUDA backend error: {e}")

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


def _compute_overlap_chunk_fused(
    xi, yi, wi, hi, areai,
    xj, yj, wj, hj, areaj,
    margin_factor, scale, alpha, valid_mask
):
    """Fused overlap computation for a chunk - optimized for speed.
    
    All operations are fused to minimize kernel launches and improve GPU utilization.
    NOTE: torch.compile() is disabled due to dynamic shape overhead.
    
    Args:
        xi, yi, wi, hi, areai: Cell i features [chunk_i, 1] or [chunk_i, chunk_j]
        xj, yj, wj, hj, areaj: Cell j features [1, chunk_j] or [chunk_i, chunk_j]
        margin_factor: Margin scaling factor
        scale, alpha: Surrogate gradient parameters
        valid_mask: Boolean mask for valid pairs [chunk_i, chunk_j]
    """
    # Fused distance computation
    dx_abs = torch.abs(xi - xj)
    dy_abs = torch.abs(yi - yj)
    
    # Fused separation and margin
    min_sep_x = 0.5 * (wi + wj)
    min_sep_y = 0.5 * (hi + hj)
    min_dim_pair = torch.minimum(torch.minimum(wi, hi), torch.minimum(wj, hj))
    margin = margin_factor * min_dim_pair
    
    # Fused overlap calculation
    # Only apply margin when there's actual overlap (dx_abs < min_sep_x)
    # When cells are properly separated (dx_abs >= min_sep_x), margin should not create penalty
    overlap_x_base = min_sep_x - dx_abs
    overlap_y_base = min_sep_y - dy_abs
    # Apply margin when overlap_base > 0 (only when there's actual overlap)
    # Tightened threshold: only apply margin when cells are actually overlapping
    # This prevents margin from creating barriers for properly separated cells
    margin_x = margin * (overlap_x_base > 0.0).float()
    margin_y = margin * (overlap_y_base > 0.0).float()
    overlap_x_raw = overlap_x_base + margin_x
    overlap_y_raw = overlap_y_base + margin_y
    
    # Use optimized StrongFastSigmoid
    overlap_x = strong_fast_sigmoid(overlap_x_raw, scale=scale, alpha=alpha)
    overlap_y = strong_fast_sigmoid(overlap_y_raw, scale=scale, alpha=alpha)
    
    # Fused penalty computation with in-place operations where possible
    overlap_area = torch.clamp(overlap_x, min=0.0) * torch.clamp(overlap_y, min=0.0)
    # Clamp to very small value for numerical stability in power operation
    overlap_area = torch.clamp(overlap_area, min=1e-15)  # Numerical stability threshold
    overlap_penalty = torch.pow(overlap_area, 2.5)
    repulsion_strength = torch.sqrt(areai * areaj)
    weighted_penalty = overlap_penalty * repulsion_strength * valid_mask.float()
    
    return torch.sum(weighted_penalty)


# NOTE: torch.compile() is disabled because it causes performance degradation
# with dynamic input shapes. The function is called with many different chunk sizes,
# causing CUDA graph recording overhead (51+ distinct shapes) that kills performance.
# The simplified surrogate gradient and fused operations provide sufficient speedup.


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
    
    # Compute overlap loss every epoch for proper gradient flow
    # Gradients are needed for optimization even when overlap is small
    
    # Extract positions, dimensions, and areas
    x = cell_features[:, CellFeatureIdx.X]  # [N]
    y = cell_features[:, CellFeatureIdx.Y]  # [N]
    w = cell_features[:, CellFeatureIdx.WIDTH]  # [N]
    h = cell_features[:, CellFeatureIdx.HEIGHT]  # [N]
    area = cell_features[:, CellFeatureIdx.AREA]  # [N]

    # Precompute static values (dimensions don't change, only positions do)
    # Cache these to avoid recomputing in each epoch
    cache_key = (N, w.sum().item(), h.sum().item())  # Simple cache key based on dimensions
    if not hasattr(overlap_repulsion_loss_original, '_dim_cache') or \
       overlap_repulsion_loss_original._dim_cache.get('key') != cache_key:
        # Precompute min dimensions for margin calculation (used frequently)
        min_dim_i = torch.minimum(w, h)  # [N]
        overlap_repulsion_loss_original._dim_cache = {
            'key': cache_key,
            'min_dim_i': min_dim_i,
            'w': w, 'h': h, 'area': area  # Cache these too
        }
    else:
        # Reuse cached dimensions (positions changed but dimensions same)
        cache = overlap_repulsion_loss_original._dim_cache
        min_dim_i = cache['min_dim_i']
        # w, h, area are same, but we need fresh references for gradient tracking
        # Actually, we should use the original w, h, area for gradients

    # Adaptive margin: starts high to force separation, then gradually reduces for fine-tuning
    # Margin reduction must be gradual to prevent instability
    # Strategy: Start high to push cells apart, then gradually reduce (not to zero) for stability
    if N > 50000:
        # Very large problems: start at 5%, peak at 7% around 50%, then gradually reduce to 0.5% at end
        if epoch_progress < 0.5:
            # First 50%: increase margin to force separation
            margin_factor = 0.05 + 0.02 * (epoch_progress / 0.5)  # 0.05 -> 0.07
        elif epoch_progress < 0.88:
            # 50-88% epochs: gradually reduce margin from 7% to 1.5%
            mid_progress = (epoch_progress - 0.5) / 0.38  # 0 to 1 over 50-88%
            margin_factor = 0.07 - 0.055 * mid_progress  # 0.07 -> 0.015
        else:
            # Last 12% epochs: reduce margin to near-zero for fine-tuning
            final_progress = (epoch_progress - 0.88) / 0.12  # 0 to 1 over last 12%
            margin_factor = 0.015 - 0.01499999 * final_progress  # 0.015 -> 0.00000001
    else:
        # Smaller problems: start at 1% and increase to 3%
        margin_factor = 0.01 + 0.02 * epoch_progress  # 0.01 -> 0.03
    
    # Use custom StrongFastSigmoid for all overlaps
    # This provides magnitude-aware gradients that increase with overlap size
    # Key features:
    # - Forward: scaled softplus activation (smooth, magnitude-preserving)
    # - Backward: sigmoid-based gradient that scales with magnitude
    # - Alpha parameter controls gradient sharpness (higher = stronger gradients)
    # - Adaptive alpha scaling provides stronger gradients than standard Softplus
    
    # Scale parameter: controls loss magnitude, increases with training progress
    if N > 50000:
        # Very large problems: increase scale gradually, maintain high value at end for stability
        base_scale = 36.5 + 26.5 * epoch_progress  # 36.5 -> 63.0: exact 357-cell state
        scale = base_scale
    else:
        base_scale = 3.0 + 12.0 * epoch_progress  # 3.0 -> 15.0 (stronger as training progresses)
        if N > 5000:
            # For large problems, need even stronger gradients
            scale = base_scale * 1.3  # 30% stronger for large problems
        else:
            scale = base_scale
    
    # Alpha parameter: controls gradient sharpness, increases with training progress
    if N > 50000:
        # Very large problems: increase alpha gradually, maintain high value at end for stability
        alpha = 21.5 + 17.5 * epoch_progress  # 21.5 -> 39.0: exact 357-cell state
    else:
        alpha = 3.0 + 12.0 * epoch_progress  # 3.0 -> 15.0
    
    # Implementation uses optimized PyTorch operations with:
    # - Spatial hashing for O(N) complexity on large problems
    # - Chunked processing to avoid memory issues
    # - Vectorized PyTorch operations for GPU acceleration (when tensors are on GPU)
    # - CUDA backend available for large problems (50k+ cells) for additional speedup
    
    N = cell_features.shape[0]
    if not hasattr(overlap_repulsion_loss_original, '_cuda_backend_disabled'):
        overlap_repulsion_loss_original._cuda_backend_disabled = False
    
    # Use CUDA backend when available for large problems (N>=50000, test cases 11-12)
    USE_CUDA_BACKEND = (
        not overlap_repulsion_loss_original._cuda_backend_disabled and
        CUDA_OVERLAP_AVAILABLE and 
        device.type == "cuda" and 
        N >= 50000  # CUDA backend only for large problems (overhead not worth it for smaller)
    )
    
    if USE_CUDA_BACKEND and not FORCE_CPU_OVERLAP:
        try:
            # Display message once when CUDA backend is used (only on first call or with debug flag)
            if not hasattr(overlap_repulsion_loss_original, '_cuda_backend_announced'):
                print(f"[OVERLAP] ✓ Using CUDA-accelerated backend (N={N} cells)")
                overlap_repulsion_loss_original._cuda_backend_announced = True
            
            # CUDA backend path for optimized performance
            
            loss = cuda_overlap_loss(
                cell_features,
                margin_factor,
                alpha,
                scale,
                epoch_progress,
            )
            
            # PyTorch autograd handles synchronization automatically when needed
            
            if DEBUG_CUDA_OVERLAP:
                stats = cuda_overlap_stats()
                print(
                    f"[CUDA-OVERLAP] backend used "
                    f"(pairs={stats.get('pairs')}, bin_size={stats.get('bin_size', 0.0):.3f})"
                )
            return loss
        except RuntimeError as exc:
            # Fall back to PyTorch implementation on failure
            print(f"[OVERLAP] ✗ CUDA backend failed, falling back to PyTorch: {exc}")
            # Disable CUDA backend for this session to prevent repeated hangs
            overlap_repulsion_loss_original._cuda_backend_disabled = True
            if DEBUG_CUDA_OVERLAP:
                print(f"[CUDA-OVERLAP] Detailed error: {type(exc).__name__}: {exc}")
            pass
        except Exception as exc:
            # Catch any other exceptions (including CUDA errors)
            print(f"[OVERLAP] CUDA backend error, falling back to CPU: {type(exc).__name__}: {exc}")
            if DEBUG_CUDA_OVERLAP:
                import traceback
                traceback.print_exc()
            pass
    
    # PyTorch implementation: Uses GPU automatically when tensors are on GPU
    # IMPORTANT: This "CPU" code actually runs on GPU when tensors are on GPU device!
    # PyTorch operations (torch.abs, torch.sum, etc.) are GPU-accelerated automatically.
    # No custom CUDA kernel needed - PyTorch handles GPU acceleration transparently.
    
    # Display backend info once per run
    if not hasattr(overlap_repulsion_loss_original, '_backend_announced'):
        if device.type == "cuda":
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                print(f"[OVERLAP] Using PyTorch GPU-accelerated operations (N={N} cells, device={device}, {gpu_memory_gb:.1f}GB GPU)")
            except:
                print(f"[OVERLAP] Using PyTorch GPU-accelerated operations (N={N} cells, device={device})")
        else:
            print(f"[OVERLAP] Using PyTorch CPU operations (N={N} cells)")
        overlap_repulsion_loss_original._backend_announced = True
    
    # Optimized approach: Maximize GPU memory utilization with adaptive chunk sizes
    # Key strategy: Use larger chunks to better utilize GPU memory and reduce overhead
    # For very large problems (100K+), use spatial hashing to reduce O(N²) to O(N)
    
    # Get GPU memory info if available (for adaptive chunk sizing)
    if device.type == "cuda":
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        except:
            gpu_memory_gb = 8.0  # Default assumption: 8GB GPU
    else:
        gpu_memory_gb = 0.0
    
    # Adaptive chunk sizing based on problem size and GPU memory
    # Goal: Use 50-80% of available GPU memory per chunk for maximum utilization
    # Memory per chunk ≈ CHUNK_SIZE_I * CHUNK_SIZE_J * 4 bytes (float32) * ~10 tensors
    # For 47GB GPU: ~20GB usable = 5B floats = sqrt(5B/10) ≈ 22k cells per dimension
    # We'll be more conservative and use ~10-15k per dimension to leave room for other ops
    
    if N >= 100000:
        # For 100K+ cells: Use spatial hashing with optimized chunk sizes
        # OPTIMIZED: Larger bins = fewer bins = faster processing (still accurate enough)
        USE_SPATIAL_HASHING = True
        BIN_SIZE = 45.0  # Bin size optimized for accuracy and performance
        MAX_NEIGHBORS_PER_BIN = 2000  # Higher limit to catch all overlaps
        
        # Use maximum chunk sizes to minimize Python overhead
        # Large chunks reduce Python loop overhead and improve GPU utilization
        if gpu_memory_gb >= 32:
            # For 100k+ cells: Use very large chunks (50k) to minimize Python loop iterations
            # Memory: 50k * 50k * 4 bytes * ~10 tensors ≈ 100GB theoretical max
            # But we only process upper triangle + spatial filtering, so actual usage is much lower
            CHUNK_SIZE_I = 50000  # Aggressive: maximize chunk size to minimize Python overhead
            CHUNK_SIZE_J = 50000  # Fewer iterations = much less Python overhead
        elif gpu_memory_gb >= 16:
            CHUNK_SIZE_I = 6000  # Large chunks for high-memory GPUs
            CHUNK_SIZE_J = 6000
        elif gpu_memory_gb >= 8:
            CHUNK_SIZE_I = 3000  # Medium chunks for mid-range GPUs
            CHUNK_SIZE_J = 3000
        else:
            CHUNK_SIZE_I = 2000  # Smaller chunks for low-memory GPUs
            CHUNK_SIZE_J = 2000
    elif N >= 50000:
        # For 50K-100K cells: Use spatial hashing with large chunks
        USE_SPATIAL_HASHING = True
        BIN_SIZE = 60.0  # Smaller bins for better accuracy
        MAX_NEIGHBORS_PER_BIN = 800  # Higher limit for better coverage
        
        if gpu_memory_gb >= 32:
            CHUNK_SIZE_I = 30000  # Aggressive: maximize chunk size for 50k-100k cells
            CHUNK_SIZE_J = 30000  # Much larger chunks = fewer Python loop iterations
        elif gpu_memory_gb >= 16:
            CHUNK_SIZE_I = 5000
            CHUNK_SIZE_J = 5000
        elif gpu_memory_gb >= 8:
            CHUNK_SIZE_I = 3000
            CHUNK_SIZE_J = 3000
        else:
            CHUNK_SIZE_I = 2000
            CHUNK_SIZE_J = 2000
    elif N > 20000:
        # For 20K-50K cells: No spatial hashing, use large chunks
        USE_SPATIAL_HASHING = False
        if gpu_memory_gb >= 32:
            CHUNK_SIZE_I = 8000  # Very large chunks for high-memory GPUs
            CHUNK_SIZE_J = 15000
        elif gpu_memory_gb >= 16:
            CHUNK_SIZE_I = 6000  # Large chunks for high-memory GPUs
            CHUNK_SIZE_J = 12000
        elif gpu_memory_gb >= 8:
            CHUNK_SIZE_I = 4000  # Large chunks for mid-range GPUs
            CHUNK_SIZE_J = 8000
        else:
            CHUNK_SIZE_I = 2000  # Medium chunks for low-memory GPUs
            CHUNK_SIZE_J = 4000
        USE_DOUBLE_CHUNKING = False
        CLEAR_CACHE_BETWEEN_CHUNKS = False
    elif N > 5000:
        # For 5K-20K cells: Use very large chunks
        USE_SPATIAL_HASHING = False
        if gpu_memory_gb >= 32:
            CHUNK_SIZE_I = 10000  # Maximum chunk size for high-memory GPUs
            CHUNK_SIZE_J = 20000
        elif gpu_memory_gb >= 16:
            CHUNK_SIZE_I = 8000
            CHUNK_SIZE_J = 16000
        elif gpu_memory_gb >= 8:
            CHUNK_SIZE_I = 5000
            CHUNK_SIZE_J = 10000
        else:
            CHUNK_SIZE_I = 3000
            CHUNK_SIZE_J = 6000
        USE_DOUBLE_CHUNKING = False
        CLEAR_CACHE_BETWEEN_CHUNKS = False
    else:
        # For <5K cells: Use maximum chunk sizes (process everything at once if possible)
        USE_SPATIAL_HASHING = False
        CHUNK_SIZE_I = min(10000, N)  # Process all cells at once for small problems
        CHUNK_SIZE_J = min(20000, N)
        USE_DOUBLE_CHUNKING = False
        CLEAR_CACHE_BETWEEN_CHUNKS = False
    
    USE_CHUNKING = N > 2000  # Only chunk for problems > 2000 cells (reduce overhead)
    
    # For very large problems, use spatial hashing: O(N) instead of O(N²)
    if USE_SPATIAL_HASHING:
        if not hasattr(overlap_repulsion_loss_original, '_chunk_info_printed'):
            print(f"[OVERLAP] Spatial hashing: N={N}, bin_size={BIN_SIZE:.1f}, chunks=({CHUNK_SIZE_I}, {CHUNK_SIZE_J}), GPU={gpu_memory_gb:.1f}GB")
            overlap_repulsion_loss_original._chunk_info_printed = True
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
        
        # OPTIMIZED: Precompute neighbor bin mappings for all bins at once (vectorized)
        # This eliminates Python loops in neighbor bin collection
        bin_to_pos = {int(bid.item()): int(pos) for pos, bid in enumerate(unique_bins)}
        
        # Precompute all neighbor bins for all unique bins (vectorized)
        bin_x_coords = (unique_bins % num_bins_x).long()
        bin_y_coords = (unique_bins // num_bins_x).long()
        
        # Collect all bin pairs first, then process in large batches
        # This reduces Python loop overhead by processing multiple bins together
        all_bin_pairs = []
        
        # First pass: collect all bin pairs
        for bin_idx, bin_id in enumerate(unique_bins):
            bin_start = bin_starts[bin_idx].item()
            bin_count = bin_counts[bin_idx].item()
            
            if bin_count == 0:
                continue
            
            # Get cells in this bin
            bin_cell_indices = sorted_indices[bin_start:bin_start + bin_count]
            # Process all cells in chunks to avoid missing overlaps
            # Only limit if absolutely necessary to prevent OOM
            if len(bin_cell_indices) > MAX_NEIGHBORS_PER_BIN * 3:
                # Only sample if bin is extremely large
                perm = torch.randperm(len(bin_cell_indices), device=device)[:MAX_NEIGHBORS_PER_BIN]
                bin_cell_indices = bin_cell_indices[perm]
            
            # Vectorized neighbor bin computation
            bin_x_coord = bin_x_coords[bin_idx].item()
            bin_y_coord = bin_y_coords[bin_idx].item()
            
            # Pre-compute all 9 neighbor offsets (same bin + 8 adjacent)
            dx_vals = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], device=device, dtype=torch.long)
            dy_vals = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], device=device, dtype=torch.long)
            nx_coords = bin_x_coord + dx_vals
            ny_coords = bin_y_coord + dy_vals
            valid_mask = (nx_coords >= 0) & (nx_coords < num_bins_x) & (ny_coords >= 0) & (ny_coords < num_bins_y)
            neighbor_bins = (ny_coords[valid_mask] * num_bins_x + nx_coords[valid_mask]).long()
            
            # Vectorized neighbor cell collection
            neighbor_indices_list = []
            for nbin_id_tensor in neighbor_bins:
                nbin_id_int = nbin_id_tensor.item()
                if nbin_id_int in bin_to_pos:
                    nbin_pos = bin_to_pos[nbin_id_int]
                    nbin_start = bin_starts[nbin_pos].item()
                    nbin_count = bin_counts[nbin_pos].item()
                    nbin_cells = sorted_indices[nbin_start:nbin_start + nbin_count]
                    # Process all neighbor cells to avoid missing overlaps
                    # Only limit if absolutely necessary to prevent OOM
                    if len(nbin_cells) > MAX_NEIGHBORS_PER_BIN * 3:
                        perm = torch.randperm(len(nbin_cells), device=device)[:MAX_NEIGHBORS_PER_BIN]
                        nbin_cells = nbin_cells[perm]
                    neighbor_indices_list.append(nbin_cells)
            
            if not neighbor_indices_list:
                continue
            
            neighbor_indices = torch.cat(neighbor_indices_list)
            neighbor_indices = torch.unique(neighbor_indices)  # Remove duplicates
            
            # Store pair for batch processing
            all_bin_pairs.append((bin_cell_indices, neighbor_indices))
        
        # Process all bin pairs in large batches to reduce Python overhead
        # Larger batches improve GPU utilization by processing more pairs simultaneously
        BATCH_SIZE = 50  # Process 50 bin pairs at a time
        for batch_start in range(0, len(all_bin_pairs), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(all_bin_pairs))
            batch_pairs = all_bin_pairs[batch_start:batch_end]
            
            # Process each pair in the batch
            for bin_cell_indices, neighbor_indices in batch_pairs:
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
                    
                    # Use optimized fused overlap computation (compiled for speed)
                    chunk_penalty = _compute_overlap_chunk_fused(
                        xi, yi, wi, hi, areai,
                        xj, yj, wj, hj, areaj,
                        margin_factor, scale, alpha, valid_mask
                    )
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
        
        # OPTIMIZED: Cache zero check result for early exit optimization
        with torch.no_grad():
            overlap_val = total_penalty.detach()
            if hasattr(overlap_val, 'item'):
                overlap_repulsion_loss_original._last_overlap_zero = (overlap_val < 1e-10).item()
            else:
                overlap_repulsion_loss_original._last_overlap_zero = False
        
        return total_penalty
    
    elif USE_CHUNKING:
        # Chunked processing: process i cells at a time against all j cells
        # Each chunk is fully vectorized (GPU-friendly), but we avoid full N×N matrix
        if not hasattr(overlap_repulsion_loss_original, '_chunk_info_printed'):
            print(f"[OVERLAP] Chunked processing: N={N}, chunks=({CHUNK_SIZE_I}, {CHUNK_SIZE_J}), GPU={gpu_memory_gb:.1f}GB")
            overlap_repulsion_loss_original._chunk_info_printed = True
        total_penalty = torch.tensor(0.0, device=device, requires_grad=True)
        
        # OPTIMIZED: Precompute max cell size (static, doesn't change)
        if not hasattr(overlap_repulsion_loss_original, '_max_cell_size_cache'):
            max_w = torch.max(w)
            max_h = torch.max(h)
            max_cell_size = torch.max(max_w, max_h)
            overlap_repulsion_loss_original._max_cell_size_cache = max_cell_size.item()
        else:
            max_cell_size = torch.tensor(
                overlap_repulsion_loss_original._max_cell_size_cache,
                device=device,
            )
        
        # Adaptive spatial filter distance based on problem size
        # Spatial filtering balances speed and convergence
        # More aggressive filtering = fewer pairs checked = faster computation
        # Less aggressive filtering = more pairs checked = better convergence
        # Filtering is tuned to check all nearby pairs while skipping distant ones
        if N > 50000:
            # Moderate: check pairs within 4x max cell size
            spatial_filter_dist = 4.0 * max_cell_size + 80.0
        elif N > 20000:
            # Moderate: check pairs within 5x max cell size
            spatial_filter_dist = 5.0 * max_cell_size + 100.0
        else:
            # Conservative: check pairs within 6x max cell size
            spatial_filter_dist = 6.0 * max_cell_size + 150.0
        
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
                    
                # Use optimized fused overlap computation (compiled for speed)
                chunk_penalty = _compute_overlap_chunk_fused(
                    xi_broadcast, yi_broadcast, wi_broadcast, hi_broadcast, areai_broadcast,
                    xj_broadcast, yj_broadcast, wj_broadcast, hj_broadcast, areaj_broadcast,
                    margin_factor, scale, alpha, valid_mask
                )
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
                
                # Use optimized fused overlap computation (compiled for speed)
                chunk_penalty = _compute_overlap_chunk_fused(
                    xi_broadcast, yi_broadcast, wi_broadcast, hi_broadcast, areai_broadcast,
                    xj_broadcast, yj_broadcast, wj_broadcast, hj_broadcast, areaj_broadcast,
                    margin_factor, scale, alpha, valid_mask
                )
                # Check for NaN/Inf before accumulating
                if torch.isfinite(chunk_penalty):
                    total_penalty = total_penalty + chunk_penalty
        
        # Final check: ensure result is finite
        if not torch.isfinite(total_penalty):
            total_penalty = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        
        # OPTIMIZED: Cache zero check result for early exit optimization
        with torch.no_grad():
            overlap_val = total_penalty.detach()
            if hasattr(overlap_val, 'item'):
                overlap_repulsion_loss_original._last_overlap_zero = (overlap_val < 1e-10).item()
            else:
                overlap_repulsion_loss_original._last_overlap_zero = False
        
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
        
        # Only apply margin when there's actual overlap or very close to overlap
        # Apply margin when overlap_base > 0 (only when there's actual overlap)
        # Tightened threshold: only apply margin when cells are actually overlapping
        overlap_x_base = min_sep_x - dx_abs  # [N, N]
        overlap_y_base = min_sep_y - dy_abs  # [N, N]
        margin_x = margin * (overlap_x_base > 0.0).float()  # [N, N]
        margin_y = margin * (overlap_y_base > 0.0).float()  # [N, N]
        overlap_x_raw = overlap_x_base + margin_x  # [N, N]
        overlap_y_raw = overlap_y_base + margin_y  # [N, N]
        
        # Upper triangle mask
        valid_mask = torch.triu(
            torch.ones(N, N, device=device, dtype=torch.bool), 
            diagonal=1
        )
        
        # Use optimized fused overlap computation (compiled for speed)
        total_penalty = _compute_overlap_chunk_fused(
            xi, yi, wi, hi, areai,
            xj, yj, wj, hj, areaj,
            margin_factor, scale, alpha, valid_mask
        )
        
        # Final check: ensure result is finite
        if not torch.isfinite(total_penalty):
            total_penalty = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        
        # OPTIMIZED: Cache zero check result for early exit optimization
        with torch.no_grad():
            overlap_val = total_penalty.detach()
            if hasattr(overlap_val, 'item'):
                overlap_repulsion_loss_original._last_overlap_zero = (overlap_val < 1e-10).item()
            else:
                overlap_repulsion_loss_original._last_overlap_zero = False
        
        return total_penalty


def overlap_repulsion_loss(cell_features, pin_features, edge_list, epoch_progress=1.0):
    """Main entry point for overlap loss."""
    return overlap_repulsion_loss_original(
        cell_features, pin_features, edge_list, epoch_progress
    )
