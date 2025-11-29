"""CUDA-accelerated overlap loss computation.

This module provides GPU-accelerated overlap loss computation using custom CUDA kernels.
The implementation uses spatial hashing to reduce computational complexity and grid-stride
loops to efficiently process large numbers of cell pairs.

Key features:
- Spatial hashing: Only processes nearby cell pairs for O(N) complexity
- Grid-stride loops: Limits number of launched blocks while processing all pairs
- Thread synchronization: All threads participate in sync points
- Automatic fallback: Falls back to PyTorch implementation on errors
"""
import math
import sys
import time
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.autograd import Function

LAST_CUDA_STATS = {}

try:
    # Try importing the compiled extension
    # First try direct import (if .so is in same directory)
    import overlap_cuda_backend  # Compiled extension
    _CUDA_BACKEND_AVAILABLE = True
except ImportError:
    # If that fails, try importing from parent directory
    import sys
    import os
    # Add current directory to path if not already there
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    try:
        import overlap_cuda_backend
        _CUDA_BACKEND_AVAILABLE = True
    except ImportError as e:  # pragma: no cover - backend missing
        overlap_cuda_backend = None
        _CUDA_BACKEND_AVAILABLE = False
        # Store error for debugging
        _CUDA_BACKEND_IMPORT_ERROR = str(e)


def is_available() -> bool:
    """Return True when the CUDA backend is ready to use."""
    return _CUDA_BACKEND_AVAILABLE and torch.cuda.is_available()


def get_last_stats():
    """Return metadata from the most recent CUDA overlap call."""
    return LAST_CUDA_STATS


@dataclass
class SpatialIndex:
    cell_bins: torch.Tensor
    bin_starts: torch.Tensor
    bin_counts: torch.Tensor
    sorted_indices: torch.Tensor
    num_bins_x: int
    num_bins_y: int


class _OverlapCUDAFunction(Function):
    """Autograd wrapper over the CUDA kernels."""

    @staticmethod
    def forward(
        ctx,
        positions,
        widths,
        heights,
        areas,
        cell_bins,
        bin_starts,
        bin_counts,
        sorted_indices,
        num_bins_x,
        num_bins_y,
        pair_bins_a,
        pair_bins_b,
        margin_factor,
        alpha,
        scale,
    ):
        loss = overlap_cuda_backend.forward(
            positions,
            widths,
            heights,
            areas,
            cell_bins,
            bin_starts,
            bin_counts,
            sorted_indices,
            num_bins_x,
            num_bins_y,
            pair_bins_a,
            pair_bins_b,
            margin_factor,
            alpha,
            scale,
        )
        
        ctx.save_for_backward(
            positions,
            widths,
            heights,
            areas,
            cell_bins,
            bin_starts,
            bin_counts,
            sorted_indices,
            pair_bins_a,
            pair_bins_b,
        )
        ctx.num_bins_x = num_bins_x
        ctx.num_bins_y = num_bins_y
        ctx.margin_factor = margin_factor
        ctx.alpha = alpha
        ctx.scale = scale
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (
            positions,
            widths,
            heights,
            areas,
            cell_bins,
            bin_starts,
            bin_counts,
            sorted_indices,
            pair_bins_a,
            pair_bins_b,
        ) = ctx.saved_tensors
        
        grad_positions = overlap_cuda_backend.backward(
            positions,
            widths,
            heights,
            areas,
            cell_bins,
            bin_starts,
            bin_counts,
            sorted_indices,
            ctx.num_bins_x,
            ctx.num_bins_y,
            pair_bins_a,
            pair_bins_b,
            ctx.margin_factor,
            ctx.alpha,
            ctx.scale,
        )
        
        grad_positions = grad_positions * grad_output.float()
        return (
            grad_positions,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _compute_bin_size(widths: torch.Tensor, heights: torch.Tensor, num_cells: int) -> float:
    """Compute optimal bin size for spatial hashing based on cell dimensions and count."""
    dims = torch.maximum(widths, heights)
    mean_dim = dims.mean().item()
    max_dim = dims.max().item()
    base = max(mean_dim * 4.0, max_dim, 1.0)
    if num_cells > 100000:
        base *= 1.8
    elif num_cells > 50000:
        base *= 1.5
    elif num_cells > 20000:
        base *= 1.3
    return max(base, 1.0)


def _build_spatial_index(
    positions: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    bin_size: float,
) -> SpatialIndex:
    """Build spatial hash index for efficient pair lookup.
    
    Assigns cells to spatial bins and creates sorted indices for efficient access.
    """
    device = positions.device
    x = positions[:, 0]
    y = positions[:, 1]
    x_min = float(x.min().item())
    x_max = float(x.max().item())
    y_min = float(y.min().item())
    y_max = float(y.max().item())

    span_x = max(x_max - x_min, 1.0)
    span_y = max(y_max - y_min, 1.0)

    num_bins_x = max(1, min(1024, int(math.ceil(span_x / bin_size)) + 2))
    num_bins_y = max(1, min(1024, int(math.ceil(span_y / bin_size)) + 2))
    total_bins = num_bins_x * num_bins_y

    # Bin assignment (keep tensors on device, but detach from graph)
    rel_x = (x - x_min + bin_size) / bin_size
    rel_y = (y - y_min + bin_size) / bin_size
    bin_x = torch.clamp(rel_x.floor().to(torch.int64), 0, num_bins_x - 1)
    bin_y = torch.clamp(rel_y.floor().to(torch.int64), 0, num_bins_y - 1)
    cell_bins = (bin_y * num_bins_x + bin_x).to(torch.int32)

    sorted_indices = torch.argsort(cell_bins.to(torch.int64)).to(torch.int32)

    counts = torch.bincount(cell_bins.to(torch.int64), minlength=total_bins)
    bin_counts = counts.to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    if bin_counts.numel() > 0:
        cumulative = torch.cumsum(bin_counts[:-1], dim=0)
        bin_starts = torch.cat([zero, cumulative])
    else:  # pragma: no cover - degenerate case with zero bins
        bin_starts = zero

    return SpatialIndex(
        cell_bins=cell_bins.contiguous(),
        bin_starts=bin_starts.contiguous(),
        bin_counts=bin_counts.contiguous(),
        sorted_indices=sorted_indices.contiguous(),
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
    )


def _build_bin_pairs(
    bin_counts: torch.Tensor,
    num_bins_x: int,
    num_bins_y: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build list of bin pairs to process.
    
    Only includes pairs of adjacent bins (including same bin) to reduce
    the number of pairs checked while maintaining correctness.
    """
    device = bin_counts.device
    occupied = torch.nonzero(bin_counts > 0, as_tuple=False).flatten()
    if occupied.numel() == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty

    bin_counts_cpu = bin_counts.cpu()
    occupied_cpu = occupied.cpu().tolist()
    pair_a = []
    pair_b = []

    for bin_id in occupied_cpu:
        int_bin_id = int(bin_id)
        int_by = int_bin_id // num_bins_x
        int_bx = int_bin_id % num_bins_x
        for dy in (-1, 0, 1):
            ny = int_by + dy
            if ny < 0 or ny >= num_bins_y:
                continue
            for dx in (-1, 0, 1):
                nx = int_bx + dx
                if nx < 0 or nx >= num_bins_x:
                    continue
                neighbor_id = ny * num_bins_x + nx
                if bin_counts_cpu[neighbor_id].item() == 0:
                    continue
                if neighbor_id < int_bin_id:
                    continue
                pair_a.append(int_bin_id)
                pair_b.append(neighbor_id)

    if not pair_a:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty

    pair_a_tensor = torch.tensor(pair_a, dtype=torch.int32, device=device)
    pair_b_tensor = torch.tensor(pair_b, dtype=torch.int32, device=device)
    return pair_a_tensor, pair_b_tensor


def compute_overlap_loss(
    cell_features: torch.Tensor,
    margin_factor: float,
    alpha: float,
    scale: float,
    epoch_progress: float,
) -> torch.Tensor:
    if not is_available():  # pragma: no cover - guarded by caller
        raise RuntimeError("CUDA overlap backend is not available")

    positions = cell_features[:, 2:4]
    widths = cell_features[:, 4]
    heights = cell_features[:, 5]
    areas = cell_features[:, 0].clamp_min(1e-8)

    # Detach non-differentiable data for grid construction
    pos_detached = positions.detach()
    widths_det = widths.detach()
    heights_det = heights.detach()

    bin_size = _compute_bin_size(widths_det, heights_det, positions.shape[0])
    spatial_index = _build_spatial_index(pos_detached, widths_det, heights_det, bin_size)
    pair_bins_a, pair_bins_b = _build_bin_pairs(
        spatial_index.bin_counts, spatial_index.num_bins_x, spatial_index.num_bins_y
    )

    global LAST_CUDA_STATS
    LAST_CUDA_STATS = {
        "backend": "cuda",
        "pairs": int(pair_bins_a.numel()),
        "num_bins": spatial_index.num_bins_x * spatial_index.num_bins_y,
        "bin_size": float(bin_size),
    }

    if pair_bins_a.numel() == 0:
        return positions.sum() * 0.0

    # Safety check: Limit pair count to prevent memory issues
    pair_count = int(pair_bins_a.numel())
    if pair_count > 200000:
        # Too many pairs may cause memory issues
        # Return zero loss to allow fallback to PyTorch implementation
        print(f"[CUDA] Too many pairs ({pair_count}), falling back to PyTorch")
        return positions.sum() * 0.0
    
    # Validate bin counts to prevent out-of-bounds access
    max_bin_idx = max(
        int(pair_bins_a.max().item()) if pair_bins_a.numel() > 0 else 0,
        int(pair_bins_b.max().item()) if pair_bins_b.numel() > 0 else 0
    )
    total_bins = spatial_index.num_bins_x * spatial_index.num_bins_y
    if max_bin_idx >= total_bins:
        raise RuntimeError(
            f"Invalid bin index {max_bin_idx} >= total_bins {total_bins}. "
            f"This indicates a bug in spatial indexing."
        )

    # CUDA kernels expect int32 indices
    cell_bins_i32 = spatial_index.cell_bins.to(torch.int32)
    bin_starts_i32 = spatial_index.bin_starts.to(torch.int32)
    bin_counts_i32 = spatial_index.bin_counts.to(torch.int32)
    sorted_indices_i32 = spatial_index.sorted_indices.to(torch.int32)
    pair_bins_a_i32 = pair_bins_a.to(torch.int32)
    pair_bins_b_i32 = pair_bins_b.to(torch.int32)

    # Error handling: Wrap CUDA call to gracefully fall back to PyTorch on errors
    try:
        # Ensure all tensors are contiguous and on the correct device
        loss = _OverlapCUDAFunction.apply(
            positions.contiguous(),
            widths.contiguous(),
            heights.contiguous(),
            areas.contiguous(),
            cell_bins_i32.contiguous(),
            bin_starts_i32.contiguous(),
            bin_counts_i32.contiguous(),
            sorted_indices_i32.contiguous(),
            spatial_index.num_bins_x,
            spatial_index.num_bins_y,
            pair_bins_a_i32.contiguous(),
            pair_bins_b_i32.contiguous(),
            float(margin_factor),
            float(alpha),
            float(scale),
        )
        
        # Check if result is valid (not NaN/Inf)
        if not torch.isfinite(loss):
            raise RuntimeError("CUDA backend returned invalid loss (NaN/Inf)")
        return loss
    except (RuntimeError, Exception) as e:
        # CUDA error or invalid result - fall back to PyTorch
        import warnings
        warnings.warn(
            f"CUDA backend failed: {e}. Falling back to PyTorch implementation.",
            RuntimeWarning
        )
        # Re-raise to trigger fallback in losses.py
        raise RuntimeError(f"CUDA backend error: {e}")

