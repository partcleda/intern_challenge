"""
Evaluation metrics for VLSI placement.
"""

import numpy as np
from .losses import wirelength_attraction_loss

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
    Uses spatial hashing for O(N) complexity on large problems.

    Args:
        cell_features: [N, 6] tensor with cell properties

    Returns:
        Set of cell indices that have overlaps with other cells
    """
    N = cell_features.shape[0]
    if N <= 1:
        return set()

    # For small problems, use simple O(N²) approach
    if N < 5000:
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
    
    # For large problems, use spatial hashing (O(N) complexity)
    # Convert to numpy for efficient spatial hashing
    positions = cell_features[:, 2:4].detach().cpu().numpy()
    widths = cell_features[:, 4].detach().cpu().numpy()
    heights = cell_features[:, 5].detach().cpu().numpy()
    
    cells_with_overlaps = set()
    
    # Spatial hashing: bin cells by position
    # Use adaptive bin size based on problem size
    x = positions[:, 0]
    y = positions[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Bin size: use average cell size * 2 for good coverage
    avg_width = widths.mean()
    avg_height = heights.mean()
    bin_size = max(avg_width, avg_height) * 2.0
    
    # Create bins
    num_bins_x = max(1, int((x_max - x_min) / bin_size) + 1)
    num_bins_y = max(1, int((y_max - y_min) / bin_size) + 1)
    
    # Bin cells
    bin_x = ((x - x_min) / bin_size).astype(int).clip(0, num_bins_x - 1)
    bin_y = ((y - y_min) / bin_size).astype(int).clip(0, num_bins_y - 1)
    bin_idx = bin_y * num_bins_x + bin_x
    
    # Group cells by bin
    bins = {}
    for i in range(N):
        b = bin_idx[i]
        if b not in bins:
            bins[b] = []
        bins[b].append(i)
    
    # Check pairs within same bin and adjacent bins
    checked_pairs = set()
    for bin_id, cell_indices in bins.items():
        bin_x_coord = bin_id % num_bins_x
        bin_y_coord = bin_id // num_bins_x
        
        # Check same bin and 8 adjacent bins
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx = bin_x_coord + dx
                ny = bin_y_coord + dy
                if 0 <= nx < num_bins_x and 0 <= ny < num_bins_y:
                    neighbor_bin_id = ny * num_bins_x + nx
                    if neighbor_bin_id in bins:
                        neighbor_cells = bins[neighbor_bin_id]
                        # Check pairs
                        for i in cell_indices:
                            for j in neighbor_cells:
                                if i >= j:  # Only check upper triangle
                                    continue
                                pair_key = (i, j)
                                if pair_key in checked_pairs:
                                    continue
                                checked_pairs.add(pair_key)
                                
                                # Calculate overlap
                                dx_val = abs(positions[i, 0] - positions[j, 0])
                                dy_val = abs(positions[i, 1] - positions[j, 1])
                                
                                min_sep_x = (widths[i] + widths[j]) / 2
                                min_sep_y = (heights[i] + heights[j]) / 2
                                
                                overlap_x = max(0, min_sep_x - dx_val)
                                overlap_y = max(0, min_sep_y - dy_val)
                                
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


