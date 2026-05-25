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
import math

import torch
from numba import njit
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


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

    return total_wirelength / edge_list.shape[0]  # Normalize y_key number of edges


"""
Multi-Radii Spatial Grid implementation for overlap loss calculation

Uses a hash based on cell posistions to find cells that are close to each other.
Using this hash, quickly find cell pairs that most likely overlap. Then calculate
overlap loss using those pairs.

Implementation is fast compared to brute force implementation when relatively few cells are overlapping.

Uses numba functions to compile python code for building the grid and finding
the relavent cell pairs that are close to each other, speeding up the implementation.
"""
 
@njit(cache=True)
def _build_grid(pos, bin_size):
    
    """
    Builds spatial grid hash map for calculating which cells are close to each other and likely to overlap
    Spatial grid is stored using a flat array to improve speed.
    
    Input: 
        pos : array giving positions of cell centers
        bin_size : size of the bin radius (Determines how close cells should be to be considered in the same bin)

    Output:
        
        cell_key_sort_ind : Index array to sort cells by their key in the spatial grid.
        bin_start : An array telling the starting index of each bin in cell_key_sort_ind (an array which sorts cells by their key). 
            Lets us quickly check which cells are in the same bin.

        keys : All keys in spatial grid
        STRIDE: Large number used so that we can calculate keys from x, y positions of the cells.
            key = x_key * STRIDE + y_key (similar to calculating an elements position in a flattened 2-d array
                using its row and column)
            Used because we are storing spatial grid in 1-D array, so this prevents incorrect bin collisions
            (Cells that are far from each other shouldn't show up in the same bin)
    """

    # Calculate key in spatial grid for each cell

    N = pos.shape[0]
    STRIDE = np.int64(1 << 20)
    cell_to_key = np.empty(N, dtype=np.int64)
    for i in range(N):
        x_key = np.int64(np.floor(pos[i, 0] / bin_size)) + np.int64(1 << 19)
        y_key = np.int64(np.floor(pos[i, 1] / bin_size)) + np.int64(1 << 19)
        cell_to_key[i] = x_key * STRIDE + y_key
 

    # Sort cells (and their keys) by their key values

    cell_key_sort_ind = np.argsort(cell_to_key)
    sort_cell_to_key = cell_to_key[cell_key_sort_ind]
 
    # Calculate number of bins by finding number of unique key values
    num_bins = np.int64(1)
    for i in range(1, N):
        if sort_cell_to_key[i] != sort_cell_to_key[i - 1]:
            num_bins += 1
 
    keys = np.empty(num_bins, dtype=np.int64)
    bin_start = np.empty(num_bins + 1, dtype=np.int64)
    bin_start[0] = 0
    keys[0] = sort_cell_to_key[0]
    b = np.int64(0)

    # Find all unique keys, and calculate bin_start by finding all bin separating indices
    #   in cell_key_sort_ind

    for i in range(1, N):
        if sort_cell_to_key[i] != sort_cell_to_key[i - 1]:
            b += 1
            bin_start[b] = i
            keys[b] = sort_cell_to_key[i]
    bin_start[num_bins] = N
 
    return cell_key_sort_ind, bin_start, keys, STRIDE
 
 
@njit(cache=True)
def _find_bin(keys, search_key):

    """
    A binary search algorthim that finds the position of search_key within
    the keys array (which has the keys of the spatial grid)
    """

    low, high = np.int64(0), np.int64(len(keys))
    while low < high:
        mid = (low + high) >> np.int64(1)
        if keys[mid] < search_key:
            low = mid + np.int64(1)
        else:
            high = mid
    if low < len(keys) and keys[low] == search_key:
        return low
    return np.int64(-1)
 
 
 
@njit(cache=True)
def _pairs_within_grid(cell_key_sort_ind, bin_start, keys, STRIDE,
                       cell_indices, buf_size):
    """
    Calculates pairs of overlapping cells using a built spatial grid.
    Pairs calculated are unique, so no double counting will occur.

    Input:
        cell_key_sort_ind : Index array to sort cells by their key in the spatial grid.
        bin_start : An array telling the starting index of each bin in cell_key_sort_ind (an array which sorts cells by their key). 
            Lets us quickly find cells in the same bin.

        keys : All keys in spatial grid
        STRIDE: Large number used so that we can calculate keys from x, y positions of the cells.
            key = x_key * STRIDE + y_key (similar to calculating an elements position in a flattened 2-d array
                using its row and column)
            Used because we are storing spatial grid in 1-D array, so this prevents incorrect bin collisions
            (Cells that are far from each other shouldn't show up in the same bin)

        cell_indices : Indices of the cells in the original data array.
        buf_size : Initial size of the arrays used to store the output overlapping pairs.
            (Will be increased in the function if needed)
        
    Output:
        src_buf: Array storing first half of all overlap pairs in the spatial grid.
        dst_buf: Array storing second half of all overlap pairs in the spatial grid.

    The bin_size is set somewhat larger than required to ensure overlapping cell pairs are found.
    As a result, some cell pairs may not be overlapping. These pairs will just have
    zero overlap though, so it won't affect the final overlap loss calculation
    """
    num_bins = len(keys)
    src_buf = np.empty(buf_size, dtype=np.int64)
    dst_buf = np.empty(buf_size, dtype=np.int64)
    n       = np.int64(0)
 
    for b in range(num_bins):

        # Extract each bin

        key_b   = keys[b]
        x_key      = key_b // STRIDE
        y_key      = key_b  % STRIDE
        i_start = bin_start[b]
        i_end   = bin_start[b + 1]
 
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):

                # Find neighboring bins by position

                nb_key = (x_key + np.int64(dx)) * STRIDE + (y_key + np.int64(dy))
                nb_b   = _find_bin(keys, nb_key)
                if nb_b < 0:
                    continue
                j_start = bin_start[nb_b]
                j_end   = bin_start[nb_b + 1]
 
                # Find cells in current bin and neighboring bins
                # Then extract all cell pairs in these bins.
                #  (Neighboring bins are considered to handle edge cases where overlapping cells are
                #   on the borders of 2 bins)
                for ii in range(i_start, i_end):
                    ci = cell_indices[cell_key_sort_ind[ii]]
                    for jj in range(j_start, j_end):
                        cj = cell_indices[cell_key_sort_ind[jj]]

                        # This check ensures each cell pair is only checked once
                        if cj > ci:
                            if n >= buf_size:
                                new_s       = np.empty(buf_size * 2, dtype=np.int64)
                                new_d       = np.empty(buf_size * 2, dtype=np.int64)
                                new_s[:n]   = src_buf[:n]
                                new_d[:n]   = dst_buf[:n]
                                src_buf     = new_s
                                dst_buf     = new_d
                                buf_size   *= 2
                            src_buf[n] = ci
                            dst_buf[n] = cj
                            n += 1
 
    return src_buf[:n], dst_buf[:n]
 
 
@njit(cache=True)
def _pairs_cross_exact(pos_small, dims_small, ids_small,
                       pos_large, dims_large, ids_large,
                       cell_key_sort_ind_large, bin_start_large, keys_large, STRIDE_large, bin_size_large,
                       buf_size, safety):
    
    """
    Calculates pairs of overlapping cells between a group of small cells and a group of large cells.
    Since separate spatial grids are used for the large and small cell groups. An overlap check
    between the small and large cells using the large cell spatial grid is necessary to get total overlap.

    Input:
        pos_small: positions of small cell group.
        dims_small: width and height of small cell group
        ids_small: indices of cells in the small cell group in the original data array.


        pos_large: positions of large cell group.
        dims_large: width and height of large cell group
        ids_large: indices of cells in the large cell group in the original data array.

        cell_key_sort_ind)large : Index array to sort large cells by their key in the large cell spatial grid.
        bin_start_large : An array telling the starting index of each bin in 
            cell_key_sort_ind_large (an array which sorts cells by their key). Lets us quickly find cells in the same bin.

        keys_large : All keys in the large cell spatial grid
        STRIDE_large : Large number used so that we can calculate keys from x, y positions of the cells for the large spatial grid.
            key = x_key * STRIDE + y_key (similar to calculating an elements position in a flattened 2-d array
                using its row and column)
            Used because we are storing spatial grid in 1-D array, so this prevents incorrect bin collisions
            (Cells that are far from each other shouldn't show up in the same bin)

        bin_size_large : radius of large spatial grid used to determine the size of the bins in the grid.

        buf_size : Initial size of the arrays used to store the output overlapping pairs.
            (Will be increased in the function if needed)
            
        safety : A multiplier used in the overlap check calculation between small and large cell pairs
                Makes overlap condition looser, so more pairs are considered. This makes sure that all overlaps
                are found. (Even if non overlapping pairs are found, their overlap value will just be zero anyway,
                so it won't affect the results)

            overlap check calculation:
                |xi - xj| < (wi + wj) / 2 * safety
                |yi - yj| < (hi + hj) / 2 * safety
        
    Output:
        src_buf: Array storing first half of all overlap pairs in the spatial grid.
        dst_buf: Array storing second half of all overlap pairs in the spatial grid.

    This check can be done in relatively linear time, because there are far fewer larger cells than small cells
    in general.
    """
    
    N_small     = len(ids_small)
    src_buf = np.empty(buf_size, dtype=np.int64)
    dst_buf = np.empty(buf_size, dtype=np.int64)
    n       = np.int64(0)
 
    for si in range(N_small):

        # Get cell information for the small cells

        px = pos_small[si, 0]
        py = pos_small[si, 1]
        wi = dims_small[si, 0]
        hi = dims_small[si, 1]
        ci = ids_small[si]
 
        x_key = np.int64(np.floor(px / bin_size_large)) + np.int64(1 << 19)
        y_key = np.int64(np.floor(py / bin_size_large)) + np.int64(1 << 19)
 
        for dx in range(-1, 2):
            for dy in range(-1, 2):

                # Find bins in large spatial grid that neighbor the small cell.

                nb_key = (x_key + np.int64(dx)) * STRIDE_large + (y_key + np.int64(dy))
                nb_b   = _find_bin(keys_large, nb_key)
                if nb_b < 0:
                    continue
 
                j_start = bin_start_large[nb_b]
                j_end   = bin_start_large[nb_b + 1]
 
                for jj in range(j_start, j_end):
                    lj = cell_key_sort_ind_large[jj]          
                    cj = ids_large[lj]        
 
                    # Check if small cells potentially overlap with each large cell in neighboring bin
                    # If it's impossible ignore this cell pair

                    sep_x = (wi + dims_large[lj, 0]) / 2.0 * safety
                    sep_y = (hi + dims_large[lj, 1]) / 2.0 * safety
                    if (abs(px - pos_large[lj, 0]) >= sep_x or
                            abs(py - pos_large[lj, 1]) >= sep_y):
                        continue            
 
                    # Save relevant small-large cell pairs
                    
                    a  = ci if ci < cj else cj
                    b_ = cj if ci < cj else ci
                    if n >= buf_size:
                        new_s       = np.empty(buf_size * 2, dtype=np.int64)
                        new_d       = np.empty(buf_size * 2, dtype=np.int64)
                        new_s[:n]   = src_buf[:n]
                        new_d[:n]   = dst_buf[:n]
                        src_buf     = new_s
                        dst_buf     = new_d
                        buf_size   *= 2
                    src_buf[n] = a
                    dst_buf[n] = b_
                    n += 1
 
    return src_buf[:n], dst_buf[:n]
 
 
 
def multi_radius_pairs(pos_np, dims_np, large_threshold=None, safety=1.5):
    """
    Calculate all potential overlapping cell pairs using spatial grid analysis.
    2 separate grids are used for small and large cells. This reduces the likelihood
    of false positive overlaps being detected as we won't need to use a large radius for the 
    several significantly smaller cells. As a result, overlap calculation will be faster.
 
    bin_size uses median(max_dim) per group so outlier macros don't result in a large radius
    used for the spatial grid, which would reduce convergence time.
 
    Input:
        pos_np: Numpy array storing all cell positions
        dims_np: Numpy array storing all cell widths and heights
        large_threshold: Threshold of cell dimension size (max(width, height)), used to determine if a cell is
                         large or small.
        safety: multiplier on spatial grid bin radius. This helps ensure overlapping pairs will be found (no false negatives)
 
    Output:
        src_buf: Array storing first half of all overlap pairs in the spatial grid.
        dst_buf: Array storing second half of all overlap pairs in the spatial grid.
    """

    # Split cells into large and small cell groups using dimension size

    max_dims = dims_np.max(axis=1)

    if large_threshold is None:
        large_threshold = float(np.percentile(max_dims, 85))
 
    ids_small = np.where(max_dims <= large_threshold)[0].astype(np.int64)
    ids_large = np.where(max_dims >  large_threshold)[0].astype(np.int64)
 
    pos_small  = pos_np[ids_small]
    dims_small = dims_np[ids_small]
    pos_large  = pos_np[ids_large]
    dims_large = dims_np[ids_large]
 
    # Use median of group dims for bin_size of each spatial grid — immune to outliers.
    def _bin_size(d):
        return safety * float(np.median(d.max(axis=1))) * 2.0 if len(d) > 0 else 1.0
 
    bs_small = _bin_size(dims_small) if len(ids_small) > 1 else 1.0
    bs_large = _bin_size(dims_large) if len(ids_large) > 0 else 1.0
 
    all_src, all_dst = [], []
 
    # Calculate potential overlap pairs for small x small cells
    if len(ids_small) > 1:
        order_small, bin_start_small, keys_small, STRIDE_small = _build_grid(
            pos_small.astype(np.float64), bs_small)
        s, d = _pairs_within_grid(
            order_small, bin_start_small, keys_small, STRIDE_small,
            ids_small, max(16 * len(ids_small), 64))
        all_src.append(s); all_dst.append(d)
 
    # Calculate potential overlap pairs for large x large cells
    if len(ids_large) > 1:
        cell_key_sort_ind_large, bin_start_large, keys_large, STRIDE_large = _build_grid(
            pos_large.astype(np.float64), bs_large)
        s, d = _pairs_within_grid(
            cell_key_sort_ind_large, bin_start_large, keys_large, STRIDE_large,
            ids_large, max(16 * len(ids_large), 64))
        all_src.append(s); all_dst.append(d)
 
    # Calculate potential overlap pairs for small x large cells
    if len(ids_small) > 0 and len(ids_large) > 0:
        cell_key_sort_ind_large, bin_start_large, keys_large, STRIDE_large = _build_grid(
            pos_large.astype(np.float64), bs_large)
        s, d = _pairs_cross_exact(
            pos_small, dims_small, ids_small,
            pos_large, dims_large, ids_large,
            cell_key_sort_ind_large, bin_start_large, keys_large, STRIDE_large, bs_large,
            max(16 * len(ids_small), 64), safety)
        all_src.append(s); all_dst.append(d)
 
    if not all_src:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    return np.concatenate(all_src), np.concatenate(all_dst)
 

def overlap_repulsion_loss(cell_features, pin_features, edge_list):
    """Calculate loss to prevent cell overlaps.

    TODO: IMPLEMENT THIS FUNCTION

    This is the main challenge. You need to implement a differentiable loss function
    that penalizes overlapping cells. The loss should:

    1. Be zero when no cells overlap
    2. Increase as overlap area increases
    3. Use only differentiable PyTorch operations (no if statements on tensors)
    4. Work efficiently with vectorized operations

    HINTS:
    - Two axis-aligned rectangles overlap if they overlap in BOTH x and y dimensions
    - For rectangles centered at (x1, y1) and (x2, y2) with widths (w1, w2) and heights (h1, h2):
      * x-overlap occurs when |x1 - x2| < (w1 + w2) / 2
      * y-overlap occurs when |y1 - y2| < (h1 + h2) / 2
    - Use torch.relu() to compute positive overlaps: overlap_x = relu((w1+w2)/2 - |x1-x2|)
    - Overlap area = overlap_x * overlap_y
    - Consider all pairs of cells: use broadcasting with unsqueeze
    - Use torch.triu() to avoid counting each pair twice (only consider i < j)
    - Normalize the loss appropriately (by number of pairs or total area)

    RECOMMENDED APPROACH:
    1. Extract positions, widths, heights from cell_features
    2. Compute all pairwise distances using broadcasting:
       positions_i = positions.unsqueeze(1)  # [N, 1, 2]
       positions_j = positions.unsqueeze(0)  # [1, N, 2]
       distances = positions_i - positions_j  # [N, N, 2]
    3. Calculate minimum separation distances for each pair
    4. Use relu to get positive overlap amounts
    5. Multiply overlaps in x and y to get overlap areas
    6. Mask to only consider upper triangle (i < j)
    7. Sum and normalize

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

    # TODO: Implement overlap detection and loss calculation here
    #
    # Your implementation should:
    # 1. Extract cell positions, widths, and heights
    # 2. Compute pairwise overlaps using vectorized operations
    # 3. Return a scalar loss that is zero when no overlaps exist
    #
    # Delete this placeholder and add your implementation:


    # calculate cell positions and dimensions (widths and heights)
    pos  = cell_features[:, 2:4]
    dims = cell_features[:, 4:6]
 
    # create numpy arrays corresponding to the cell positions and heights
    # Note: we do not need the gradients generated from operations used to 
    #  calculate potential overlap pairs with spatial grid analysis, only the gradients
    #  from the overlap calculations of said pairs. Therefore, we can perform
    #  spatial grid analysis with numpy operations instead of pytorch operations.
    with torch.no_grad():
        pos_np  = pos.detach().cpu().numpy().astype(np.float64)
        dims_np = dims.detach().cpu().numpy().astype(np.float64)
 

    # Find all overlapping pairs using spatial grid analysis
    large_threshold = None
    safety_margin = 1.5

    src_np, dst_np = multi_radius_pairs(
        pos_np, dims_np,
        large_threshold=large_threshold,
        safety=safety_margin,
    )
 
    # no cell overlapping pairs found
    if src_np.size == 0:
        return torch.tensor(0.0, requires_grad=True)
 

    # calculate overlap for each cell pair found using formula described in the problem statement
    src = torch.from_numpy(src_np)
    dst = torch.from_numpy(dst_np)
 
    dist = (pos[src] - pos[dst]).abs()
    sep  = (dims[src] + dims[dst]) / 2.0
    ov   = torch.relu(sep - dist)
    area = ov[:, 0] * ov[:, 1]
 
    # Normalize overlap loss using number of cells
    return area.sum() / N


def train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=7000,
    lr=0.11,
    lambda_wirelength=5.0,
    lambda_overlap=1.0,
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

    torch.set_num_threads(6)

    # Clone features and create learnable positions
    cell_features = cell_features.clone()
    initial_cell_features = cell_features.clone()

    # Make only cell positions require gradients
    cell_positions = cell_features[:, 2:4].clone().detach()
    cell_positions.requires_grad_(True)

    # Create optimizer
    optimizer = optim.Adam([cell_positions], lr=lr)

    # Create learning rate scheduler to reach a strong loss minima
    lr_schedule = CosineAnnealingWarmRestarts(optimizer, int(num_epochs/4))

    # Track loss history
    loss_history = {
        "total_loss": [],
        "wirelength_loss": [],
        "overlap_loss": [],
    }

    # Calculate lambda ovelap using number of cell_features, this makes so that when overlap is small
    # but nonzero, optimizer will continue reducing overlap to zero.

    lambda_wirelength = 1.0
    lambda_overlap = cell_features.size(0) / 2.0

    # Training loop
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

        # Combined loss
        total_loss = lambda_wirelength * wl_loss + lambda_overlap * overlap_loss

        # Backward pass
        total_loss.backward()

        # Gradient clipping to prevent extreme updates
        torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=5.0)

        # Update positions and learning rate
        optimizer.step()
        lr_schedule.step()

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


        # Temporarily increase wirelength loss parameter to promote lower wirelength value in optimization
        if epoch == 500:
            lambda_wirelength = 100.0
            lambda_overlap = 1.0

        # Calculate lambda ovelap using number of cell_features, this makes so that when overlap is small
        # but nonzero, optimizer will continue reducing overlap to zero.
        if epoch == 1000:
            lambda_wirelength = 1.0
            lambda_overlap = cell_features.size(0)

    # Create final cell features
    final_cell_features = cell_features.clone()
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
    #total_cells = cell_features.shape[0]
    #spread_radius = 30.0
    #angles = torch.rand(total_cells) * 2 * 3.14159
    #radii = torch.rand(total_cells) * spread_radius

    #cell_features[:, 2] = radii * torch.cos(angles)
    #cell_features[:, 3] = radii * torch.sin(angles)

    # Initialize cells by evenly spreading them out based on max dimension size and number of cells
    # Cells are placed around in a square pattern

    N = cell_features.shape[0]
    sqrt_N = math.ceil(math.sqrt(N))

    max_width = cell_features[:, 4].max().item()
    max_height = cell_features[:, 5].max().item()

    x_pos = 0.0
    y_pos= 0.0

    col_counter = 0

    x_init = []
    y_init = []

    for i in range(N):
        if col_counter > sqrt_N:
            col_counter = 0
            x_pos = 0.0
            y_pos += max_height/40

        x_init.append(x_pos)
        y_init.append(y_pos)

        x_pos += max_width/40
        col_counter += 1

    x_init = torch.tensor(x_init)
    y_init = torch.tensor(y_init)

    cell_features[:, CellFeatureIdx.X] = x_init
    cell_features[:, CellFeatureIdx.Y] = y_init

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
