import torch

def std_repulsion_loss(cell_features, pin_features, edge_list, n_neighbors=20):
    """
    Penalize overlaps between up to the nth neighbor in sorted x and y (O(n*k)).
    Args:
        cell_features: [N, 6] tensor
        pin_features: unused
        edge_list: unused
        max_neighbor: int, number of neighbors to check (default 2)
    Returns:
        Scalar loss (differentiable)
    """
    N = cell_features.shape[0]
    if N <= n_neighbors:
        return torch.tensor(0.0, requires_grad=True)

    x = cell_features[:, 2]
    y = cell_features[:, 3]
    w = cell_features[:, 4]
    h = cell_features[:, 5]

    total_overlap = 0.0

    # --- Sorted by x ---
    x_sorted, idx_x = torch.sort(x)
    y_sorted_x = y[idx_x]
    w_sorted_x = w[idx_x]
    h_sorted_x = h[idx_x]

    # --- Sorted by y ---
    y_sorted, idx_y = torch.sort(y)
    x_sorted_y = x[idx_y]
    w_sorted_y = w[idx_y]
    h_sorted_y = h[idx_y]

    for k in range(1, n_neighbors + 1):
        # x-sorted neighbors
        dx = x_sorted[k:] - x_sorted[:-k]
        sum_w = (w_sorted_x[k:] + w_sorted_x[:-k]) / 2
        overlap_x = torch.relu(sum_w - dx)

        dy = torch.abs(y_sorted_x[k:] - y_sorted_x[:-k])
        sum_h = (h_sorted_x[k:] + h_sorted_x[:-k]) / 2
        overlap_y = torch.relu(sum_h - dy)

        overlap_area = overlap_x * overlap_y
        total_overlap = total_overlap + overlap_area.sum()

        # y-sorted neighbors
        dy_y = y_sorted[k:] - y_sorted[:-k]
        sum_h_y = (h_sorted_y[k:] + h_sorted_y[:-k]) / 2
        overlap_y_y = torch.relu(sum_h_y - dy_y)

        dx_y = torch.abs(x_sorted_y[k:] - x_sorted_y[:-k])
        sum_w_y = (w_sorted_y[k:] + w_sorted_y[:-k]) / 2
        overlap_x_y = torch.relu(sum_w_y - dx_y)

        overlap_area_y = overlap_x_y * overlap_y_y
        total_overlap = total_overlap + overlap_area_y.sum()

    return total_overlap / N


def macro_repulsion_loss(cell_features, pin_features, edge_list, padding_constant = 0.25):
    N = cell_features.shape[0]
    if N <= 1:
        return torch.tensor(0.0, requires_grad=True)
    
    areas = cell_features[:, 0]
    macros = areas > 100
    positions = cell_features[:, 2:4]
    positions_i = positions[macros].unsqueeze(1)    # [M, 1, 2]
    positions_j = positions.unsqueeze(0)            # [1, N, 2]
    distances = positions_i - positions_j           # [M, N, 2]
    distances = torch.abs(distances)

    dims = cell_features[:, 4:]
    dims_i = dims[macros].unsqueeze(1)              # [M, 1, 2]
    dims_j = dims.unsqueeze(0)                      # [1, N, 2]
    added_dims = (dims_i + dims_j) / 2              # [M, N, 2]
    added_dims += padding_constant

    overlap = torch.relu(added_dims - distances)
    overlap_areas = overlap[:,:,0] * overlap[:,:,1]
    overlap_areas = torch.triu(overlap_areas, diagonal=1)

    total_overlap_area = torch.sum(overlap_areas @ (areas ** 0.5))

    loss_score = total_overlap_area / N
    return loss_score

def overlap_repulsion_loss(cell_features, pin_features, edge_list, padding_constant = 5):
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

    areas = cell_features[:, 0]
    # macros = areas > 1000
    positions = cell_features[:, 2:4]
    positions_i = positions.unsqueeze(1)    # [N, 1, 2]
    positions_j = positions.unsqueeze(0)    # [1, N, 2]
    distances = positions_i - positions_j   # [N, N, 2]
    distances = torch.abs(distances)

    dims = cell_features[:, 4:]
    dims_i = dims.unsqueeze(1)              # [N, 1, 2]
    dims_j = dims.unsqueeze(0)              # [1, N, 2]
    added_dims = (dims_i + dims_j) / 2      # [N, N, 2]
    added_dims += padding_constant

    overlap = torch.relu(added_dims - distances)
    overlap_areas = overlap[:,:,0] * overlap[:,:,1]
    overlap_areas = torch.triu(overlap_areas, diagonal=1)

    total_overlap_area = torch.sum(overlap_areas@areas)

    loss_score = total_overlap_area / N
    return loss_score
