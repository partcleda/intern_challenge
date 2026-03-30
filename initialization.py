from collections import deque
import torch

from data import PinFeatureIdx, CellFeatureIdx
from constants import MIN_MACRO_AREA, CELL_SPACING, MAX_ROW_WIDTH

# from placement import generate_placement_input, plot_placement, calculate_normalized_metrics


def build_cell_level_edge_list(pin_features, edge_list):
    """
    Convert a pin-level edge list to a cell-level edge list.

    Each edge [src_pin_idx, tgt_pin_idx] is mapped to
    [src_cell_idx, tgt_cell_idx] via pin_features[:, PinFeatureIdx.CELL_IDX].

    Self-edges (both pins on the same cell) and duplicates are removed.

    Args:
        pin_features: [P, 7] tensor; column 0 is the owning cell index
        edge_list:    [E, 2] tensor of pin-index pairs [src_pin_idx, tgt_pin_idx]
    Returns:
        cell_edge_list: [E', 2] tensor of deduplicated cell-index pairs
    """
    if edge_list.shape[0] == 0:
        return torch.zeros((0, 2), dtype=torch.long)

    pin_to_cell = pin_features[:, PinFeatureIdx.CELL_IDX].long()  # [P]
    src_cells = pin_to_cell[edge_list[:, 0]]  # [E]
    tgt_cells = pin_to_cell[edge_list[:, 1]]  # [E]

    # Remove self-edges — pins on the same cell have no placement implication
    valid     = src_cells != tgt_cells
    src_cells = src_cells[valid]
    tgt_cells = tgt_cells[valid]

    # Canonicalise so smaller index is always first, then deduplicate
    # This collapses both (i,j) and (j,i) into one undirected edge
    cell_edges = torch.stack([
        torch.minimum(src_cells, tgt_cells),
        torch.maximum(src_cells, tgt_cells),
    ], dim=1)

    return torch.unique(cell_edges, dim=0)  # [E', 2]


def build_adjacency_list(cell_edge_list, num_cells):
    """
    Build an undirected adjacency list from a cell-level edge list.

    Args:
        cell_edge_list: [E, 2] tensor of cell-index pairs
        num_cells:      int, total number of cells
    Returns:
        adjacency: List[List[int]], one entry per cell
    """
    adjacency = [[] for _ in range(num_cells)]
    for src, tgt in cell_edge_list.tolist():
        src, tgt = int(src), int(tgt)
        adjacency[src].append(tgt)
        adjacency[tgt].append(src)
    return adjacency


def compute_bfs_scores_from_macros(adjacency, macro_indices, num_cells, decay=0.5):
    """
    For each macro, run a BFS outward through the cell graph and assign a
    dampened reachability score to every cell it reaches.

    Traversal rules:
      - BFS starts at the macro and expands freely through std cells
      - When another macro is reached, record its score but do NOT expand through it
        (macros act as boundaries — paths cannot pass through them)
      - Score contribution at each cell = decay ^ depth, where depth is the
        number of edges from the source macro to that cell

    This produces a score matrix where:
    scores[m][c] = decay ^ shortest_path_depth(macro_m, cell_c); 0 if cell_c is unreachable from macro_m
    Slicing gives both outputs needed downstream:
      macro_to_macro[m][j] = scores[m][macro_indices[j]]   — for macro ordering
      macro_to_std[m][s]   = scores[m][std_indices[s]]     — for std cell assignment

    Args:
        cell_edge_list: [E, 2] tensor of global cell-index pairs (already cell-level)
        macro_indices:  List[int], global cell indices of macros
        num_cells:      int, total number of cells N
        decay:          float in (0, 1), per-hop dampening factor
    Returns:
        scores: torch.Tensor of shape [M, num_cells], dtype float32
    """
    M         = len(macro_indices)
    macro_set = set(macro_indices)
    scores    = torch.zeros(M, num_cells, dtype=torch.float32)

    for m, macro_idx in enumerate(macro_indices):
        visited = {macro_idx}
        scores[m, macro_idx] = 1.0 # source scores itself at decay^0
        # Queue entries: (cell_index, depth, accumulated_decay)
        # Carrying accumulated_decay avoids recomputing decay^depth each step
        queue = deque([(macro_idx, 0, 1.0)])

        while queue:
            cell, depth, cell_decay = queue.popleft()

            neighbor_decay = cell_decay * decay   # decay^(depth+1)

            for neighbor in adjacency[cell]:
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                scores[m, neighbor] += neighbor_decay

                if neighbor in macro_set:
                    # Boundary reached — score recorded, do not expand
                    continue

                # Std cell — expand through it
                queue.append((neighbor, depth + 1, neighbor_decay))
    return scores # [M, num_cells]

def order_std_cells(score_matrix, std_indices):
    """
    Sort std cells so that cells connected to the same macro
    are adjacent in the ordering. Within each macro group,
    strongest affinity comes first.
    
    Args:
        score_matrix: [M, num_cells] tensor
        std_indices:  List[int], global indices of std cells
    Returns:
        ordered: List[int], indices into std_indices in packing order
    """
    std_scores = score_matrix[:, std_indices]  # [M, S]
    primary = torch.argmax(std_scores, dim=0)  # [S]
    affinity = std_scores.gather(0, primary.unsqueeze(0)).squeeze(0)

    sort_key = primary.float() - affinity / (affinity.max() + 1e-12)
    return torch.argsort(sort_key).tolist()


def pack_std_cells(order, std_indices, cell_features,
                   row_height=1.0, spacing=CELL_SPACING):
    """
    Tile std cells into a dense rectangular block following
    the given order. Left-to-right, top-to-bottom.
    
    Args:
        order:         List[int], indices into std_indices
        std_indices:   List[int], global cell indices
        cell_features: [N, 6] tensor
        row_height:    float
        spacing:       float
    Returns:
        cell_features: updated clone with std positions written
        bbox:          (x_min, x_max, y_min, y_max) of the block
    """
    cell_features = cell_features.clone()
    widths = cell_features[std_indices, CellFeatureIdx.WIDTH]

    total_width = widths.sum().item()
    num_rows = max(1, int(total_width ** 0.5 / row_height))
    row_width = total_width / num_rows

    cursor_x = 0.0
    cursor_y = row_height / 2

    for s in order:
        idx = std_indices[s]
        w = widths[s].item()

        if cursor_x + w > row_width:
            cursor_y += row_height + spacing
            cursor_x = 0.0

        cell_features[idx, CellFeatureIdx.X] = cursor_x + w / 2
        cell_features[idx, CellFeatureIdx.Y] = cursor_y
        cursor_x += w + spacing

    std_x = cell_features[std_indices, CellFeatureIdx.X]
    std_y = cell_features[std_indices, CellFeatureIdx.Y]
    std_w = cell_features[std_indices, CellFeatureIdx.WIDTH]
    std_h = cell_features[std_indices, CellFeatureIdx.HEIGHT]
    bbox = (
        (std_x - std_w / 2).min().item(),
        (std_x + std_w / 2).max().item(),
        (std_y - std_h / 2).min().item(),
        (std_y + std_h / 2).max().item(),
    )

    return cell_features, bbox

def place_macros_around_block(score_matrix, macro_indices, std_indices,
                              cell_features, bbox, standoff=5.0):
    """
    Place each macro outside the std cell block, on the side
    where its most connected std cells live.
    """
    x_min, x_max, y_min, y_max = bbox
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    std_xy = cell_features[std_indices][:, [CellFeatureIdx.X, CellFeatureIdx.Y]]

    cell_features = cell_features.clone()

    for m, macro_idx in enumerate(macro_indices):
        weights = score_matrix[m, std_indices]
        w_sum = weights.sum().clamp(min=1e-12)
        target = (weights @ std_xy) / w_sum

        dx = target[0] - cx
        dy = target[1] - cy
        length = (dx**2 + dy**2).sqrt().clamp(min=1e-12)
        dx, dy = dx / length, dy / length

        mw = cell_features[macro_idx, CellFeatureIdx.WIDTH].item()
        mh = cell_features[macro_idx, CellFeatureIdx.HEIGHT].item()
        half_w = (x_max - x_min) / 2 + mw / 2 + standoff
        half_h = (y_max - y_min) / 2 + mh / 2 + standoff

        t_x = half_w / abs(dx) if abs(dx) > 1e-12 else float('inf')
        t_y = half_h / abs(dy) if abs(dy) > 1e-12 else float('inf')
        t = min(t_x, t_y)

        cell_features[macro_idx, CellFeatureIdx.X] = cx + dx * t
        cell_features[macro_idx, CellFeatureIdx.Y] = cy + dy * t

    return cell_features

def create_virtual_macro(bbox):
    """
    Create a single-row cell feature tensor representing
    the std cell block as one large immovable macro.

    Args:
        bbox: (x_min, x_max, y_min, y_max) of packed std cells
    Returns:
        virtual_macro: [1, 6] tensor with
            [area, num_pins=0, cx, cy, width, height]
    """
    x_min, x_max, y_min, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    return torch.tensor([[
        width * height,  # area
        0.0,             # num_pins (no pins — wirelength ignores it)
        (x_min + x_max) / 2,  # x center
        (y_min + y_max) / 2,  # y center
        width,
        height,
    ]])

def initialize_placement(cell_features, pin_features, edge_list):
    """
    Full initialization pipeline:
      1. Build cell-level graph from pin connectivity
      2. BFS from each macro to score reachability to all cells
      3. Order std cells by macro affinity (clustering)
      4. Dense-pack std cells into a rectangular block
      5. Place macros outside the block, each on its natural side
    """
    num_cells = cell_features.shape[0]

    cell_edge_list = build_cell_level_edge_list(pin_features, edge_list)
    adjacency = build_adjacency_list(cell_edge_list, num_cells)

    macro_indices = [i for i in range(num_cells)
                     if cell_features[i, CellFeatureIdx.AREA] >= MIN_MACRO_AREA]
    std_indices = [i for i in range(num_cells)
                   if cell_features[i, CellFeatureIdx.AREA] < MIN_MACRO_AREA]

    score_matrix = compute_bfs_scores_from_macros(
        adjacency, macro_indices, num_cells
    )

    order = order_std_cells(score_matrix, std_indices)
    cell_features, bbox = pack_std_cells(order, std_indices, cell_features)
    cell_features = place_macros_around_block(
        score_matrix, macro_indices, std_indices, cell_features, bbox
    )

    virtual_macro = create_virtual_macro(bbox)

    return cell_features, virtual_macro

'''
if __name__ == "__main__":
    num_macros = 10
    num_std_cells = 1000
    cell_features, pin_features, edge_list = generate_placement_input(num_macros, num_std_cells)
    updated_cell_features, virtual_macro = initialize_placement(cell_features, pin_features, edge_list)

    plot_placement(cell_features, updated_cell_features, "cell_placement.png")
    metrics = calculate_normalized_metrics(updated_cell_features, pin_features, edge_list)
    print("Initial placement metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
'''