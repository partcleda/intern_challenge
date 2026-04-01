"""Analytical tools: minimum-wirelength LP and adjacency matrix statistics."""

from __future__ import annotations


def calculate_min_possible_normalized_wl(cell_features, pin_features, edge_list):
    """Compute the exact minimum normalized WL for this instance via linear programming.

    This solves the globally optimal unconstrained (overlap-ignored) placement for
    Manhattan edge cost:
        minimize Σ_e (|dx_e| + |dy_e|),
    where each edge connects two pins and each pin location is cell center plus fixed pin
    offset. Because the objective is convex and piecewise-linear in cell centers, this is
    formulated as an LP and solved to global optimality.

    NOTE: This is the true minimum for the LP/Manhattan formulation (overlap ignored).

    Returns:
        Dict with exact minimum totals and normalized metric.
    """
    import numpy as np
    from scipy.optimize import linprog

    from placement import CellFeatureIdx, PinFeatureIdx

    N = int(cell_features.shape[0])
    E = int(edge_list.shape[0])
    if N == 0 or E == 0:
        return {
            "min_total_wirelength": 0.0,
            "min_normalized_wl": 0.0,
            "num_nets": E,
            "solver_status": "trivial",
        }

    cell_idx = pin_features[:, PinFeatureIdx.CELL_IDX].long().cpu().numpy()
    pin_x = pin_features[:, PinFeatureIdx.PIN_X].cpu().numpy()
    pin_y = pin_features[:, PinFeatureIdx.PIN_Y].cpu().numpy()
    src = edge_list[:, 0].long().cpu().numpy()
    tgt = edge_list[:, 1].long().cpu().numpy()

    c_x = pin_x[tgt] - pin_x[src]
    c_y = pin_y[tgt] - pin_y[src]
    i_idx = cell_idx[src]
    j_idx = cell_idx[tgt]

    def _solve_axis(c_axis):
        V = N + E
        c_obj = np.zeros(V, dtype=np.float64)
        c_obj[N:] = 1.0

        A_rows = []
        b_vals = []
        for e in range(E):
            i = int(i_idx[e])
            j = int(j_idx[e])
            ce = float(c_axis[e])

            row1 = np.zeros(V, dtype=np.float64)
            row1[i] = 1.0
            row1[j] = -1.0
            row1[N + e] = -1.0
            A_rows.append(row1)
            b_vals.append(ce)

            row2 = np.zeros(V, dtype=np.float64)
            row2[i] = -1.0
            row2[j] = 1.0
            row2[N + e] = -1.0
            A_rows.append(row2)
            b_vals.append(-ce)

        A_ub = np.vstack(A_rows)
        b_ub = np.array(b_vals, dtype=np.float64)

        A_eq = np.zeros((1, V), dtype=np.float64)
        A_eq[0, 0] = 1.0
        b_eq = np.array([0.0], dtype=np.float64)

        bounds = [(None, None)] * N + [(0.0, None)] * E
        res = linprog(
            c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            raise RuntimeError(f"LP solve failed: {res.message}")
        return float(res.fun), res.status

    min_x, status_x = _solve_axis(c_x)
    min_y, status_y = _solve_axis(c_y)
    min_total = min_x + min_y

    total_area = float(cell_features[:, CellFeatureIdx.AREA].sum().item())
    min_norm = (min_total / E) / (total_area ** 0.5) if total_area > 0 else 0.0
    return {
        "min_total_wirelength": float(min_total),
        "min_normalized_wl": float(min_norm),
        "num_nets": E,
        "solver_status": f"x={status_x}, y={status_y}",
    }


def print_adjacency_matrix_and_stats(
    cell_features,
    pin_features,
    edge_list,
    max_matrix_size=80,
):
    """Print cell-level adjacency matrix and key connectivity statistics.

    Adjacency is derived from pin-level edges by mapping each pin to its owning cell.
    Multiple pin edges between the same pair of cells are counted as weighted adjacency.
    """
    import numpy as np

    from placement import PinFeatureIdx

    n_cells = int(cell_features.shape[0])
    if n_cells <= 0:
        print("Adjacency report: no cells.")
        return {
            "num_cells": 0,
            "num_undirected_edges": 0,
            "density": 0.0,
        }

    cell_idx = pin_features[:, PinFeatureIdx.CELL_IDX].long().cpu().numpy()
    if edge_list.shape[0] > 0:
        src_p = edge_list[:, 0].long().cpu().numpy()
        dst_p = edge_list[:, 1].long().cpu().numpy()
    else:
        src_p = np.array([], dtype=int)
        dst_p = np.array([], dtype=int)

    adj_w = np.zeros((n_cells, n_cells), dtype=np.int64)
    for s, t in zip(src_p, dst_p):
        i = int(cell_idx[s])
        j = int(cell_idx[t])
        if i == j:
            continue
        adj_w[i, j] += 1
        adj_w[j, i] += 1

    adj_bin = (adj_w > 0).astype(np.int64)
    np.fill_diagonal(adj_bin, 0)

    upper = np.triu(adj_bin, k=1)
    num_edges = int(upper.sum())
    max_edges = n_cells * (n_cells - 1) // 2
    density = (num_edges / max_edges) if max_edges > 0 else 0.0

    degrees = adj_bin.sum(axis=1)
    isolated = np.where(degrees == 0)[0].tolist()
    avg_deg = float(degrees.mean()) if n_cells > 0 else 0.0
    min_deg = int(degrees.min()) if n_cells > 0 else 0
    max_deg = int(degrees.max()) if n_cells > 0 else 0

    visited = np.zeros(n_cells, dtype=bool)
    component_sizes = []
    for start in range(n_cells):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            nbrs = np.where(adj_bin[u] > 0)[0]
            for v in nbrs:
                if not visited[v]:
                    visited[v] = True
                    stack.append(int(v))
        component_sizes.append(size)

    upper_w = np.triu(adj_w, k=1)
    nz_w = upper_w[upper_w > 0]
    avg_weight = float(nz_w.mean()) if nz_w.size > 0 else 0.0
    max_weight = int(nz_w.max()) if nz_w.size > 0 else 0

    print("\n" + "=" * 70)
    print("ADJACENCY REPORT (cell-level)")
    print("=" * 70)
    print(f"Cells: {n_cells}")
    print(f"Undirected edges: {num_edges}/{max_edges}")
    print(f"Density: {density:.4f}")
    print(f"Degree (min/avg/max): {min_deg}/{avg_deg:.2f}/{max_deg}")
    print(f"Connected components: {len(component_sizes)}")
    print(f"Largest component size: {max(component_sizes) if component_sizes else 0}")
    print(f"Isolated cells: {len(isolated)}")
    print(f"Weighted edge multiplicity (avg/max): {avg_weight:.2f}/{max_weight}")

    if n_cells <= int(max_matrix_size):
        print("\nBinary adjacency matrix (0/1):")
        for r in range(n_cells):
            print(" ".join(str(int(v)) for v in adj_bin[r]))
    else:
        print(
            f"\nBinary adjacency matrix skipped (size {n_cells} > max_matrix_size={int(max_matrix_size)})."
        )

    return {
        "num_cells": n_cells,
        "num_undirected_edges": num_edges,
        "density": float(density),
        "degree_min": min_deg,
        "degree_avg": avg_deg,
        "degree_max": max_deg,
        "num_components": len(component_sizes),
        "largest_component_size": int(max(component_sizes) if component_sizes else 0),
        "num_isolated": len(isolated),
        "isolated_cells": isolated,
        "weighted_edge_avg": avg_weight,
        "weighted_edge_max": max_weight,
    }
