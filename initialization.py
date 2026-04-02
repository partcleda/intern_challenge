"""Cell placement initialization: random disk, eplace, replace-lite, spectral clustering, quadratic WL.

NOTE: This file is slightly more experimental than the placement.py file.
"""

from __future__ import annotations

import math

import torch


def apply_random_disk_initial(cell_features, random_spread_radius=None):
    """Sample (x, y) uniformly in a disk (polar method). Mutates cell_features in-place."""
    from placement import CellFeatureIdx

    n = cell_features.shape[0]
    if n <= 0:
        return
    if random_spread_radius is None:
        total_area = cell_features[:, CellFeatureIdx.AREA].sum().item()
        spread_r = (total_area ** 0.5) * 0.6
    else:
        spread_r = float(random_spread_radius)
    dev = cell_features.device
    dt = cell_features.dtype
    ang = torch.rand(n, device=dev, dtype=dt) * 2 * math.pi
    rad = torch.rand(n, device=dev, dtype=dt) * spread_r
    cell_features[:, CellFeatureIdx.X] = rad * torch.cos(ang)
    cell_features[:, CellFeatureIdx.Y] = rad * torch.sin(ang)


def eplace_lite_initial(cell_features, num_iters=100, step=0.2):
    """Spreading step inspired by global placement density force (overlap-driven, no gradients).

    Iteratively pushes cell centers apart along the inter-center direction, weighted by
    pairwise overlap area. Mutates x, y in-place.
    """
    from placement import CellFeatureIdx

    n = cell_features.shape[0]
    if n <= 1:
        return
    step = float(step)
    pos = cell_features[:, 2:4].clone().detach()
    w = cell_features[:, CellFeatureIdx.WIDTH]
    h = cell_features[:, CellFeatureIdx.HEIGHT]
    eye = torch.eye(n, device=pos.device, dtype=torch.bool)
    for _ in range(int(num_iters)):
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)
        dxabs = torch.abs(diff[:, :, 0])
        dyabs = torch.abs(diff[:, :, 1])
        min_sx = 0.5 * (w.unsqueeze(1) + w.unsqueeze(0))
        min_sy = 0.5 * (h.unsqueeze(1) + h.unsqueeze(0))
        ovx = torch.nn.functional.relu(min_sx - dxabs)
        ovy = torch.nn.functional.relu(min_sy - dyabs)
        overlap_area = ovx * ovy
        overlap_area = torch.where(eye, overlap_area.new_zeros(()), overlap_area)
        dir_x = diff[:, :, 0] / dist
        dir_y = diff[:, :, 1] / dist
        fx = (overlap_area * dir_x).sum(dim=1)
        fy = (overlap_area * dir_y).sum(dim=1)
        pos = pos + step * torch.stack([fx, fy], dim=1)
    cell_features[:, CellFeatureIdx.X] = pos[:, 0]
    cell_features[:, CellFeatureIdx.Y] = pos[:, 1]


def replace_lite_initial(
    cell_features, pin_features, edge_list, margin=1.2, row_gap=0.5
):
    """Coarse placement inspired by recursive partition: Fiedler order + row shelf packing.

    Builds a cell–cell adjacency from nets, uses the second Laplacian eigenvector to order
    cells for locality, then packs rows in a strip sized from total area. Mutates x, y.
    """
    from placement import CellFeatureIdx, PinFeatureIdx

    n = cell_features.shape[0]
    if n <= 1:
        return
    device = cell_features.device
    dtype = cell_features.dtype
    A = torch.zeros(n, n, device=device, dtype=dtype)
    if edge_list.numel() > 0:
        p_cell = pin_features[:, PinFeatureIdx.CELL_IDX].long()
        for k in range(edge_list.shape[0]):
            i = int(p_cell[int(edge_list[k, 0])].item())
            j = int(p_cell[int(edge_list[k, 1])].item())
            if i != j:
                A[i, j] += 1.0
                A[j, i] += 1.0
    d = A.sum(dim=1)
    L = torch.diag(d) - A
    if device.type == "mps":
        _, V = torch.linalg.eigh(L.cpu())
    else:
        _, V = torch.linalg.eigh(L)
    V = V.to(device)
    coord1d = V[:, 1] if n > 1 else V[:, 0]
    order = torch.argsort(coord1d)
    total_area = cell_features[:, CellFeatureIdx.AREA].sum().item()
    side = math.sqrt(max(total_area, 1e-12)) * float(margin)
    x_left = -0.5 * side
    x_right = 0.5 * side
    y = 0.0
    row_h = 0.0
    x_cursor = x_left
    pos = torch.zeros(n, 2, device=device, dtype=dtype)
    g = float(row_gap)
    for t in range(n):
        idx = int(order[t].item())
        wv = float(cell_features[idx, CellFeatureIdx.WIDTH].item())
        hv = float(cell_features[idx, CellFeatureIdx.HEIGHT].item())
        if x_cursor > x_left and x_cursor + wv > x_right:
            y -= row_h + g
            x_cursor = x_left
            row_h = 0.0
        center_x = x_cursor + wv * 0.5
        pos[idx, 0] = center_x
        pos[idx, 1] = y
        x_cursor += wv + g
        row_h = max(row_h, hv)
    pos = pos - pos.mean(dim=0, keepdim=True)
    cell_features[:, CellFeatureIdx.X] = pos[:, 0]
    cell_features[:, CellFeatureIdx.Y] = pos[:, 1]


def spectral_clustering_initial(
    cell_features, pin_features, edge_list, margin=1.2, row_gap=0.5
):
    """Simple spectral-clustering placement: cluster by graph, pack clusters nearby."""
    from placement import CellFeatureIdx, PinFeatureIdx

    n = int(cell_features.shape[0])
    if n <= 1:
        return
    device = cell_features.device
    dtype = cell_features.dtype
    A = torch.zeros(n, n, device=device, dtype=dtype)
    if edge_list.numel() > 0:
        p_cell = pin_features[:, PinFeatureIdx.CELL_IDX].long()
        for k in range(edge_list.shape[0]):
            i = int(p_cell[int(edge_list[k, 0])].item())
            j = int(p_cell[int(edge_list[k, 1])].item())
            if i != j:
                A[i, j] += 1.0
                A[j, i] += 1.0
    d = A.sum(dim=1)
    if float(d.sum()) <= 0.0:
        apply_random_disk_initial(cell_features, random_spread_radius=None)
        return
    L = torch.diag(d) - A

    if device.type == "mps":
        _, V = torch.linalg.eigh(L.cpu())
    else:
        _, V = torch.linalg.eigh(L)
    V = V.to(device)

    k_clusters = max(2, min(8, int(round(math.sqrt(n)))))
    emb_dim = min(max(1, k_clusters - 1), max(1, n - 1))
    X = V[:, 1 : 1 + emb_dim].clone()
    if n < k_clusters:
        k_clusters = n
    centers = X[torch.linspace(0, n - 1, steps=k_clusters).long()].clone()
    labels = torch.zeros(n, device=device, dtype=torch.long)
    for _ in range(8):
        dist2 = torch.cdist(X, centers, p=2) ** 2
        labels = torch.argmin(dist2, dim=1)
        for c in range(k_clusters):
            mask = labels == c
            if torch.any(mask):
                centers[c] = X[mask].mean(dim=0)
    total_area = float(cell_features[:, CellFeatureIdx.AREA].sum().item())
    side = math.sqrt(max(total_area, 1e-12)) * float(margin)
    cluster_r = 0.35 * side
    pos = torch.zeros(n, 2, device=device, dtype=dtype)
    g = float(row_gap)
    for c in range(k_clusters):
        idxs = torch.nonzero(labels == c, as_tuple=False).flatten()
        if idxs.numel() == 0:
            continue
        ang = 2.0 * math.pi * c / max(1, k_clusters)
        cx = cluster_r * math.cos(ang)
        cy = cluster_r * math.sin(ang)
        x_cursor = 0.0
        y_cursor = 0.0
        row_h = 0.0
        for idx in idxs.tolist():
            wv = float(cell_features[idx, CellFeatureIdx.WIDTH].item())
            hv = float(cell_features[idx, CellFeatureIdx.HEIGHT].item())
            if x_cursor > 0.0 and x_cursor + wv > 0.45 * side:
                y_cursor -= row_h + g
                x_cursor = 0.0
                row_h = 0.0
            pos[idx, 0] = cx + x_cursor + 0.5 * wv
            pos[idx, 1] = cy + y_cursor
            x_cursor += wv + g
            row_h = max(row_h, hv)
    pos = pos - pos.mean(dim=0, keepdim=True)
    cell_features[:, CellFeatureIdx.X] = pos[:, 0]
    cell_features[:, CellFeatureIdx.Y] = pos[:, 1]


def rectangle_hull_span(cell_features):
    """Max side length of axis-aligned bounding box of all cell rectangles."""
    from placement import CellFeatureIdx

    cx = cell_features[:, CellFeatureIdx.X]
    cy = cell_features[:, CellFeatureIdx.Y]
    w = cell_features[:, CellFeatureIdx.WIDTH]
    h = cell_features[:, CellFeatureIdx.HEIGHT]
    xmin = (cx - 0.5 * w).min()
    xmax = (cx + 0.5 * w).max()
    ymin = (cy - 0.5 * h).min()
    ymax = (cy + 0.5 * h).max()
    return torch.maximum(xmax - xmin, ymax - ymin)


def normalize_rectangle_hull_to_core(cell_features, side, pad_frac=0.92):
    """Re-center and uniformly scale **centers** so the layout fits ~``side`` (plus padding).

    Width/height are fixed; scaling only ``(x,y)`` does **not** shrink rectangles, so we
    size the **center** spread to leave room for ``max(width,height)`` inside
    ``side*pad_frac``. Without this, a hull computed from corners can never match a pure
    center-scale target.
    """
    from placement import CellFeatureIdx

    n = cell_features.shape[0]
    if n <= 0:
        return
    cx = cell_features[:, CellFeatureIdx.X]
    cy = cell_features[:, CellFeatureIdx.Y]
    w = cell_features[:, CellFeatureIdx.WIDTH]
    h = cell_features[:, CellFeatureIdx.HEIGHT]
    max_wh = torch.maximum(w.max(), h.max())
    budget = float(side) * float(pad_frac) - float(max_wh.detach().clamp_min(1e-6).item())
    target = max(budget, float(side) * 0.35)
    xmin = cx.min()
    xmax = cx.max()
    ymin = cy.min()
    ymax = cy.max()
    mid_x = 0.5 * (xmin + xmax)
    mid_y = 0.5 * (ymin + ymax)
    span = torch.maximum(xmax - xmin, ymax - ymin).clamp_min(
        torch.as_tensor(
            torch.finfo(cell_features.dtype).eps,
            device=cx.device,
            dtype=cx.dtype,
        )
    )
    s = torch.as_tensor(target, device=cx.device, dtype=cx.dtype) / span
    cell_features[:, CellFeatureIdx.X] = (cx - mid_x) * s
    cell_features[:, CellFeatureIdx.Y] = (cy - mid_y) * s


def clamp_centers_to_core_budget(cell_features, side, pad_frac=0.92):
    """Center the rectangle hull at the origin; scale **centers** uniformly if hull is too wide.

    Uses bisection so the true rectangle AABB (fixed w,h) fits within ``side*pad_frac``,
    without the bug where scaling centers was sized for center-span only.
    """
    from placement import CellFeatureIdx

    cx = cell_features[:, CellFeatureIdx.X]
    cy = cell_features[:, CellFeatureIdx.Y]
    w = cell_features[:, CellFeatureIdx.WIDTH]
    h = cell_features[:, CellFeatureIdx.HEIGHT]
    xmin = (cx - 0.5 * w).min()
    xmax = (cx + 0.5 * w).max()
    ymin = (cy - 0.5 * h).min()
    ymax = (cy + 0.5 * h).max()
    mid_x = 0.5 * (xmin + xmax)
    mid_y = 0.5 * (ymin + ymax)
    cell_features[:, CellFeatureIdx.X] = cx - mid_x
    cell_features[:, CellFeatureIdx.Y] = cy - mid_y
    max_wh = float(torch.maximum(w.max(), h.max()).item())
    target = max(float(side) * float(pad_frac), max_wh * 1.02)
    span = rectangle_hull_span(cell_features)
    if float(span) <= target:
        return
    cx0 = cell_features[:, CellFeatureIdx.X].clone()
    cy0 = cell_features[:, CellFeatureIdx.Y].clone()

    def span_if(scale):
        s = float(scale)
        cell_features[:, CellFeatureIdx.X] = cx0 * s
        cell_features[:, CellFeatureIdx.Y] = cy0 * s
        return float(rectangle_hull_span(cell_features))

    lo, hi = 1e-7, 1.0
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if span_if(mid) <= target:
            lo = mid
        else:
            hi = mid
    cell_features[:, CellFeatureIdx.X] = cx0 * lo
    cell_features[:, CellFeatureIdx.Y] = cy0 * lo


def analytic_quadratic_wl_initial(
    cell_features, pin_features, edge_list, margin=1.2
):
    """Wirelength-driven quadratic (spectral) placement: 2D Laplacian eigenvectors.

    Eigenvectors approximate minimum quadratic wirelength for **point** nodes. Mapping
    uses **isotropic** scaling (same x/y scale) plus **winsorization** so outliers in
    v₁/v₂ do not compress the bulk into one corner. A moderate **eplace_lite** pass
    separates overlapping rectangles; **hull normalization** fits centers into the core
    budget; **clamp** (rectangle AABB, bisection on center scale) runs after eplace so
    fixed widths/heights cannot leave a few macros at ±1e3 while others sit at 0.
    No nets → random disk fallback.
    """
    from placement import CellFeatureIdx, PinFeatureIdx

    n = cell_features.shape[0]
    if n <= 1:
        return
    device = cell_features.device
    dtype = cell_features.dtype
    A = torch.zeros(n, n, device=device, dtype=dtype)
    if edge_list.numel() > 0:
        p_cell = pin_features[:, PinFeatureIdx.CELL_IDX].long()
        for k in range(edge_list.shape[0]):
            i = int(p_cell[int(edge_list[k, 0])].item())
            j = int(p_cell[int(edge_list[k, 1])].item())
            if i != j:
                A[i, j] += 1.0
                A[j, i] += 1.0
    if float(A.sum()) < 1e-18:
        apply_random_disk_initial(cell_features, random_spread_radius=None)
        return
    d = A.sum(dim=1)
    L = torch.diag(d) - A
    if device.type == "mps":
        _, V = torch.linalg.eigh(L.cpu())
    else:
        _, V = torch.linalg.eigh(L)
    V = V.to(device)
    if n == 2:
        vx = V[:, 1]
        vy = torch.zeros(n, device=device, dtype=dtype)
    else:
        vx = V[:, 1].clone()
        vy = V[:, 2].clone()
    if n >= 8:
        qs = torch.tensor([0.05, 0.95], device=device, dtype=dtype)
        qx = torch.quantile(vx, qs)
        qy = torch.quantile(vy, qs)
        vx = vx.clamp(qx[0], qx[1])
        vy = vy.clamp(qy[0], qy[1])
    vx = vx - vx.mean()
    vy = vy - vy.mean()
    radii = torch.sqrt(vx * vx + vy * vy)
    eps = torch.tensor(torch.finfo(dtype).eps, device=device, dtype=dtype)
    r_lo = torch.quantile(radii, torch.tensor(0.5, device=device, dtype=dtype)).clamp_min(
        eps
    )
    r_hi = torch.quantile(radii, torch.tensor(0.92, device=device, dtype=dtype))
    scale_denom = r_hi.clamp_min(r_lo * 0.25)
    total_area = cell_features[:, CellFeatureIdx.AREA].sum().item()
    side = math.sqrt(max(total_area, 1e-12)) * float(margin)
    half = side * 0.5
    scale = (half * 0.92) / scale_denom
    x = vx * scale
    y = vy * scale
    cell_features[:, CellFeatureIdx.X] = x
    cell_features[:, CellFeatureIdx.Y] = y
    normalize_rectangle_hull_to_core(cell_features, side, pad_frac=0.92)
    spread_iters = min(60, max(32, 15 + n // 3))
    eplace_lite_initial(cell_features, num_iters=spread_iters, step=0.08)
    clamp_centers_to_core_budget(cell_features, side, pad_frac=0.92)


def gaussian_jitter_cell_centers(cell_features, std):
    """Add independent Gaussian noise to each cell center (x, y). No-op if std <= 0."""
    from placement import CellFeatureIdx

    n = cell_features.shape[0]
    if n <= 0:
        return
    s = float(std)
    if s <= 0.0:
        return
    dev = cell_features.device
    dt = cell_features.dtype
    noise = torch.randn(n, 2, device=dev, dtype=dt) * s
    cell_features[:, CellFeatureIdx.X] = cell_features[:, CellFeatureIdx.X] + noise[:, 0]
    cell_features[:, CellFeatureIdx.Y] = cell_features[:, CellFeatureIdx.Y] + noise[:, 1]


def apply_initial_placement(
    mode,
    cell_features,
    pin_features,
    edge_list,
    random_spread_radius=None,
    eplace_lite_iters=100,
    eplace_lite_step=0.2,
    replace_lite_margin=1.2,
    replace_lite_row_gap=0.5,
    replace_lite_post_noise_std=2.0,
):
    mode = (mode or "preserve").lower().replace("-", "_")
    if mode in ("preserve", "none", "as_given"):
        return
    if mode in ("random", "disk"):
        apply_random_disk_initial(cell_features, random_spread_radius)
        return
    if mode in ("eplace", "eplace_lite", "eplace-lite"):
        eplace_lite_initial(
            cell_features, num_iters=eplace_lite_iters, step=eplace_lite_step
        )
        return
    if mode in (
        "quadratic_wl",
        "quadratic_wirelength",
        "analytic_wl",
        "analytic",
        "spectral_wl",
        "spectral_quadratic",
        "wl_quadratic",
    ):
        analytic_quadratic_wl_initial(
            cell_features,
            pin_features,
            edge_list,
            margin=replace_lite_margin,
        )
        return
    if mode in ("spectral_clustering", "cluster", "spectral_cluster"):
        spectral_clustering_initial(
            cell_features,
            pin_features,
            edge_list,
            margin=replace_lite_margin,
            row_gap=replace_lite_row_gap,
        )
        return
    if mode in (
        "random_then_quadratic_wl",
        "random_then_analytic_wl",
        "random_quadratic_wl",
    ):
        apply_random_disk_initial(cell_features, random_spread_radius)
        analytic_quadratic_wl_initial(
            cell_features,
            pin_features,
            edge_list,
            margin=replace_lite_margin,
        )
        return
    if mode in ("replace", "replace_lite", "replace-lite", "recursive_partition_lite"):
        replace_lite_initial(
            cell_features,
            pin_features,
            edge_list,
            margin=replace_lite_margin,
            row_gap=replace_lite_row_gap,
        )
        return
    if mode in (
        "random_then_replace_lite",
        "random_replace_lite",
        "rand_replace_lite",
        "random_then_replacelite",
    ):
        apply_random_disk_initial(cell_features, random_spread_radius)
        replace_lite_initial(
            cell_features,
            pin_features,
            edge_list,
            margin=replace_lite_margin,
            row_gap=replace_lite_row_gap,
        )
        return
    if mode in (
        "replace_lite_noisy",
        "replace_lite_gaussian",
        "replacelite_noisy",
    ):
        replace_lite_initial(
            cell_features,
            pin_features,
            edge_list,
            margin=replace_lite_margin,
            row_gap=replace_lite_row_gap,
        )
        gaussian_jitter_cell_centers(cell_features, replace_lite_post_noise_std)
        return
    if mode in (
        "random_then_replace_lite_noisy",
        "random_then_replace_lite_gaussian",
        "random_replace_lite_noisy",
        "rand_replace_lite_noisy",
        "random_then_replacelite_noisy",
    ):
        apply_random_disk_initial(cell_features, random_spread_radius)
        replace_lite_initial(
            cell_features,
            pin_features,
            edge_list,
            margin=replace_lite_margin,
            row_gap=replace_lite_row_gap,
        )
        gaussian_jitter_cell_centers(cell_features, replace_lite_post_noise_std)
        return
    raise ValueError(
        f"Unknown initial_placement {mode!r}. Use: preserve, random, eplace_lite, "
        "quadratic_wl, spectral_clustering, random_then_quadratic_wl, "
        "replace_lite, replace_lite_noisy, random_then_replace_lite, "
        "random_then_replace_lite_noisy."
    )
