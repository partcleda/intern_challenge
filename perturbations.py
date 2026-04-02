"""Non-gradient placement perturbations: greedy swaps and position noise kicks."""

from __future__ import annotations

import torch


def combined_placement_objective_value(cell_features, pin_features, edge_list):
    """Scalar objective for greedy swaps (wirelength-only scoring by default)."""
    from placement import wirelength_attraction_loss

    wl = wirelength_attraction_loss(
        cell_features, pin_features, edge_list, use_smooth_manhattan=False
    )
    return wl.item()


def greedy_cell_position_swaps(
    cell_features_template,
    positions_2d,
    pin_features,
    edge_list,
    max_rounds=3,
    max_pairs_per_round=None,
    generator=None,
    max_swap_cell_area=50.0,
):
    """Try swapping cell centers (x,y); keep swap if combined objective decreases.

    If ``max_swap_cell_area`` is set and **positive**, only cells whose feature
    ``AREA`` (column 0) is **at most** that value participate in swaps—so any pair
    involving a larger cell is never tried. ``None`` or ``<= 0`` turns this filter
    off (all cells may swap).

    Eligibility is computed with plain Python floats on CPU so device / ``nonzero``
    quirks do not empty the candidate set mid-training.

    Mutates ``positions_2d`` in-place (and uses a work clone of ``cell_features_template``).
    Returns the number of accepted swaps across all rounds.
    """
    from placement import CellFeatureIdx

    n = int(positions_2d.shape[0])
    if n < 2:
        return 0
    area_col = (
        cell_features_template[:, CellFeatureIdx.AREA]
        .detach()
        .float()
        .view(-1)
        .cpu()
        .tolist()
    )
    if max_swap_cell_area is None:
        swap_idx = list(range(n))
    else:
        mx_limit = float(max_swap_cell_area)
        if mx_limit <= 0.0:
            swap_idx = list(range(n))
        else:
            swap_idx = [i for i in range(n) if float(area_col[i]) <= mx_limit]
    if len(swap_idx) < 2:
        return 0
    pairs = [
        (i, j) for a, i in enumerate(swap_idx) for j in swap_idx[a + 1 :]
    ]
    if generator is not None:
        perm = torch.randperm(len(pairs), generator=generator, device="cpu").tolist()
        pairs = [pairs[k] for k in perm]
    else:
        perm = torch.randperm(len(pairs), device="cpu").tolist()
        pairs = [pairs[k] for k in perm]
    if max_pairs_per_round is not None and len(pairs) > int(max_pairs_per_round):
        pairs = pairs[: int(max_pairs_per_round)]

    total_accepted = 0

    for _ in range(int(max_rounds)):
        round_accepted = 0
        cf_work = cell_features_template.clone()
        cf_work[:, CellFeatureIdx.X] = positions_2d[:, 0].detach()
        cf_work[:, CellFeatureIdx.Y] = positions_2d[:, 1].detach()

        for i, j in pairs:
            loss_before = combined_placement_objective_value(
                cf_work, pin_features, edge_list
            )
            ti = cf_work[i, 2:4].clone()
            tj = cf_work[j, 2:4].clone()
            cf_work[i, 2:4] = tj
            cf_work[j, 2:4] = ti

            loss_after = combined_placement_objective_value(
                cf_work, pin_features, edge_list
            )

            if loss_after >= loss_before:
                cf_work[i, 2:4] = ti
                cf_work[j, 2:4] = tj
            else:
                round_accepted += 1

        positions_2d[:, 0] = cf_work[:, CellFeatureIdx.X].to(positions_2d.dtype)
        positions_2d[:, 1] = cf_work[:, CellFeatureIdx.Y].to(positions_2d.dtype)
        total_accepted += round_accepted
        if round_accepted == 0:
            break

    return total_accepted


def maybe_apply_position_noise_kick(
    cell_positions,
    epoch,
    *,
    enabled,
    interval,
    kick_std,
    kick_until_epoch,
    swap_gen,
    verbose,
):
    """Add in-place Gaussian noise to centers when the configured interval/epoch gates pass."""
    if (
        not enabled
        or interval is None
        or int(interval) <= 0
        or (epoch + 1) % int(interval) != 0
        or (epoch + 1) > int(kick_until_epoch)
        or float(kick_std) <= 0.0
    ):
        return
    shape = cell_positions.shape
    if swap_gen is not None:
        noise = torch.randn(
            shape,
            generator=swap_gen,
            dtype=cell_positions.dtype,
        ).to(cell_positions.device)
    else:
        noise = torch.randn(
            shape,
            device=cell_positions.device,
            dtype=cell_positions.dtype,
        )
    noise = noise * float(kick_std)
    with torch.no_grad():
        cell_positions.add_(noise)
    if verbose:
        print(
            f"Epoch {epoch}: position noise kick "
            f"(std={float(kick_std):g}) applied."
        )


def test_greedy_cell_position_swaps():
    """Two unit-square cells, one edge. Cell 0 is far from its pin partner; swapping
    should reduce wirelength and be accepted."""
    from placement import CellFeatureIdx, PinFeatureIdx

    # Cell 0 at (10, 0), Cell 1 at (0, 0).  Both are 1×1, area=1.
    cell_features = torch.zeros(2, 6)
    cell_features[0, CellFeatureIdx.AREA] = 1.0
    cell_features[1, CellFeatureIdx.AREA] = 1.0
    cell_features[0, CellFeatureIdx.NUM_PINS] = 1.0
    cell_features[1, CellFeatureIdx.NUM_PINS] = 1.0
    cell_features[0, CellFeatureIdx.X] = 10.0
    cell_features[0, CellFeatureIdx.Y] = 0.0
    cell_features[1, CellFeatureIdx.X] = 0.0
    cell_features[1, CellFeatureIdx.Y] = 0.0
    cell_features[0, CellFeatureIdx.WIDTH] = 1.0
    cell_features[0, CellFeatureIdx.HEIGHT] = 1.0
    cell_features[1, CellFeatureIdx.WIDTH] = 1.0
    cell_features[1, CellFeatureIdx.HEIGHT] = 1.0

    # One pin per cell, centred.  One edge connecting them.
    pin_features = torch.zeros(2, 7)
    pin_features[0, PinFeatureIdx.CELL_IDX] = 0
    pin_features[0, PinFeatureIdx.PIN_X] = 0.5
    pin_features[0, PinFeatureIdx.PIN_Y] = 0.5
    pin_features[1, PinFeatureIdx.CELL_IDX] = 1
    pin_features[1, PinFeatureIdx.PIN_X] = 0.5
    pin_features[1, PinFeatureIdx.PIN_Y] = 0.5
    edge_list = torch.tensor([[0, 1]], dtype=torch.long)

    positions = cell_features[:, 2:4].clone()
    wl_before = combined_placement_objective_value(cell_features, pin_features, edge_list)

    n_accepted = greedy_cell_position_swaps(
        cell_features,
        positions,
        pin_features,
        edge_list,
        max_rounds=1,
        max_swap_cell_area=None,
    )

    # Apply swapped positions back for scoring.
    cell_features[:, 2:4] = positions
    wl_after = combined_placement_objective_value(cell_features, pin_features, edge_list)

    # With only two identical cells, a swap has no effect on wirelength (symmetric),
    # so 0 accepted swaps is expected and WL should be unchanged.
    assert n_accepted == 0, f"Expected 0 accepted swaps, got {n_accepted}"
    assert abs(wl_after - wl_before) < 1e-6, (
        f"WL should be unchanged for symmetric swap: before={wl_before}, after={wl_after}"
    )

    # --- asymmetric case: 3 cells, swap should help ---
    # Cell 0 at (100, 0), Cell 1 at (1, 0), Cell 2 at (0, 0).
    # Edge between cell 0's pin and cell 2's pin ⇒ WL is large.
    # After swapping cell 0 ↔ cell 1, cell-with-the-edge moves closer.
    cf3 = torch.zeros(3, 6)
    for c in range(3):
        cf3[c, CellFeatureIdx.AREA] = 1.0
        cf3[c, CellFeatureIdx.NUM_PINS] = 1.0
        cf3[c, CellFeatureIdx.WIDTH] = 1.0
        cf3[c, CellFeatureIdx.HEIGHT] = 1.0
    cf3[0, CellFeatureIdx.X] = 100.0
    cf3[1, CellFeatureIdx.X] = 1.0
    cf3[2, CellFeatureIdx.X] = 0.0

    pf3 = torch.zeros(3, 7)
    for p in range(3):
        pf3[p, PinFeatureIdx.CELL_IDX] = p
        pf3[p, PinFeatureIdx.PIN_X] = 0.5
        pf3[p, PinFeatureIdx.PIN_Y] = 0.5
    el3 = torch.tensor([[0, 2]], dtype=torch.long)

    pos3 = cf3[:, 2:4].clone()
    wl3_before = combined_placement_objective_value(cf3, pf3, el3)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    n3 = greedy_cell_position_swaps(
        cf3, pos3, pf3, el3, max_rounds=3, max_swap_cell_area=None, generator=gen,
    )
    cf3[:, 2:4] = pos3
    wl3_after = combined_placement_objective_value(cf3, pf3, el3)

    assert n3 > 0, f"Expected at least 1 accepted swap, got {n3}"
    assert wl3_after < wl3_before, (
        f"WL should decrease: before={wl3_before}, after={wl3_after}"
    )
    print(
        f"test_greedy_cell_position_swaps PASSED  "
        f"(symmetric: accepted={n_accepted}, WL unchanged; "
        f"asymmetric: accepted={n3}, WL {wl3_before:.4f} → {wl3_after:.4f})"
    )


if __name__ == "__main__":
    test_greedy_cell_position_swaps()
