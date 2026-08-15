"""How much of the wirelength metric is actually reachable by a placer?

Edges are generated between *pins*, and a pin's absolute position is
`cell_pos + pin_offset`. When both endpoints of an edge sit on the same cell,
the term

    alpha * log( exp(|dx|/alpha) + exp(|dy|/alpha) )

depends only on the two pin offsets, so it is identical for every legal
placement. That fraction of `normalized_wl` is a floor: no solver, however good,
can go below it. This script measures the floor per test case so the leaderboard
numbers can be read against it.
"""

import math

import torch

from test import TEST_CASES
from placement import generate_placement_input

ALPHA = 0.1


def edge_cost(dx, dy):
    dx, dy = abs(dx), abs(dy)
    hi, lo = max(dx, dy), min(dx, dy)
    return hi + ALPHA * math.log1p(math.exp(-(hi - lo) / ALPHA))


print(f"{'test':>4} {'cells':>6} {'edges':>7} {'intra%':>7} {'floor':>8}")
floors = []
for test_id, num_macros, num_std_cells, seed in TEST_CASES[:10]:
    torch.manual_seed(seed)
    cf, pf, el = generate_placement_input(num_macros, num_std_cells)
    area = cf[:, 0].sum().item()
    pin_cell = pf[:, 0].long().numpy()
    ox = pf[:, 1].numpy()
    oy = pf[:, 2].numpy()
    e = el.numpy()

    const_sum = 0.0
    intra = 0
    for a, b in e:
        if pin_cell[a] == pin_cell[b]:
            const_sum += edge_cost(float(ox[a] - ox[b]), float(oy[a] - oy[b]))
            intra += 1
    floor = (const_sum / len(e)) / math.sqrt(area)
    floors.append(floor)
    print(
        f"{test_id:>4} {cf.shape[0]:>6} {len(e):>7} "
        f"{100.0 * intra / len(e):>6.1f}% {floor:>8.4f}"
    )

print(f"\nMean irreducible floor over tests 1-10: {sum(floors) / len(floors):.4f}")
