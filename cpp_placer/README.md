# cpp_placer - a C++20 mixed-size placer for the par.tcl intern challenge

A standard-cell + macro placer written in C++20, called from `placement.py`
through a ~90 line ctypes shim. No dependencies beyond a C++20 compiler; the
library builds itself on first import.

## Run it

```bash
python run_first10.py        # tests 1-10, the leaderboard numbers
python run_first10.py 11 12  # extra credit
PARTCL_BUDGET=1.0 python run_first10.py   # seconds of search per design
PARTCL_SOLVER=torch python run_first10.py # the PyTorch reference path instead
```

`test.py` as shipped runs all 12 cases and averages over 12; the README asks for
the first 10. `run_first10.py` reuses `test.py`'s own `run_placement_test` and
`TEST_CASES` verbatim and only changes which slice is averaged.

## What the objective actually is

`wirelength_attraction_loss()` costs each edge as
`alpha * log(exp(|dx|/alpha) + exp(|dy|/alpha))` with `alpha = 0.1`. The
docstring calls this "a smooth approximation of Manhattan distance", but
`alpha*logsumexp(dx/alpha, dy/alpha)` approximates **max**, not the sum: it is
within `alpha*ln2 = 0.069` of `max(|dx|,|dy|)` everywhere. The scored objective
is Chebyshev (L-inf) wirelength. This solver optimises that function directly.

Second, pins sit at `cell_pos + pin_offset` with the offset drawn from
`[0,w] x [0,h]`, while the overlap check treats `cell_pos` as the cell *centre*.
A cell's pin cloud is therefore its body translated by `(+w/2, +h/2)`, and the
shift is bigger for bigger cells. The solver never assumes pins are centred, so
a small cell can park where its pin cloud lands inside a large neighbour's.

Neither of these is worth "fixing" in the harness: changing them would move the
metric and make the leaderboard incomparable. They are noted, not patched.

## Pipeline

1. **Global placement** - Adam on smoothed-L-inf wirelength plus a pairwise
   overlap-area penalty. The penalty multiplier is auto-scaled each iteration
   against the wirelength gradient norm, so one schedule works from 22 to
   100,000 cells with no per-design tuning. Neighbour search is a uniform grid
   for standard cells; macros are few and checked against everything. The step
   size decays geometrically, which is what stops the layout from bouncing
   instead of settling.
2. **Macro placement** - either legalised from the global placement, or
   shelf-packed (first-fit decreasing) into one edge band of the die. Macros
   have roughly 100x lower pin density per unit area than standard cells, so
   giving up the middle is usually right.
3. **Row legalization** - every standard cell has height exactly 1.0, so they go
   into a unit-pitch row grid bounded by the die, with macros as blocked
   intervals. Legality here is structural rather than hoped for: rows cannot
   overlap in y, and intervals within a row are kept disjoint by construction.
   So zero overlap does not depend on the penalty weight converging.
4. **Detailed placement** - each cell is relocated to the exact 1-D optimum of
   the real objective (the cost is convex and piecewise linear in x for fixed y,
   so the optimum sits on a breakpoint), then snapped to the nearest free slot.
   Plus equal-width swaps, which are legal by construction. Then the die is
   dropped and the same passes run unbounded: there is no fixed outline in this
   problem, the die is only a device to force a dense pack.
5. **Multi-start** - round 0 sweeps 13 die aspect ratios against 9 constructive
   macro arrangements; later rounds hill-climb macro offsets at the winning
   aspect. The die utilisation is found by bisection rather than guessed.

## Zero overlap is verified, not assumed

Before returning, positions are rounded to float32 (the dtype the harness stores
them in) and checked with a sweep-line implementation of
`calculate_cells_with_overlaps`'s exact predicate. If any cell overlaps, the
shim raises rather than returning a placement that would score a nonzero
overlap. The legality margin is 5e-3, about 80x the float32 resolution at these
coordinates, so rounding cannot open a gap.

## Files

- `src/partcl_place.cpp` - the whole solver, single translation unit.
- `build.sh` - `c++ -std=c++20 -O3 -shared -fPIC`. Nothing else.
- `__init__.py` - ctypes bridge, builds the library on first import.
