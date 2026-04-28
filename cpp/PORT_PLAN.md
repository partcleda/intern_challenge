# C++ Port Coordination Plan

## Summary

This file is the shared recovery plan for porting the Python placement flow from
`placement.py` and `test.py` into the C++ implementation under `cpp/`.

## Current Status

- The C++ project builds.
- `cpp/generation.cpp` contains a real implementation for synthetic placement
  input generation and initial cell placement.
- `cpp/metrics.cpp` now implements overlap metrics and normalized metrics to
  match the Python metric behavior.
- `cpp/tests/metrics_tests.cpp` covers the metrics implementation with a
  deterministic hand-authored placement, a generated-placement integration
  smoke test, loss parity checks, loss edge cases, an autograd smoke test, and
  focused training-loop and benchmark-runner checks.
- `cpp/tests/visualization_tests.cpp` covers SVG placement visualization output.
- `cpp/CMakeLists.txt` registers `placement_unit_tests` with CTest when
  `BUILD_TESTING` is enabled.
- `cpp/CMakeLists.txt` also exposes opt-in LLVM source coverage through
  `PLACEMENT_ENABLE_COVERAGE` and the `placement_coverage` target.
- `cpp/CMakeLists.txt` now prefers a PATH Python that can import `torch` when no
  repo-local `.venv` exists, avoiding macOS CMake selecting a newer Python
  without the required Torch package.
- `cpp/CMakePresets.json` includes a `coverage` preset that writes coverage
  build artifacts to `cpp/build-coverage`.
- `cpp/losses.cpp` now implements `wirelengthAttractionLoss`,
  `computePairwiseOverlapAreas`, and `overlapRepulsionLoss` to match the
  Python formulas in `placement.py`.
- `cpp/training.cpp` now implements the core `train_placement` loop: Adam over
  cell positions, weighted wirelength/overlap losses, gradient clipping,
  supported learning-rate schedulers, best-position tracking, and early-stop
  metadata.
- Tests and coverage were re-verified on 2026-04-26 after the graphing and
  notebook artifact work.
- `cpp/benchmark.cpp` now implements active benchmark case metadata, single-case
  execution using the configured device enum, serial multi-case execution,
  ordered results, aggregate averages, total elapsed time, and pass/fail counts.
- `cpp/main.cpp` now implements the `placement` binary CLI for single placement
  runs, ordered serial active benchmark runs, SVG visualization output, and
  notebook-friendly CSV/JSON artifact files.
- `cpp/visualization.cpp` now emits a dependency-free SVG visualization with
  side-by-side initial/final placements and overlap metrics.

## Source Of Truth

- `placement.py` is the source of truth for placement generation semantics,
  loss formulas, overlap metrics, normalized metrics, training behavior, and the
  single-run CLI flow.
- `test.py` is the source of truth for benchmark execution, per-case result
  fields, aggregate metrics, and pass/fail reporting.
- `benchmark_test_cases.py` is the source of truth for active benchmark cases.
- C++ code should preserve the public struct fields already declared in
  `cpp/include/placement/types.h` unless a later step explicitly approves an
  interface change.

## Working Protocol

After each implementation step, stop and ask for explicit permission before
editing code for the next step.

Use branch/worktree coordination for delegated work, and integrate branches into
`main` with fast-forward-only merges.

Permission checkpoints:

1. Write or update this plan file.
2. Ask permission before editing `cpp/metrics.cpp`.
3. Add unit coverage for features implemented so far before moving to the next
   porting area.
4. Continue this pattern for training, benchmark runner, and CLI work.

## Immediate Next Step

The current scoped roadmap is complete. Optional deferred items remain for a
later scope decision: SQLite loss history, profiling, and multiprocessing.

Recently verified commands from `cpp/`:

```sh
cmake --build --preset release --target placement_unit_tests
cmake --build --preset release --target placement
ctest --test-dir build -R placement_unit_tests --output-on-failure
cmake --build --preset coverage --target placement_coverage
./build/placement --num-macros 0 --num-std-cells 1 --num-epochs 0 --scheduler none --quiet --write-output-files --output-dir <tmpdir>
./build/placement --benchmark --num-epochs 0 --scheduler none --write-output-files --output-dir <tmpdir>
```

The latest coverage run reported:

- `benchmark.cpp`: 96.43% line coverage
- `generation.cpp`: 92.26% line coverage
- `losses.cpp`: 100.00% line coverage
- `metrics.cpp`: 93.89% line coverage
- `training.cpp`: 57.51% line coverage
- `visualization.cpp`: 88.53% line coverage
- Total: 83.63% line coverage

The implemented metrics behavior is:

- `calculateOverlapMetrics` computes overlap pair count, total overlap area,
  max single-pair overlap area, overlap percentage, cells involved in at least
  one overlap, and whether the placement has zero overlap.
- `calculateNormalizedMetrics` computes total cells, number of nets, number of
  cells with overlaps, overlap ratio, and normalized wirelength.
- The normalized wirelength formula matches `calculate_normalized_metrics` in
  `placement.py`: `(wirelength / num_nets) / sqrt(total_area)`, with zero
  returned when there are no nets or total area is zero.
- `wirelengthAttractionLoss` matches Python's average smooth Manhattan
  wirelength with `alpha = 0.1`.
- `computePairwiseOverlapAreas` returns the full pairwise overlap-area matrix,
  including diagonal self areas.
- `overlapRepulsionLoss` matches Python's
  `log1p(sum(upper_triangle_overlap_area)) * 200` formula.
- `trainPlacement` preserves the C++ result interface while porting the Python
  optimizer behavior that affects placement quality and early-stop selection.
- `activeBenchmarkCases` matches `benchmark_test_cases.py` active cases.
- `runBenchmarkCase` mirrors the per-case benchmark flow from `test.py`: seed,
  generate on the configured device enum, initialize, train with quiet logging,
  calculate normalized metrics, record elapsed time, and mark pass/fail from
  zero overlapping cells.
- `runBenchmarkCases` and `runActiveBenchmarkCases` provide ordered serial
  benchmark execution and aggregate average overlap/wirelength reporting.
- The `placement` CLI supports the shared training hyperparameters, device
  selection, single-run problem generation by size or test-case id, and active
  benchmark reporting.
- `plotPlacement` writes an SVG with initial/final placement panels, cell
  rectangles, grid lines, and overlap metrics.
- `--write-output-files` writes notebook-friendly artifacts:
  `placement_result_summary.csv`, `placement_result_summary.json`, and
  `placement_result.svg` for single runs; `placement_benchmark_cases.csv`,
  `placement_benchmark_summary.csv`, and `placement_benchmark_summary.json` for
  benchmark runs.
- SQLite loss history, profiling, and multiprocessing remain deferred.

## Port Roadmap

- [x] Metrics parity.
- [x] Loss function parity.
- [x] Training loop parity.
- [x] Benchmark/test runner parity.
- [x] CLI and output cleanup.
- [x] Add graphing parity.
- [x] Create output files that can be consumed by notebooks in the lab directory.

## Recovery Notes

If a future session resumes from scratch, inspect these files first:

- `cpp/PORT_PLAN.md`
- `cpp/metrics.cpp`
- `cpp/losses.cpp`
- `cpp/training.cpp`
- `cpp/benchmark.cpp`
- `cpp/main.cpp`
- `cpp/visualization.cpp`
- `cpp/include/placement/visualization.h`
- `cpp/include/placement/types.h`
- `cpp/tests/metrics_tests.cpp`
- `cpp/tests/visualization_tests.cpp`
- `cpp/cmake/RunCoverage.cmake`
- `placement.py`
- `test.py`

Useful read-only commands:

```sh
rg -n "calculate_overlap|calculate_normalized|wirelength|overlap_repulsion|train_placement" placement.py test.py cpp
sed -n '1,220p' cpp/metrics.cpp
sed -n '1,220p' cpp/tests/metrics_tests.cpp
sed -n '829,995p' placement.py
sed -n '320,535p' placement.py
cmake --build cpp/build --target placement_unit_tests --config Release
ctest --test-dir cpp/build -R placement_unit_tests --output-on-failure
cmake --preset coverage
cmake --build --preset coverage --target placement_coverage
cat cpp/build-coverage/coverage/placement_unit_tests.txt
```

Coverage outputs:

- Text summary: `cpp/build-coverage/coverage/placement_unit_tests.txt`
- HTML report: `cpp/build-coverage/coverage/html/index.html`
