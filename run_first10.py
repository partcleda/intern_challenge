"""Leaderboard runner: runs EXACTLY the first 10 test cases and reports the three
numbers the README asks for (Average Overlap, Average Wirelength, total Runtime).

`test.py` as shipped runs all 12 cases and averages over 12, but the README says
"Run the first 10 tests to evaluate your solution", and cases 11/12 are extra
credit. This runner reuses `test.py`'s own `run_placement_test` and `TEST_CASES`
verbatim so the numbers are produced by the challenge's own harness, and only
changes which slice of the list is averaged.

Usage:
    python run_first10.py              # tests 1-10 (leaderboard)
    python run_first10.py 11 12        # extra credit cases, by test id
"""

import json
import sys

from test import TEST_CASES, run_placement_test


def main():
    ids = [int(a) for a in sys.argv[1:]]
    cases = (
        [c for c in TEST_CASES if c[0] in ids] if ids else TEST_CASES[:10]
    )

    rows = []
    for test_id, num_macros, num_std_cells, seed in cases:
        r = run_placement_test(test_id, num_macros, num_std_cells, seed)
        rows.append(r)
        print(
            f"test {r['test_id']:>2}  cells={r['total_cells']:>6}  nets={r['num_nets']:>7}  "
            f"overlap={r['overlap_ratio']:.4f} ({r['num_cells_with_overlaps']}/{r['total_cells']})  "
            f"wl={r['normalized_wl']:.4f}  {r['elapsed_time']:.2f}s",
            flush=True,
        )

    n = len(rows)
    avg_overlap = sum(r["overlap_ratio"] for r in rows) / n
    avg_wl = sum(r["normalized_wl"] for r in rows) / n
    total_time = sum(r["elapsed_time"] for r in rows)

    print()
    print(f"Average Overlap:    {avg_overlap:.4f}")
    print(f"Average Wirelength: {avg_wl:.4f}")
    print(f"Total Runtime:      {total_time:.2f}s")

    with open("last_run.json", "w") as f:
        json.dump(
            {
                "avg_overlap": avg_overlap,
                "avg_wirelength": avg_wl,
                "total_runtime_s": total_time,
                "per_test": rows,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
