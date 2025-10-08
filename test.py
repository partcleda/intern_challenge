"""
Test Harness for VLSI Cell Placement Challenge
==============================================

This script runs the placement optimizer on 10 randomly generated netlists
of various sizes and reports metrics for leaderboard submission.

Usage:
    python test_placement.py

Metrics Reported:
    - Average Overlap: (num cells with overlaps / total num cells)
    - Average Wirelength: (total wirelength / num nets) / sqrt(total area)
      This normalization allows fair comparison across different design sizes.

Note: This test uses the default hyperparameters from train_placement() in
vb_playground.py. The challenge is to implement the overlap loss function,
not to tune hyperparameters.
"""

import time

import torch

# Import from the challenge file
from placement import (
    calculate_normalized_metrics,
    generate_placement_input,
    train_placement,
)


# Test case configurations: (test_id, num_macros, num_std_cells, seed)
TEST_CASES = [
    # Small designs
    (1, 2, 20, 1001),
    (2, 3, 25, 1002),
    (3, 2, 30, 1003),
    # Medium designs
    (4, 3, 50, 1004),
    (5, 4, 75, 1005),
    (6, 5, 100, 1006),
    # Large designs
    (7, 5, 150, 1007),
    (8, 7, 150, 1008),
    (9, 8, 200, 1009),
    (10, 10, 2000, 1010),
]


def run_placement_test(
    test_id,
    num_macros,
    num_std_cells,
    seed=None,
):
    """Run placement optimization on a single test case.

    Uses default hyperparameters from train_placement() function.

    Args:
        test_id: Test case identifier
        num_macros: Number of macro cells
        num_std_cells: Number of standard cells
        seed: Random seed for reproducibility

    Returns:
        Dictionary with test results and metrics
    """
    if seed:
        # Set seed for reproducibility
        torch.manual_seed(seed)

    # Generate netlist
    cell_features, pin_features, edge_list = generate_placement_input(
        num_macros, num_std_cells
    )

    # Initialize positions with random spread
    total_cells = cell_features.shape[0]
    total_area = cell_features[:, 0].sum().item()
    spread_radius = (total_area ** 0.5) * 0.6

    angles = torch.rand(total_cells) * 2 * 3.14159
    radii = torch.rand(total_cells) * spread_radius

    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)

    # Run optimization with default hyperparameters
    start_time = time.time()
    result = train_placement(
        cell_features,
        pin_features,
        edge_list,
        verbose=False,  # Suppress per-epoch output
    )
    elapsed_time = time.time() - start_time

    # Calculate final metrics using shared implementation
    final_cell_features = result["final_cell_features"]
    metrics = calculate_normalized_metrics(final_cell_features, pin_features, edge_list)

    return {
        "test_id": test_id,
        "num_macros": num_macros,
        "num_std_cells": num_std_cells,
        "total_cells": metrics["total_cells"],
        "num_nets": metrics["num_nets"],
        "seed": seed,
        "elapsed_time": elapsed_time,
        # Final metrics
        "num_cells_with_overlaps": metrics["num_cells_with_overlaps"],
        "overlap_ratio": metrics["overlap_ratio"],
        "normalized_wl": metrics["normalized_wl"],
        "core_area": metrics["core_area"],
        "core_utilization": metrics["core_utilization"],
    }


def run_all_tests():
    """Run all test cases and compute aggregate metrics.

    Uses default hyperparameters from train_placement() function.

    Returns:
        Dictionary with all test results and aggregate statistics
    """
    print("=" * 70)
    print("PLACEMENT CHALLENGE TEST SUITE")
    print("=" * 70)
    print(f"\nRunning {len(TEST_CASES)} test cases with various netlist sizes...")
    print("Using default hyperparameters from train_placement()")
    print()

    all_results = []

    for idx, (test_id, num_macros, num_std_cells, seed) in enumerate(TEST_CASES, 1):
        size_category = (
            "Small" if num_std_cells <= 30
            else "Medium" if num_std_cells <= 100
            else "Large"
        )

        print(f"Test {idx}/{len(TEST_CASES)}: {size_category} ({num_macros} macros, {num_std_cells} std cells)")
        print(f"  Seed: {seed}")

        # Run test
        result = run_placement_test(
            test_id,
            num_macros,
            num_std_cells,
            seed,
        )

        all_results.append(result)

        # Print summary
        status = "✓ PASS" if result["num_cells_with_overlaps"] == 0 else "✗ FAIL"
        print(f"  Overlap Ratio: {result['overlap_ratio']:.4f} ({result['num_cells_with_overlaps']}/{result['total_cells']} cells)")
        print(f"  Normalized WL: {result['normalized_wl']:.4f}")
        print(f"  Core Area: {result['core_area']:.2f}")
        print(f"  Core Utilization: {result['core_utilization']:.2f}")
        print(f"  Time: {result['elapsed_time']:.2f}s")
        print(f"  Status: {status}")
        print()

    # Compute aggregate statistics
    avg_overlap_ratio = sum(r["overlap_ratio"] for r in all_results) / len(all_results)
    avg_normalized_wl = sum(r["normalized_wl"] for r in all_results) / len(all_results)
    avg_core_utilization = sum(r["core_utilization"] for r in all_results) / len(all_results)
    total_time = sum(r["elapsed_time"] for r in all_results)

    # Print aggregate results
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Average Overlap: {avg_overlap_ratio:.4f}")
    print(f"Average Wirelength: {avg_normalized_wl:.4f}")
    print(f"Average Core Utilization: {avg_core_utilization:.2f}")
    print(f"Total Runtime: {total_time:.2f}s")
    print()

    return {
        "avg_overlap": avg_overlap_ratio,
        "avg_wirelength": avg_normalized_wl,
        "total_time": total_time,
    }


def main():
    """Main entry point for the test suite."""
    # Run all tests with default hyperparameters
    run_all_tests()


if __name__ == "__main__":
    main()
