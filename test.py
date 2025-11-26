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

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Test case configurations: (test_id, num_macros, num_std_cells, seed)
TEST_CASES = [
    # Small designs
    # (1, 2, 20, 1001),
    # (2, 3, 25, 1002),
    # (3, 2, 30, 1003),
    # # Medium designs
    # (4, 3, 50, 1004),
    # (5, 4, 75, 1005),
    # (6, 5, 100, 1006),
    # # Large designs
    # (7, 5, 150, 1007),
    # (8, 7, 150, 1008),
    # (9, 8, 200, 1009),
    # (10, 10, 2000, 1010),
    # Realistic designs
    (11, 10, 10000, 1011),
    # (12, 10, 100000, 1012),
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
    
    # Move all tensors to GPU if available
    cell_features = cell_features.to(device)
    pin_features = pin_features.to(device)
    edge_list = edge_list.to(device)

    # Initialize positions with random spread
    # For larger problems, use larger initial spread to reduce initial overlaps
    total_cells = cell_features.shape[0]
    total_area = cell_features[:, 0].sum().item()
    # Scale spread radius with problem size: larger problems need more space
    # Base spread: sqrt(area) * 0.6, scaled by log10(N/50) for large problems
    import math
    base_spread = (total_area ** 0.5) * 0.6
    spread_scale = 1.0 + 0.2 * math.log10(max(total_cells / 50.0, 1.0))
    spread_radius = base_spread * spread_scale

    # Create random tensors on the same device
    angles = torch.rand(total_cells, device=device) * 2 * 3.14159
    radii = torch.rand(total_cells, device=device) * spread_radius

    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)

    # Run optimization with default hyperparameters
    # Time measurement with GPU synchronization
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure GPU is ready
    start_time = time.time()
    
    result = train_placement(
        cell_features,
        pin_features,
        edge_list,
        verbose=True,  # Suppress per-epoch output
    )
    
    # Measure elapsed time with GPU synchronization
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for GPU operations to complete
    elapsed_time = time.time() - start_time

    # Calculate final metrics using shared implementation
    # Move to CPU for evaluation (numpy operations)
    final_cell_features = result["final_cell_features"]
    final_cell_features_cpu = final_cell_features.cpu() if final_cell_features.is_cuda else final_cell_features
    pin_features_cpu = pin_features.cpu() if pin_features.is_cuda else pin_features
    edge_list_cpu = edge_list.cpu() if edge_list.is_cuda else edge_list
    
    metrics = calculate_normalized_metrics(final_cell_features_cpu, pin_features_cpu, edge_list_cpu)

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
        print(f"  Time: {result['elapsed_time']:.2f}s")
        print(f"  Status: {status}")
        print()

    # Compute aggregate statistics
    avg_overlap_ratio = sum(r["overlap_ratio"] for r in all_results) / len(all_results)
    avg_normalized_wl = sum(r["normalized_wl"] for r in all_results) / len(all_results)
    total_time = sum(r["elapsed_time"] for r in all_results)

    # Print aggregate results
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Average Overlap: {avg_overlap_ratio:.4f}")
    print(f"Average Wirelength: {avg_normalized_wl:.4f}")
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
