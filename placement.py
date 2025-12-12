"""
VLSI Cell Placement Optimization Challenge
==========================================

Main entry point for the placement optimization system.
"""

import os
import math
import time
import torch

from placement_modules.losses import overlap_repulsion_loss
from placement_modules.training import train_placement
from placement_modules.metrics import calculate_overlap_metrics, calculate_cells_with_overlaps, calculate_normalized_metrics
from placement_modules.visualization import plot_placement
from placement_modules.cuda_setup import check_and_setup_cuda_backend
from placement_modules.utils import (
    CellFeatureIdx, PinFeatureIdx,
    MIN_MACRO_AREA, MAX_MACRO_AREA,
    STANDARD_CELL_AREAS, STANDARD_CELL_HEIGHT,
    MIN_STANDARD_CELL_PINS, MAX_STANDARD_CELL_PINS,
    OUTPUT_DIR
)

# Set OUTPUT_DIR in utils module
import placement_modules.utils as placement_utils
placement_utils.OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_placement_input(num_macros, num_std_cells):
    """Generate synthetic placement input data.

    Args:
        num_macros: Number of macros to generate
        num_std_cells: Number of standard cells to generate

    Returns:
        Tuple of (cell_features, pin_features, edge_list):
            - cell_features: torch.Tensor of shape [N, 6] with columns [area, num_pins, x, y, width, height]
            - pin_features: torch.Tensor of shape [total_pins, 7] with columns
              [cell_instance_index, pin_x, pin_y, x, y, pin_width, pin_height]
            - edge_list: torch.Tensor of shape [E, 2] with [src_pin_idx, tgt_pin_idx]
    """
    total_cells = num_macros + num_std_cells

    # Step 1: Generate macro areas (uniformly distributed between min and max)
    macro_areas = (
        torch.rand(num_macros) * (MAX_MACRO_AREA - MIN_MACRO_AREA) + MIN_MACRO_AREA
    )

    # Step 2: Generate standard cell areas (randomly pick from 1, 2, or 3)
    std_cell_areas = torch.tensor(STANDARD_CELL_AREAS)[
        torch.randint(0, len(STANDARD_CELL_AREAS), (num_std_cells,))
    ]

    # Combine all areas
    areas = torch.cat([macro_areas, std_cell_areas])

    # Step 3: Calculate cell dimensions
    # Macros are square
    macro_widths = torch.sqrt(macro_areas)
    macro_heights = torch.sqrt(macro_areas)

    # Standard cells have fixed height = 1, width = area
    std_cell_widths = std_cell_areas / STANDARD_CELL_HEIGHT
    std_cell_heights = torch.full((num_std_cells,), STANDARD_CELL_HEIGHT)

    # Combine dimensions
    cell_widths = torch.cat([macro_widths, std_cell_widths])
    cell_heights = torch.cat([macro_heights, std_cell_heights])

    # Step 4: Calculate number of pins per cell
    num_pins_per_cell = torch.zeros(total_cells, dtype=torch.int)

    # Macros: between sqrt(area) and 2*sqrt(area) pins
    for i in range(num_macros):
        sqrt_area = int(torch.sqrt(macro_areas[i]).item())
        num_pins_per_cell[i] = torch.randint(sqrt_area, 2 * sqrt_area + 1, (1,)).item()

    # Standard cells: between 3 and 6 pins
    num_pins_per_cell[num_macros:] = torch.randint(
        MIN_STANDARD_CELL_PINS, MAX_STANDARD_CELL_PINS + 1, (num_std_cells,)
    )

    # Step 5: Create cell features tensor [area, num_pins, x, y, width, height]
    cell_features = torch.zeros(total_cells, 6)
    cell_features[:, CellFeatureIdx.AREA] = areas
    cell_features[:, CellFeatureIdx.NUM_PINS] = num_pins_per_cell.float()
    cell_features[:, CellFeatureIdx.X] = 0.0  # x position (initialized to 0)
    cell_features[:, CellFeatureIdx.Y] = 0.0  # y position (initialized to 0)
    cell_features[:, CellFeatureIdx.WIDTH] = cell_widths
    cell_features[:, CellFeatureIdx.HEIGHT] = cell_heights

    # Step 6: Generate pins for each cell
    total_pins = num_pins_per_cell.sum().item()
    pin_features = torch.zeros(total_pins, 7)

    # Fixed pin size for all pins (square pins)
    PIN_SIZE = 0.1  # All pins are 0.1 x 0.1

    pin_idx = 0
    for cell_idx in range(total_cells):
        n_pins = num_pins_per_cell[cell_idx].item()
        cell_width = cell_widths[cell_idx].item()
        cell_height = cell_heights[cell_idx].item()

        # Generate random pin positions within the cell
        # Offset from edges to ensure pins are fully inside
        margin = PIN_SIZE / 2
        if cell_width > 2 * margin and cell_height > 2 * margin:
            pin_x = torch.rand(n_pins) * (cell_width - 2 * margin) + margin
            pin_y = torch.rand(n_pins) * (cell_height - 2 * margin) + margin
        else:
            # For very small cells, just center the pins
            pin_x = torch.full((n_pins,), cell_width / 2)
            pin_y = torch.full((n_pins,), cell_height / 2)

        # Fill pin features
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.CELL_IDX] = cell_idx
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.PIN_X] = (
            pin_x  # relative to cell
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.PIN_Y] = (
            pin_y  # relative to cell
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.X] = (
            pin_x  # absolute (same as relative initially)
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.Y] = (
            pin_y  # absolute (same as relative initially)
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.WIDTH] = PIN_SIZE
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.HEIGHT] = PIN_SIZE

        pin_idx += n_pins

    # Step 7: Generate edges with simple random connectivity
    # Each pin connects to 1-3 random pins (preferring different cells)
    edge_list = []
    avg_edges_per_pin = 2.0

    pin_to_cell = torch.zeros(total_pins, dtype=torch.long)
    pin_idx = 0
    for cell_idx, n_pins in enumerate(num_pins_per_cell):
        pin_to_cell[pin_idx : pin_idx + n_pins] = cell_idx
        pin_idx += n_pins

    # Create adjacency set to avoid duplicate edges
    adjacency = [set() for _ in range(total_pins)]

    for pin_idx in range(total_pins):
        pin_cell = pin_to_cell[pin_idx].item()
        num_connections = torch.randint(1, 4, (1,)).item()  # 1-3 connections per pin

        # Try to connect to pins from different cells
        for _ in range(num_connections):
            # Random candidate
            other_pin = torch.randint(0, total_pins, (1,)).item()

            # Skip self-connections and existing connections
            if other_pin == pin_idx or other_pin in adjacency[pin_idx]:
                continue

            # Add edge (always store smaller index first for consistency)
            if pin_idx < other_pin:
                edge_list.append([pin_idx, other_pin])
            else:
                edge_list.append([other_pin, pin_idx])

            # Update adjacency
            adjacency[pin_idx].add(other_pin)
            adjacency[other_pin].add(pin_idx)

    # Convert to tensor and remove duplicates
    if edge_list:
        edge_list = torch.tensor(edge_list, dtype=torch.long)
        edge_list = torch.unique(edge_list, dim=0)
    else:
        edge_list = torch.zeros((0, 2), dtype=torch.long)

    print(f"\nGenerated placement data:")
    print(f"  Total cells: {total_cells}")
    print(f"  Total pins: {total_pins}")
    print(f"  Total edges: {len(edge_list)}")
    print(f"  Average edges per pin: {2 * len(edge_list) / total_pins:.2f}")

    return cell_features, pin_features, edge_list


def main():
    """Main function demonstrating the placement optimization challenge."""
    # Check and setup CUDA backend if needed
    if not check_and_setup_cuda_backend():
        return  # User chose to exit
    
    print("=" * 70)
    print("VLSI CELL PLACEMENT OPTIMIZATION CHALLENGE")
    print("=" * 70)
    print("\nObjective: Implement overlap_repulsion_loss() to eliminate cell overlaps")
    print("while minimizing wirelength.\n")
    
    # Print backend status
    if torch.cuda.is_available():
        print("ℹ GPU available - PyTorch operations will use GPU acceleration automatically")
    else:
        print("ℹ Using CPU-optimized PyTorch implementation")
    print()

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate placement problem
    num_macros = 3
    num_std_cells = 50

    print(f"Generating placement problem:")
    print(f"  - {num_macros} macros")
    print(f"  - {num_std_cells} standard cells")

    cell_features, pin_features, edge_list = generate_placement_input(
        num_macros, num_std_cells
    )
    
    # Move tensors to GPU if available
    cell_features = cell_features.to(device)
    pin_features = pin_features.to(device)
    edge_list = edge_list.to(device)

    # Initialize positions with random spread to reduce initial overlaps
    total_cells = cell_features.shape[0]
    spread_radius = 30.0
    # Create random tensors on the same device as cell_features
    angles = torch.rand(total_cells, device=device) * 2 * 3.14159
    radii = torch.rand(total_cells, device=device) * spread_radius

    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)

    # Calculate initial metrics (move to CPU for numpy operations)
    print("\n" + "=" * 70)
    print("INITIAL STATE")
    print("=" * 70)
    cell_features_cpu = cell_features.cpu() if cell_features.is_cuda else cell_features
    initial_metrics = calculate_overlap_metrics(cell_features_cpu)
    print(f"Overlap count: {initial_metrics['overlap_count']}")
    print(f"Total overlap area: {initial_metrics['total_overlap_area']:.2f}")
    print(f"Max overlap area: {initial_metrics['max_overlap_area']:.2f}")
    print(f"Overlap percentage: {initial_metrics['overlap_percentage']:.2f}%")

    # Run optimization
    print("\n" + "=" * 70)
    print("RUNNING OPTIMIZATION")
    print("=" * 70)
    
    # Time the optimization
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)
    start_time = time.time()

    result = train_placement(
        cell_features,
        pin_features,
        edge_list,
        num_epochs=1000,
        lr=0.1,  # Maximum LR for cosine annealing (will decay smoothly)
        lambda_wirelength=1.0,
        lambda_overlap=100.0,  # Maximum overlap penalty (increased for stronger push)
        verbose=True,
        log_interval=50,
    )
    
    # Measure elapsed time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print(f"\nOptimization completed in {elapsed_time:.2f} seconds")

    # Calculate final metrics (both detailed and normalized)
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    final_cell_features = result["final_cell_features"]
    
    # Move back to CPU for evaluation (if on GPU)
    final_cell_features_cpu = final_cell_features.cpu() if final_cell_features.is_cuda else final_cell_features
    pin_features_cpu = pin_features.cpu() if pin_features.is_cuda else pin_features
    edge_list_cpu = edge_list.cpu() if edge_list.is_cuda else edge_list

    # Detailed metrics
    final_metrics = calculate_overlap_metrics(final_cell_features_cpu)
    print(f"Overlap count (pairs): {final_metrics['overlap_count']}")
    print(f"Total overlap area: {final_metrics['total_overlap_area']:.2f}")
    print(f"Max overlap area: {final_metrics['max_overlap_area']:.2f}")

    # Normalized metrics (matching test suite)
    print("\n" + "-" * 70)
    print("TEST SUITE METRICS (for leaderboard)")
    print("-" * 70)
    normalized_metrics = calculate_normalized_metrics(
        final_cell_features_cpu, pin_features_cpu, edge_list_cpu
    )
    print(f"Overlap Ratio: {normalized_metrics['overlap_ratio']:.4f} "
          f"({normalized_metrics['num_cells_with_overlaps']}/{normalized_metrics['total_cells']} cells)")
    print(f"Normalized Wirelength: {normalized_metrics['normalized_wl']:.4f}")

    # Success check with validation
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA")
    print("=" * 70)
    
    # Validate metrics for NaN/Inf values and invalid states
    # Check if metrics are valid numbers
    normalized_wl = normalized_metrics['normalized_wl']
    overlap_ratio = normalized_metrics['overlap_ratio']
    num_cells_with_overlaps = normalized_metrics['num_cells_with_overlaps']
    
    # Check for NaN or Inf in metrics
    has_invalid_metrics = (
        not math.isfinite(normalized_wl) or 
        not math.isfinite(overlap_ratio) or
        normalized_wl < 0 or  # Wirelength should be non-negative
        overlap_ratio < 0 or overlap_ratio > 1  # Overlap ratio should be in [0, 1]
    )
    
    # Check if final cell positions are valid (no NaN/Inf)
    final_positions = final_cell_features_cpu[:, 2:4].detach().numpy()
    has_invalid_positions = (
        not math.isfinite(final_positions.min()) or 
        not math.isfinite(final_positions.max())
    )
    
    # Check if loss history indicates broken optimization
    loss_history = result.get("loss_history", {})
    total_losses = loss_history.get("total_loss", [])
    has_broken_optimization = False
    nan_count = 0
    if total_losses:
        # Check if more than 50% of losses are NaN (indicating broken optimization)
        nan_count = sum(1 for loss in total_losses if not math.isfinite(loss))
        has_broken_optimization = (nan_count / len(total_losses)) > 0.5
    
    # Determine if solution is valid
    is_valid_solution = (
        not has_invalid_metrics and 
        not has_invalid_positions and 
        not has_broken_optimization
    )
    
    # Only pass if solution is valid AND no overlaps exist
    if is_valid_solution and num_cells_with_overlaps == 0:
        print("✓ PASS: No overlapping cells!")
        print("✓ PASS: Overlap ratio is 0.0")
        print("✓ PASS: All metrics are valid (no NaN/Inf)")
        print("\nCongratulations! Your implementation successfully eliminated all overlaps.")
        print(f"Your normalized wirelength: {normalized_wl:.4f}")
    else:
        # Determine failure reason
        failure_reasons = []
        
        if num_cells_with_overlaps > 0:
            failure_reasons.append(f"Overlaps still exist ({num_cells_with_overlaps} cells)")
        
        if has_invalid_metrics:
            failure_reasons.append("Invalid metrics detected (NaN/Inf values)")
            if not math.isfinite(normalized_wl):
                print(f"  ⚠ WARNING: Normalized wirelength is {normalized_wl}")
            if not math.isfinite(overlap_ratio):
                print(f"  ⚠ WARNING: Overlap ratio is {overlap_ratio}")
        
        if has_invalid_positions:
            failure_reasons.append("Invalid cell positions detected (NaN/Inf)")
            print(f"  ⚠ WARNING: Final cell positions contain NaN or Inf values")
        
        if has_broken_optimization:
            failure_reasons.append("Optimization appears broken (loss values are NaN)")
            print(f"  ⚠ WARNING: {nan_count}/{len(total_losses)} loss values are NaN/Inf")
            print(f"  This indicates numerical instability in your loss function")
        
        print("✗ FAIL: Solution does not meet success criteria")
        for reason in failure_reasons:
            print(f"  - {reason}")
        
        print("\nSuggestions:")
        if has_invalid_metrics or has_broken_optimization:
            print("  1. Check for numerical instability in your loss functions")
            print("     - Avoid operations that can produce NaN/Inf (e.g., exp() of large values)")
            print("     - Add numerical stability checks (clamping, epsilon values)")
            print("     - Verify gradients are finite before optimizer.step()")
        if num_cells_with_overlaps > 0:
            print("  2. Check your overlap_repulsion_loss() implementation")
            print("  3. Try increasing lambda_overlap or adjusting learning rate")

    # Generate visualization only for smaller problems (plotting 100k+ cells is too slow)
    # Skip placement visualization for problems with >10000 cells
    if cell_features.shape[0] <= 10000:
        initial_cell_features_cpu = result["initial_cell_features"].cpu() if result["initial_cell_features"].is_cuda else result["initial_cell_features"]
        final_cell_features_cpu_viz = final_cell_features.cpu() if final_cell_features.is_cuda else final_cell_features
        plot_placement(
            initial_cell_features_cpu,
            final_cell_features_cpu_viz,
            pin_features_cpu,
            edge_list_cpu,
            filename="placement_result.png",
        )
    else:
        print(f"  Skipping placement visualization (too many cells: {cell_features.shape[0]})")
    
    # Plot overlap loss over epochs (for single test case in main)
    from placement_modules.visualization import plot_overlap_loss_history
    loss_history = result.get("loss_history", {})
    if loss_history and "overlap_loss" in loss_history:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = plot_overlap_loss_history(
            [{"overlap_loss": loss_history["overlap_loss"]}],
            ["Single Test Case"],
            output_dir,
            "overlap_loss_history_single.png"
        )
        if plot_path:
            print(f"\n✓ Overlap loss plot saved to: {plot_path}")


if __name__ == "__main__":
    main()

