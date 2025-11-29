"""
Visualization functions for VLSI placement.
"""

import os
import numpy as np
from .utils import OUTPUT_DIR
from .metrics import calculate_overlap_metrics

def plot_placement(
    initial_cell_features,
    final_cell_features,
    pin_features,
    edge_list,
    filename="placement_result.png",
):
    """Create side-by-side visualization of initial vs final placement.

    Args:
        initial_cell_features: Initial cell positions and properties
        final_cell_features: Optimized cell positions and properties
        pin_features: Pin information
        edge_list: Edge connectivity
        filename: Output filename for the plot
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot both initial and final placements
        for ax, cell_features, title in [
            (ax1, initial_cell_features, "Initial Placement"),
            (ax2, final_cell_features, "Final Placement"),
        ]:
            N = cell_features.shape[0]
            positions = cell_features[:, 2:4].detach().numpy()
            widths = cell_features[:, 4].detach().numpy()
            heights = cell_features[:, 5].detach().numpy()

            # Draw cells
            for i in range(N):
                x = positions[i, 0] - widths[i] / 2
                y = positions[i, 1] - heights[i] / 2
                rect = Rectangle(
                    (x, y),
                    widths[i],
                    heights[i],
                    fill=True,
                    facecolor="lightblue",
                    edgecolor="darkblue",
                    linewidth=0.5,
                    alpha=0.7,
                )
                ax.add_patch(rect)

            # Calculate and display overlap metrics
            metrics = calculate_overlap_metrics(cell_features)

            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title(
                f"{title}\n"
                f"Overlaps: {metrics['overlap_count']}, "
                f"Total Overlap Area: {metrics['total_overlap_area']:.2f}",
                fontsize=12,
            )

            # Set axis limits with margin
            all_x = positions[:, 0]
            all_y = positions[:, 1]
            margin = 10
            ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
            ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

        plt.tight_layout()
        # Save to results folder
        results_dir = os.path.join(OUTPUT_DIR, "results") if OUTPUT_DIR else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    except ImportError as e:
        print(f"Could not create visualization: {e}")
        print("Install matplotlib to enable visualization: pip install matplotlib")


def plot_overlap_loss_history(loss_histories, test_labels=None, output_dir=None, filename="overlap_loss_history.png"):
    """Plot overlap loss over epochs for multiple test cases on a single plot.
    
    Args:
        loss_histories: List of dictionaries, each containing:
            - 'overlap_loss': List of overlap loss values per epoch
            - 'test_id': (optional) Test case identifier
        test_labels: (optional) List of labels for each test case. If None, uses test_id or default labels.
        output_dir: Directory to save the plot. If None, uses OUTPUT_DIR.
        filename: Output filename for the plot.
    
    Returns:
        Path to saved plot file, or None if plotting failed.
    """
    try:
        import matplotlib.pyplot as plt
        
        if not loss_histories:
            return None
        
        # Use provided output_dir or fall back to OUTPUT_DIR
        if output_dir is None:
            output_dir = OUTPUT_DIR if OUTPUT_DIR else os.path.dirname(os.path.abspath(__file__))
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Define a color palette with distinct colors
        colors = [
            '#9D4EDD',  # Bright purple
            '#FF6B6B',  # Coral red
            '#4ECDC4',  # Turquoise
            '#45B7D1',  # Sky blue
            '#FFA07A',  # Light salmon
            '#98D8C8',  # Mint green
            '#F7DC6F',  # Yellow
            '#BB8FCE',  # Light purple
            '#85C1E2',  # Light blue
            '#F8B88B',  # Peach
            '#82E0AA',  # Light green
            '#F1948A',  # Pink
        ]
        
        # Plot each test case
        for idx, loss_data in enumerate(loss_histories):
            overlap_losses = loss_data.get('overlap_loss', [])
            if not overlap_losses:
                continue
            
            # Generate label
            if test_labels and idx < len(test_labels):
                label = test_labels[idx]
            elif 'test_id' in loss_data:
                label = f"Test {loss_data['test_id']}"
            elif 'num_macros' in loss_data and 'num_std_cells' in loss_data:
                label = f"{loss_data['num_macros']} macros, {loss_data['num_std_cells']} std cells"
            else:
                label = f"Test Case {idx + 1}"
            
            # Filter out NaN/Inf values for plotting
            epochs = np.arange(1, len(overlap_losses) + 1)
            valid_mask = np.isfinite(overlap_losses)
            valid_epochs = epochs[valid_mask]
            valid_losses = np.array(overlap_losses)[valid_mask]
            
            if len(valid_losses) == 0:
                continue
            
            # Downsample for large datasets to speed up plotting
            max_points = 2000  # Maximum points to plot for performance
            if len(valid_losses) > max_points:
                step = len(valid_losses) // max_points
                valid_epochs = valid_epochs[::step]
                valid_losses = valid_losses[::step]
            
            # Select color (cycle through palette)
            color = colors[idx % len(colors)]
            
            # Plot with thick line (no markers for large datasets)
            ax.plot(valid_epochs, valid_losses, 
                   color=color,
                   linewidth=2.5,
                   alpha=0.85,
                   label=label)
        
        # Styling
        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Overlap Loss (log scale)', fontsize=14, fontweight='bold')
        ax.set_title('Overlap Loss Over Training Epochs (All Test Cases)', fontsize=16, fontweight='bold', pad=20)
        ax.set_yscale('log')  # Always use log scale as requested
        ax.grid(True, alpha=0.3, linestyle='--', which='both')  # Show both major and minor grid lines for log scale
        ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
        
        # Improve layout
        plt.tight_layout()
        
        # Save to results folder
        plot_path = os.path.join(results_dir, filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except ImportError:
        print("\n⚠ Could not create overlap loss plot: matplotlib not available")
        return None
    except Exception as e:
        print(f"\n⚠ Could not create overlap loss plot: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_learning_rate_history(loss_histories, test_labels=None, output_dir=None, filename="learning_rate_history.png"):
    """Plot learning rate over epochs for multiple test cases on a single plot.
    
    Args:
        loss_histories: List of dictionaries, each containing:
            - 'learning_rate': List of learning rate values per epoch
            - 'test_id': (optional) Test case identifier
        test_labels: (optional) List of labels for each test case. If None, uses test_id or default labels.
        output_dir: Directory to save the plot. If None, uses OUTPUT_DIR.
        filename: Output filename for the plot.
    
    Returns:
        Path to saved plot file, or None if plotting failed.
    """
    try:
        import matplotlib.pyplot as plt
        
        if not loss_histories:
            return None
        
        # Use provided output_dir or fall back to OUTPUT_DIR
        if output_dir is None:
            output_dir = OUTPUT_DIR if OUTPUT_DIR else os.path.dirname(os.path.abspath(__file__))
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Define a color palette with distinct colors
        colors = [
            '#9D4EDD',  # Bright purple
            '#FF6B6B',  # Coral red
            '#4ECDC4',  # Turquoise
            '#45B7D1',  # Sky blue
            '#FFA07A',  # Light salmon
            '#98D8C8',  # Mint green
            '#F7DC6F',  # Yellow
            '#BB8FCE',  # Light purple
            '#85C1E2',  # Light blue
            '#F8B88B',  # Peach
            '#82E0AA',  # Light green
            '#F1948A',  # Pink
        ]
        
        # Plot each test case
        for idx, loss_data in enumerate(loss_histories):
            learning_rates = loss_data.get('learning_rate', [])
            if not learning_rates:
                continue
            
            epochs = np.arange(len(learning_rates))
            learning_rates = np.array(learning_rates)
            
            # Downsample for large datasets to speed up plotting
            max_points = 2000  # Maximum points to plot for performance
            if len(learning_rates) > max_points:
                step = len(learning_rates) // max_points
                epochs = epochs[::step]
                learning_rates = learning_rates[::step]
            
            color = colors[idx % len(colors)]
            
            # Get label
            if test_labels and idx < len(test_labels):
                label = test_labels[idx]
            elif 'test_id' in loss_data:
                label = f"Test {loss_data['test_id']}"
            else:
                label = f"Test {idx + 1}"
            
            ax.plot(epochs, learning_rates, label=label, color=color, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Learning Rate', fontsize=14, fontweight='bold')
        ax.set_title('Learning Rate History Over Training', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
        
        # Use log scale if LR varies significantly
        all_lrs = []
        for loss_data in loss_histories:
            lrs = loss_data.get('learning_rate', [])
            if lrs:
                all_lrs.extend(lrs)
        
        if all_lrs:
            lr_min = min(all_lrs)
            lr_max = max(all_lrs)
            if lr_max / lr_min > 10:
                ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(results_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    except Exception as e:
        print(f"Error plotting learning rate history: {e}")
        import traceback
        traceback.print_exc()
        return None

