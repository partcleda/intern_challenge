# Technical Explanation: VLSI Cell Placement Optimization

This document describes the modifications made to the original placement optimization code to achieve zero cell overlaps while minimizing wirelength. The implementation passes all 10 test cases, with results available in the `res/` folder.

## Part 1: Test Harness Modifications (`test.py`)

### Changes from original `test.py`

**1. Directory Creation for Results**

Added automatic directory creation before running tests:

```python
if not os.path.exists('res'):
    os.makedirs('res')
```

This ensures the output directory exists before attempting to save visualization files.

**2. Filename Parameter Addition**

Modified the `run_placement_test` function signature:

```python
def run_placement_test(
    test_id,
    num_macros,
    num_std_cells,
    seed=None,
    filename=None  # New parameter
):
```

Added filename argument when calling `train_placement`:

```python
result = train_placement(
    cell_features,
    pin_features,
    edge_list,
    verbose=False,
    filename=f'res/placement_result_test{test_id}.png'
)
```

This modification enables saving individual placement visualizations for each test case, allowing analysis of the optimizer's performance across different problem sizes.

## Part 2: Placement Algorithm Modifications (`placement.py`)

### Overview

The original implementation provided only a placeholder for the `overlap_repulsion_loss` function and used a simple single-phase training loop. The modified version implements a complete overlap detection system with a multi-stage training strategy.

### Overlap Repulsion Loss Implementation

**1. Pairwise Overlap Calculation**

Implemented vectorized computation of all cell-pair overlaps using PyTorch broadcasting:

```python
# Extract cell properties
positions = cell_features[:, 2:4]
widths = cell_features[:, 4]
heights = cell_features[:, 5]

# Compute all pairwise position differences
pos_i = positions.unsqueeze(1)  # [N, 1, 2]
pos_j = positions.unsqueeze(0)  # [1, N, 2]
delta_pos = pos_i - pos_j        # [N, N, 2]
```

This approach creates an N×N matrix of position differences, allowing parallel processing of all cell pairs without explicit loops.

**2. Overlap Detection Logic**

For each cell pair, calculated the minimum required separation distance and compared it to the actual distance:

```python
min_sep_x = (width_i + width_j) / 2
min_sep_y = (height_i + height_j) / 2
dist_x = torch.abs(delta_pos[:, :, 0])
dist_y = torch.abs(delta_pos[:, :, 1])

overlap_x = torch.relu((min_sep_x - dist_x) + progress)
overlap_y = torch.relu((min_sep_y - dist_y) + progress)
```

The `progress` parameter (ranging from 0 to 1 during training) provides a small epsilon value that prevents gradient vanishing when cells are exactly touching. This maintains gradient flow throughout training.

**3. Penalty Calculation**

Applied quadratic penalty to overlap areas:

```python
overlap_area = overlap_x * overlap_y
overlap_penalty = overlap_area ** 2
```

The quadratic penalty creates stronger repulsion for larger overlaps, helping cells separate more effectively.

**4. Upper Triangle Masking**

Used upper triangle masking to count each cell pair only once:

```python
mask = torch.triu(torch.ones((N, N), dtype=torch.float32, device=cell_features.device), diagonal=1)
overlap_penalty = overlap_penalty * mask
total_overlap = torch.sum(overlap_penalty)
```

This prevents double-counting overlaps since the overlap between cells i and j is the same as between j and i.

### Training Loop Modifications

**1. Hyperparameter Adjustments**

Changed default training parameters:

- `num_epochs`: 1000 → 2500 (more training time)
- `lr`: 0.01 → 1.2 (higher initial learning rate)
- `lambda_overlap`: 10.0 → 250.0 (stronger overlap penalty)

**2. Five-Stage Training Strategy**

Implemented a staged training approach that balances wirelength optimization with overlap elimination:

**Stage 1 (0-10%): Pure Wirelength Optimization**
- Learning rate: 1.6
- Lambda overlap: 0.0
- Rationale: Focus exclusively on minimizing wirelength without overlap constraints, establishing a good initial layout.

**Stage 2 (10-50%): Gradual Overlap Introduction**
- Learning rate: 1.0
- Lambda overlap: 0 → 50 (exponential ramp with exponent 10)
- Rationale: Slowly introduce overlap penalties while maintaining high learning rate for continued wirelength optimization.

**Stage 3 (50-75%): Moderate Overlap Enforcement**
- Learning rate: 0.6
- Lambda overlap: 50 → 250 (exponential ramp with exponent 10)
- Rationale: Increase overlap penalty more aggressively while reducing learning rate for stability.

**Stage 4 (75-90%): Overlap Elimination**
- Learning rate: 0.4
- Lambda overlap: 250 (fixed)
- Rationale: Apply full overlap penalty with moderate learning rate to eliminate remaining overlaps.

**Stage 5 (90-100%): Final Fine-tuning**
- Learning rate: 0.2
- Lambda overlap: 187.5 (75% of maximum)
- Rationale: Small learning rate with slightly reduced overlap penalty allows minor wirelength improvements without reintroducing overlaps.

**3. Dynamic Learning Rate and Lambda Updates**

Implemented per-epoch adjustment of both learning rate and overlap penalty weight:

```python
progress = epoch / num_epochs
# Stage-based LR and lambda calculation
for param_group in optimizer.param_groups:
    param_group['lr'] = current_lr
```

This dynamic adjustment allows fine-grained control over the optimization trajectory.

**4. Gradient Clipping**

Changed gradient clipping from `max_norm=5.0` to `max_norm=10.0`:

```python
torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=10.0)
```

This allows larger gradient updates during high learning rate phases while still preventing instability.

**5. Progress-Aware Loss Function**

Modified the overlap loss function call to include training progress:

```python
overlap_loss = overlap_repulsion_loss(
    cell_features_current, pin_features, edge_list, progress
)
```

This couples the loss function behavior to the training stage, providing smoother gradients in later stages.

**6. Visualization Integration**

Added optional visualization saving at the end of training:

```python
if filename:
    plot_placement(
        initial_cell_features,
        final_cell_features,
        pin_features,
        edge_list,
        filename
    )
```

This allows the test harness to automatically generate result images for each test case.

### Design Rationale

The key insight behind this implementation is that wirelength optimization and overlap elimination are competing objectives that require careful balancing. The staged training approach addresses this by:

1. Allowing the optimizer to find a good wirelength-optimal layout first
2. Gradually introducing overlap constraints to avoid disrupting the layout
3. Using exponential ramping (with high exponents like 10) to keep penalties low for most of the ramp duration, then sharply increase near the end
4. Reducing learning rates in later stages to prevent oscillations when both objectives are active
5. Slightly relaxing overlap penalty in the final stage to allow minor wirelength improvements

The progress-aware epsilon in the overlap calculation prevents a common failure mode where gradients vanish once overlaps are eliminated, making it difficult to maintain zero overlaps in the final epochs.

## Results

All 10 test cases pass with zero overlaps. Visualization results for each test case are saved in the `res/` directory as PNG files named `placement_result_test{N}.png` where N ranges from 1 to 10.
