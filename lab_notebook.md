# Lab notebook — placement overlap / training

Concise log of what was tried and measured. Re-run `python test.py` for full leaderboard-style numbers.

## Approaches tried

| Approach | Notes |
|----------|--------|
| **Baseline relu overlap** | Pairwise `relu(min_sep − \|Δ\|)` overlap area, mean over all pairs; overlap term often ~100% of objective vs wirelength. |
| **Legacy modes** | `area`, `squared`, `both` (mean over all pairs); selectable via `overlap_loss_mode`. |
| **Fast loss (`mode="fast"`, default)** | `relu` overlap area sum ÷ (number of overlapping pairs + 1). Stronger gradients than mean over all pairs when few pairs overlap. |
| **Per-cell grad clip** | Clip L2 norm **per cell** on position grads (`per_cell_grad_clip_norm`), instead of global `clip_grad_norm_`. |
| **Loss diagnostics** | `loss_history`: weighted WL / OL, overlap share, scheduled λ, `lr`. Plots: overlap-ratio tags. Overlap checks use `_fast_overlap_ratio` (torch, no Python pair loops). |
| **Overlap sweep script** | `plot_overlap_loss_vs_cells.py`: overlap loss vs N (2…50) for heights 1,2,3 (`overlap_loss_vs_num_cells.png`). |

## Hyperparameters (see `train_placement` defaults in `placement.py`)

## Test results (harness `test.run_placement_test`, seeds from `TEST_CASES`)

Recorded on one local run (macOS / project env). **Overlap target:** `num_cells_with_overlaps == 0`.

| Test id | Macros × std cells | Overlap ratio | Cells w/ overlap | Time (s) | Result |
|--------:|---------------------|---------------|------------------|----------|--------|
| 1 | 2 × 20 | 0.0 | 0 / 22 | ~5.6 | PASS |
| 2 | 3 × 25 | 0.0 | 0 / 28 | ~6.5 | PASS |
| 3 | 2 × 30 | 0.0 | 0 / 32 | ~6.0 | PASS |
| 4–10 | — | — | — | — | *Re-run `python test.py` (test 10: 2010 cells × 10k epochs, long)* |

## Failed / partial experiments (historical)

- **λ_ol=10, legacy loss, short epochs:** overlap ratio often remained > 0; needed more epochs and/or higher λ_ol.

## Commands

```bash
python placement.py          # demo + placement_result.png + training_loss_curves.png if enabled
python test.py               # full suite (12 cases; 11–12 extra credit / very large)
python plot_overlap_loss_vs_cells.py
```
