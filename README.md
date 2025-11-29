# Intern Challenge: Placement Problem

Welcome to the par.tcl 2026 ML Sys intern challenge! Your task is to solve a placement problem involving standard cells (small blocks) and macros (large blocks). The **primary goal is to minimize overlap** between blocks. Wirelength is also evaluated, but **overlap is the dominant objective**. A valid placement must eventually ensure no blocks overlap, but we will judge solutions by how effectively you reduce overlap and, secondarily, how well you handle wirelength.

The deadline is when all intern slots for summer 2026 are filled. We will review submissions on a rolling basis.

## Problem Statement

- **Objective:** Place a set of standard cells and macros on a chip layout to **minimize overlap (most important)** and wirelength (secondary).  
  - Overlap will be measured as `num overlapping cells / num total cells`, though you are encouraged to define and implement your own overlap loss function if you think it’s better.  
  - Solving this problem will require designing a strong overlap loss, tuning hyperparameters, and experimenting with optimizers. Creativity is encouraged — nothing is off the table.  
- **Input:** Randomly generated netlists.  
- **Output:** Average normalized **overlap (primary metric)** and wirelength (secondary metric) across a set of randomized placements.  

## Submission Instructions

1. Fork this repository.  
2. Solve the placement problem using your preferred tools or scripts.  
3. Run the test script to evaluate your solution and obtain the overlap and wirelength metrics.  
4. Submit a pull request with your updated leaderboard entry and instructions for me to access your actual submission (it's fine if it's public).  

Note: You can use any libraries or frameworks you like, but please ensure that your code is well-documented and easy to follow.  

Also, if you think there are any bugs in the provided code, feel free to fix them and mention the changes in your submission.  

You may submit multiple solutions to try and increase your score.

We will review submissions on a rolling basis.

## New Leaderboard (sorted by overlap)

| Rank | Name            | Overlap     | Wirelength (um) | Runtime (s) | Notes                |
|------|-----------------|-------------|-----------------|-------------|----------------------|
| 1    |   example       | 0.5000      | 0.5             |  10         |   example submission |
| 2    | Add Yours!      |             |                 |             |                      |



## Leaderboard (sorted by overlap) (OLD; test suite has been updated; see above)

| Rank | Name            | Overlap     | Wirelength (um) | Runtime (s) | Notes                |
|------|-----------------|-------------|-----------------|-------------|----------------------|
| 1    | Shashank Shriram  | 0.0000     | 0.1310          |  11.32      |   🏎️💥               |
| 2    | Brayden Rudisill  | 0.0000    | 0.2611          |   50.51     |   Timed on a mac air |
| 3    | manuhalapeth      | 0.0000    | 0.2630          |  196.8      |                      |
| 4    | Neil Teje         | 0.0000    | 0.2700          | 24.00s      |                      |
| 5    | Leison Gao      | 0.0000      | 0.2796          | 50.14s      |                      |
| 6    | William Pan     | 0.0000      | 0.2848          | 155.33s     |                      |
| 7    | Ashmit Dutta    | 0.0000      | 0.2870          | 995.58      |  Spent my entire morning (12 am - 6 am) doing this :P       |
| 8    | Pawan Paleja     | 0.0000      | 0.3311         | 1.74s     |   Implemented hint for loss func, cosine annealing on learning rate with warmup, std annealing on lambda weight. Used optuna to tune hyperparam. Tested on gh codespaces 2-core.                   |
| 9    | Gabriel Del Monte  | 0.0000      | 0.3427          | 606.07      |                                                              |
| 10    | Aleksey  Valouev| 0.0000      | 0.3577          | 118.98      |                      |        
| 11   | Mohul Shukla    | 0.0000      | 0.5048          | 54.60s      |                      |
| 12    | Ryan Hulke      | 0.0000      | 0.5226          | 166.24      |                      |
| 13    | Neel  Shah      | 0.0000      | 0.5445          | 45.40       |  Zero overlaps on all tests, adaptive schedule + early stop |
| 14   | Nawel Asgar    | 0.0000     | 0.5675          | 81.49      | Adaptive penalty scaling with cubic gradients and design-size optimization
| 15   | Shiva Baghel     | 0.0000     | 0.5885          | 491.00      | Stable zero-overlap with balanced optimization      |
| 16   | Vansh Jain      | 0.0000      | 0.9352          | 86.36       |                      |
| 17    | Akash Pai       | 0.0006      | 0.4933          | 326.25s     |                      |
| 18    | Zade Mahayni     | 0.00665     | 0.5157          |  127.4     | Will try again tomorrow |
| 19    | Nithin Yanna    | 0.0148      | 0.5034          | 247.30s     | aggressive overlap penalty with quadratic scaling |
| 20    | Sean Ko         | 0.0271      |  .5138          | 31.83s      | lr increase, decrease epoch, increase lambda overlap and decreased lambda wire_length + log penalty loss |
| 21    | Keya Gohil    | 0.0155      | 0.4678         | 1513.07     | Still working |
| 22    | Prithvi Seran   | 0.0499      | 0.4890          | 398.58      |                      |
| 23    | partcl example  | 0.8         | 0.4             | 5           | example              |
| 24    | Add Yours!      |             |                 |             |                      |

> **To add your results:**  
> Insert a new row in the table above with your name, overlap, wirelength, and any notes. Ensure you sort by overlap.

Good luck!


---

## Code Organization

The codebase has been refactored into a modular structure for better maintainability. The original monolithic `placement.py` (2372 lines) has been split into logical components.

### Directory Structure

```
intern_challenge/
├── placement.py              # Main entry point
├── test.py                   # Test suite
├── placement_modules/        # Modular components
│   ├── __init__.py
│   ├── utils.py             # Shared enums, constants, and utilities
│   ├── losses.py            # Wirelength and overlap loss functions
│   ├── training.py          # Training loop with adaptive hyperparameters (2500 epochs for 100k+ cells)
│   ├── metrics.py           # Evaluation metrics (overlap, wirelength)
│   ├── visualization.py     # Plotting functions
│   ├── surrogate_gradients.py  # Surrogate gradient functions for optimization
│   ├── cuda_setup.py        # CUDA backend setup and checking
│   ├── README.md            # Detailed module documentation
│   └── SUCCESS_CRITERIA_IMPROVEMENT.md  # Success criteria validation improvements
└── cuda_backend/            # CUDA-accelerated overlap computation
    ├── __init__.py
    ├── setup_cuda.py        # Build script for CUDA extension
    ├── overlap_cuda.py      # Python wrapper for CUDA kernels
    ├── overlap_cuda.cpp     # C++ wrapper binding CUDA kernels to PyTorch
    └── overlap_cuda_kernel.cu  # CUDA kernels for forward/backward overlap computation
```

### Module Overview

- **`placement.py`**: Main entry point containing `generate_placement_input()` and `main()` functions
- **`placement_modules/utils.py`**: Shared utilities (enums, constants, debug flags)
- **`placement_modules/losses.py`**: Core loss functions for wirelength and overlap computation
- **`placement_modules/training.py`**: Training loop with adaptive learning rate and penalty scheduling (2500 epochs for 100k+ cells)
- **`placement_modules/metrics.py`**: Evaluation functions for overlap and wirelength metrics
- **`placement_modules/visualization.py`**: Visualization of placement results and loss history
- **`placement_modules/surrogate_gradients.py`**: Custom gradient functions for optimization
- **`placement_modules/cuda_setup.py`**: CUDA backend availability checking

### CUDA Backend

The `cuda_backend/` folder contains CUDA-accelerated overlap computation for improved performance on large designs (50k+ cells):

- **`setup_cuda.py`**: Build script that compiles CUDA extension using PyTorch's JIT compilation
- **`overlap_cuda.py`**: Python wrapper providing PyTorch-compatible interface to CUDA kernels
- **`overlap_cuda.cpp`**: C++ wrapper that binds CUDA kernels to PyTorch tensors and handles memory management
- **`overlap_cuda_kernel.cu`**: CUDA kernels implementing forward and backward passes for overlap loss computation with spatial hashing and grid-stride loops

The CUDA backend provides 2-3x speedup over PyTorch operations for large problems and is automatically used when available.

See `placement_modules/README.md` for detailed documentation on each module.

### Benefits

- **Maintainability**: Each module has a single, clear responsibility
- **Readability**: Smaller files are easier to understand and navigate
- **Testability**: Individual modules can be tested in isolation
- **Reusability**: Modules can be imported and used independently

---

## Results

Training results, including loss history plots and learning rate schedules, are documented in the `/results` folder.
