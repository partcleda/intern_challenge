# VLSI Cell Placement — Writeup

## Overview

I tired a variety of methods to achieve optimal results. This report goes over the many things I tried and summarizes the issues I encountered and the results I achieved.

## Optimization
I tried different learning rates, SGD vs Adam and different values for the betas and weight decay.

An important improvement was in dividing the process into two phases: overlap dominant and wire length dominant. I added both head and tail versions of overlap dominant, determined both by epoch number and a miniumum threshhold for overlap loss. I also tried various learning rate schedulers such cosine annealing, cosine with warm restarts, linear, one cycle, step and multi-step. Also experimented with epoch number.

# Monitoring
I added saving of plots along the way, as well as for the initialization changes explained below. We also have logic to capture the best historical run along the process of training rather than just returning the final run. This allows it to benefit from noisy exploration. Since we don't care about only the final outcome, it can jump around and select the best result in the entire process.

# Initialization
I tried a variety of initialization methods. I tried randomization, methods placed on both RePlAce and ePlace, a quadratic analytic solution method as well as spectral clustering. I also added method to combine each of these with randomness of varying degrees.

# Perturbation
I found that the training often appeared to get stuck at local minima. Therefore, I felt that some form of perturbation might help it escape these minima (sinch momentum methods alone didn't seem to achieve this.) I added greedy swaps (no macros only regular cells) and noise kicks, both periodic and gated by loss thresholds. Greedy swaps simply swaps two random non-macro cells, scoring the WL loss before and after and only accepting if the loss is improved. We avoid macros because of their size. Noise kicks simple add noise periodically. I also tried introducing some of the initialization methods periodically.

# Analytics
The `calculate_min_possible_normalized_wl` function estimates a theoretical lower bound on total wirelength (WL) for a placement, normalizing this value for fair comparison across problem sizes—it benchmarks the best WL possible if each net’s pins or cells could be placed ideally (e.g., at their geometric medians) without physical constraints. Meanwhile, `print_adjacency_matrix_and_stats` produces and optionally prints the graph's adjacency matrix along with key stats, like node degree distribution, number of isolated nodes, and density, providing insight into the netlist’s connectivity, presence of hubs, and structural bottlenecks. These metrics were useful for understanding lower bounds.

# Density loss
Due to the competing gradients between WL and overlap, I thought another loss for overlap would be beneficial. To help learn reduced wire length without competing with overlap loss, I implemented a density loss. The density loss function is controlled by parameters such as the grid bin size (which determines the spatial resolution for density estimation), penalty exponent (which sets how strongly to penalize overfull regions), kernel radius (controlling the spread of each cell’s influence), and the target density, which can be annealed over the course of training. We also tried combining this with the previous overlap loss.

# Hyperparameter Search
We used grid search across the parameter space to find the optimal hyperparamters.

# Bugs Found
I found two key bugs in the provided function for wire length loss. One is that the Manhattan distance provided is actually a smooth max of the two axis distances. I left the original version of this function as a basis for comparison against the leaderboard, but also implemented a true smooth Manhattan. The other is that this file uses the X, Y position of the cells to mean its center position in some places but the corner of the bottom left cell in the WL loss function, which can lead to miscalculation. This is most easily addressed by adapting it to use the center in that function, which I have provided. All bug fix code is prefixed with the comment "BUG FIX".

# Result
The best parameters I found are the ones used as defaults in train_placement. Most notably, the strategy of a WL phase followed by an overlap phase seemed to work the best. The WL is totally dominated by WL, while the subsequent overlap phase is a combined loss. Also, of all the initialization methods used, the one that worked the best was spectral clustering. I also include the greedy swap method periodically in this solution.

# Summary
The key difficulty of this problem is the competition between the gradients. Overlap loss pushes the cells apart while wire length loss pulls them together. I believe this is the reason my solution struggled to learn an optimal placement, since these competing gradients tend to cancel out and impede learning. I tried multiple strategies around this in order to escape local minima, such as various learning parameters, different phases of training (WL or overlap focused), various initialization and perturbation methods to try to jump start the placement into a more favorable part of the loss tree and also adding a new type of density loss rather than training solely on overlap loss. Unfortunately, while some of these methods improved the results, they still struggled to find an optimal solution. Regardless, this was a fun assignment. I learned a lot about the challenges implicit in this problem, and I am very curious about other techniques that might have performed better.
