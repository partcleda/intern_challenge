# Intern Challenge: Placement Problem

Welcome to the par.tcl 2026 ML Sys intern challenge! Your task is to solve a placement problem involving standard cells (small blocks) and macros (large blocks). The goal is to minimize the overlap area and the wirelength of the connections between the blocks. A valid placement must ensure that no blocks overlap, but we will look at amount of overlap and wirelength as a metric for how good the placement is.

## Problem Statement

- **Objective:** Place a set of standard cells and macros on a chip layout to minimize overlap and wirelength. We will define overlap as num overlapping cells / num total cells. But you can implement your overlap loss however you think is best. Solving this problem will require designing a good overlap loss, and deciding how to play with hyperparameters and optimizers to get the best results. Nothing is off the table. 
- **Input:** Randomly generated netlists.
- **Output:** Average normalized wirelength and overlap across a set of randomized placements.

## Submission Instructions

1. Fork this repository.
2. Solve the placement problem using your preferred tools or scripts.
3. Run the test script to evaluate your solution and obtain the overlap and wirelength metrics.
4. Submit a pull request with your updated leaderboard entry and instructions for me to access your actual submission (it's fine if it's public).

Note: You can use any libraries or frameworks you like, but please ensure that your code is well-documented and easy to follow.

Also, if you think there are any bugs in the provided code, please feel free to fix them and mention the changes in your submission.


## Leaderboard

| Rank | Name            | Overlap     | Wirelength (um) | Runtime (s) | Notes                |
|------|-----------------|-------------|-----------------|-------------|----------------------|
| 1    | partcl baseline | 0.8         | 0.4             | 5           | Baseline solution    |
| 2    | Mohul Shukla    | 0.0000      | 0.5048          | 54.60s      |                      |

> **To add your results:**  
> Insert a new row in the table above with your name, overlap, wirelength, and any notes.

Good luck!