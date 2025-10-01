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


## Leaderboard (sorted by overlap)

| Rank | Name            | Overlap     | Wirelength (um) | Runtime (s) | Notes                |
|------|-----------------|-------------|-----------------|-------------|----------------------|
| 1    | Neel  Shah      | 0.0000      | 0.5445          | 45.40       |  Zero overlaps on all tests, adaptive schedule + early stop |
| 2    | Akash Pai       | 0.0006      | 0.4933          | 326.25s     |                      |
| 3    | Prithvi Seran   | 0.0499      | 0.4890          | 398.58      |                      |
| 4    | partcl example  | 0.8         | 0.4             | 5           | example              |
| 5    | Add Yours!      |             |                 |             |                      |

> **To add your results:**  
> Insert a new row in the table above with your name, overlap, wirelength, and any notes. Ensure you sort by overlap.

Good luck!
