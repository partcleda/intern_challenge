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
| 1    | Brayden Rudisill  | 0.0000      | 0.2611          | 50.51       |  Timed on a mac air |
| 2    | Neil Teje         | 0.0000  | 0.2700          | 24.00s      |                                      |
| 3    | Leison Gao      | 0.0000      | 0.2796          | 50.14s      |                      |
| 4    | William Pan     | 0.0000      | 0.2848          | 155.33s     |                      |
| 5    | Ashmit Dutta    | 0.0000      | 0.2870          | 995.58      |  Spent my entire morning (12 am - 6 am) doing this :P       |
| 6    | Gabriel Del Monte  | 0.0000      | 0.3427          | 606.07      |                                                              |
| 7    | Aleksey  Valouev| 0.0000      | 0.3577          | 118.98      |                      |
| 8    | Shashank Shriram| 0.0000      | 0.4634          |   7.08      | 🏎️                    |         
| 9    | Mohul Shukla    | 0.0000      | 0.5048          | 54.60s      |                      |
| 10    | Ryan Hulke      | 0.0000      | 0.5226          | 166.24      |                      |
| 11    | Neel  Shah      | 0.0000      | 0.5445          | 45.40       |  Zero overlaps on all tests, adaptive schedule + early stop |
| 12   | Nawel Asgar    | 0.0000     | 0.5675          | 81.49      | Adaptive penalty scaling with cubic gradients and design-size optimization      |
| 12   | Shiva Baghel.     | 0.0000     | 0.5885          | 491.00      | Stable zero-overlap with balanced optimization      |
| 13   | Vansh Jain      | 0.0000      | 0.9352          | 86.36       |                      |
| 14    | Akash Pai       | 0.0006      | 0.4933          | 326.25s     |                      |
| 15    | Zade Mahayni     | 0.00665     | 0.5157          |  127.4     | Will try again tomorrow |
| 16    | Nithin Yanna    | 0.0148      | 0.5034          | 247.30s     | aggressive overlap penalty with quadratic scaling |
| 17    | Sean Ko         | 0.0271      |  .5138          | 31.83s      | lr increase, decrease epoch, increase lambda overlap and decreased lambda wire_length + log penalty loss |  
| 18    | Prithvi Seran   | 0.0499      | 0.4890          | 398.58      |                      |
| 19    | partcl example  | 0.8         | 0.4             | 5           | example              |
| 20    | Add Yours!      |             |                 |             |                      |

> **To add your results:**  
> Insert a new row in the table above with your name, overlap, wirelength, and any notes. Ensure you sort by overlap.

Good luck!
