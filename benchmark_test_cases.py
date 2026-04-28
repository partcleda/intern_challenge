"""Shared benchmark configurations for placement test cases."""

ACTIVE_TEST_CASES = [
    # Small designs
    (1, 2, 20, 1001),
    (2, 3, 25, 1002),
    (3, 2, 30, 1003),
    # Medium designs
    (4, 3, 50, 1004),
    (5, 4, 75, 1005),
    (6, 5, 100, 1006),
    # Large designs
    (7, 5, 150, 1007),
    (8, 7, 150, 1008),
    (9, 8, 200, 1009),
    (10, 10, 2000, 1010),
]

OPTIONAL_TEST_CASES = [
    # Realistic designs
    (11, 10, 10000, 1011),
    (12, 10, 100000, 1012),
]

TEST_CASES = ACTIVE_TEST_CASES + OPTIONAL_TEST_CASES

TEST_CASES_BY_ID = {
    test_id: {
        "test_id": test_id,
        "num_macros": num_macros,
        "num_std_cells": num_std_cells,
        "seed": seed,
    }
    for test_id, num_macros, num_std_cells, seed in TEST_CASES
}
