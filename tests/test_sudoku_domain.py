from __future__ import annotations

import numpy as np

from ldt.domains.sudoku import (
    DihedralTransform,
    SudokuDomain,
    SudokuPuzzle,
    apply_digit_permutation,
    transform_grid,
)

GIVENS = (
    "530070000600195000098000060800060003400803001700020006"
    "060000280000419005000080079"
)

SOLUTION = (
    "534678912672195348198342567859761423426853791713924856"
    "961537284287419635345286179"
)


def test_encode_sets_singletons_for_givens_and_full_candidates_for_blanks() -> None:
    domain = SudokuDomain()
    instance = domain.encode(f"{GIVENS}\n{SOLUTION}", puzzle_id="easy")
    state = instance.initial_state

    assert state.candidates.shape == (81, 9)
    assert np.array_equal(
        state.candidates[0],
        [False, False, False, False, True, False, False, False, False],
    )
    assert np.all(state.candidates[2])
    assert instance.solutions is not None
    assert instance.solutions.shape == (1, 81)
    assert instance.solutions[0, 0] == 4


def test_validate_accepts_complete_valid_solution() -> None:
    domain = SudokuDomain()
    state = domain.solution_to_state(SOLUTION)

    result = domain.validate_solution(GIVENS, state)

    assert result.valid


def test_validate_rejects_givens_mismatch() -> None:
    domain = SudokuDomain()
    bad_solution = "1" + SOLUTION[1:]
    state = domain.solution_to_state(bad_solution)

    result = domain.validate_solution(GIVENS, state)

    assert not result.valid
    assert result.reason == "solution violates puzzle givens"


def test_validate_rejects_invalid_completed_grid() -> None:
    domain = SudokuDomain()
    invalid = np.ones((9, 9), dtype=np.int64)
    puzzle = SudokuPuzzle(np.zeros((9, 9), dtype=np.int64))
    state = domain.solution_to_state(invalid)

    result = domain.validate_solution(puzzle, state)

    assert not result.valid
    assert result.reason == "row 0 is invalid"


def test_dihedral_transform_round_trips_with_inverse() -> None:
    grid = np.arange(81, dtype=np.int64).reshape(9, 9) % 10

    rotated = transform_grid(grid, DihedralTransform.ROT90)
    restored = transform_grid(rotated, DihedralTransform.ROT270)

    assert np.array_equal(restored, grid)


def test_digit_permutation_preserves_zero_and_maps_digits() -> None:
    grid = np.asarray(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [9, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 0, 1, 2, 3, 4, 5, 6],
            [7, 8, 9, 0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 0, 1, 2],
            [3, 4, 5, 6, 7, 8, 9, 0, 1],
            [2, 3, 4, 5, 6, 7, 8, 9, 0],
        ],
        dtype=np.int64,
    )
    permutation = [9, 8, 7, 6, 5, 4, 3, 2, 1]

    permuted = apply_digit_permutation(grid, permutation)

    assert permuted[0, 0] == 0
    assert permuted[0, 1] == 9
    assert permuted[0, 8] == 2
