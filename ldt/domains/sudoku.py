from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from ldt.domains.base import DomainSpec, PuzzleInstance, ValidationResult
from ldt.lattice import LatticeState

GRID_SIZE = 9
BOX_SIZE = 3
NUM_CELLS = GRID_SIZE * GRID_SIZE
VOCAB_SIZE = 9


class DihedralTransform(StrEnum):
    IDENTITY = "identity"
    ROT90 = "rot90"
    ROT180 = "rot180"
    ROT270 = "rot270"
    FLIP_HORIZONTAL = "flip_horizontal"
    FLIP_VERTICAL = "flip_vertical"
    TRANSPOSE = "transpose"
    ANTI_TRANSPOSE = "anti_transpose"


@dataclass(frozen=True)
class SudokuPuzzle:
    givens: NDArray[np.int64]
    solution: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        givens = _as_grid(self.givens, name="givens", allow_zero=True)
        solution = None
        if self.solution is not None:
            solution = _as_grid(self.solution, name="solution", allow_zero=False)
        object.__setattr__(self, "givens", givens)
        object.__setattr__(self, "solution", solution)


class SudokuDomain:
    """Sudoku-Extreme benchmark adapter.

    This adapter encodes/validates Sudoku instances. It intentionally does not
    implement Sudoku search or symbolic deduction; the LDT core sees only the
    lattice state, optional sampled solutions, and validator.
    """

    def __init__(self) -> None:
        self._spec = DomainSpec(
            name="sudoku",
            num_positions=NUM_CELLS,
            vocab_size=VOCAB_SIZE,
            active=np.ones(NUM_CELLS, dtype=np.bool_),
            position_shape=(GRID_SIZE, GRID_SIZE),
        )

    @property
    def spec(self) -> DomainSpec:
        return self._spec

    def encode(self, puzzle: SudokuPuzzle | str, *, puzzle_id: str = "") -> PuzzleInstance:
        parsed = parse_puzzle(puzzle)
        candidates = np.ones((NUM_CELLS, VOCAB_SIZE), dtype=np.bool_)
        flat_givens = parsed.givens.reshape(NUM_CELLS)
        for position, digit in enumerate(flat_givens):
            if digit == 0:
                continue
            candidates[position, :] = False
            candidates[position, digit - 1] = True

        solutions = None
        if parsed.solution is not None:
            solutions = (parsed.solution.reshape(1, NUM_CELLS) - 1).astype(np.int64)

        return PuzzleInstance(
            puzzle_id=puzzle_id,
            initial_state=LatticeState(candidates, self.spec.active),
            solutions=solutions,
            raw=parsed,
        )

    def solution_to_state(self, solution: NDArray[np.integer] | str) -> LatticeState:
        grid = _parse_grid_string(solution) if isinstance(solution, str) else solution
        solution_grid = _as_grid(grid, name="solution", allow_zero=False)
        candidates = np.zeros((NUM_CELLS, VOCAB_SIZE), dtype=np.bool_)
        for position, digit in enumerate(solution_grid.reshape(NUM_CELLS)):
            candidates[position, int(digit) - 1] = True
        return LatticeState(candidates, self.spec.active)

    def decode_solution(self, state: LatticeState) -> NDArray[np.int64]:
        empty = LatticeState(np.zeros((NUM_CELLS, VOCAB_SIZE), dtype=np.bool_), self.spec.active)
        state.require_compatible(empty)
        values = state.as_solution_indices().reshape(GRID_SIZE, GRID_SIZE)
        return values + 1

    def validate_solution(
        self,
        puzzle: SudokuPuzzle | str,
        state: LatticeState,
    ) -> ValidationResult:
        parsed = parse_puzzle(puzzle)
        if bool(state.is_conflict()):
            return ValidationResult(False, "state is conflicted")
        if not bool(state.is_complete()):
            return ValidationResult(False, "state is not complete")

        try:
            grid = self.decode_solution(state)
        except ValueError as exc:
            return ValidationResult(False, str(exc))

        given_mask = parsed.givens != 0
        if np.any(grid[given_mask] != parsed.givens[given_mask]):
            return ValidationResult(False, "solution violates puzzle givens")

        if parsed.solution is not None and not np.array_equal(grid, parsed.solution):
            return ValidationResult(False, "solution does not match provided reference solution")

        for idx, row in enumerate(grid):
            if not _is_unit(row):
                return ValidationResult(False, f"row {idx} is invalid")
        for idx, column in enumerate(grid.T):
            if not _is_unit(column):
                return ValidationResult(False, f"column {idx} is invalid")
        for box_row in range(0, GRID_SIZE, BOX_SIZE):
            for box_col in range(0, GRID_SIZE, BOX_SIZE):
                box = grid[box_row : box_row + BOX_SIZE, box_col : box_col + BOX_SIZE].reshape(-1)
                if not _is_unit(box):
                    box_id = (box_row // BOX_SIZE, box_col // BOX_SIZE)
                    return ValidationResult(False, f"box {box_id} is invalid")

        return ValidationResult(True)


def parse_puzzle(puzzle: SudokuPuzzle | str) -> SudokuPuzzle:
    if isinstance(puzzle, SudokuPuzzle):
        return puzzle

    parts = [part.strip() for part in puzzle.strip().splitlines() if part.strip()]
    if len(parts) == 1:
        return SudokuPuzzle(_parse_grid_string(parts[0]))
    if len(parts) == 2:
        return SudokuPuzzle(_parse_grid_string(parts[0]), _parse_grid_string(parts[1]))
    raise ValueError("Sudoku puzzle string must contain one givens row or givens plus solution")


def transform_grid(grid: NDArray[np.integer], transform: DihedralTransform) -> NDArray[np.int64]:
    array = _as_grid(grid, name="grid", allow_zero=True)
    match transform:
        case DihedralTransform.IDENTITY:
            transformed = array
        case DihedralTransform.ROT90:
            transformed = np.rot90(array, k=1)
        case DihedralTransform.ROT180:
            transformed = np.rot90(array, k=2)
        case DihedralTransform.ROT270:
            transformed = np.rot90(array, k=3)
        case DihedralTransform.FLIP_HORIZONTAL:
            transformed = np.fliplr(array)
        case DihedralTransform.FLIP_VERTICAL:
            transformed = np.flipud(array)
        case DihedralTransform.TRANSPOSE:
            transformed = array.T
        case DihedralTransform.ANTI_TRANSPOSE:
            transformed = np.fliplr(np.flipud(array)).T
        case _:
            raise ValueError(f"unknown dihedral transform: {transform}")
    return np.asarray(transformed, dtype=np.int64)


def apply_digit_permutation(
    grid: NDArray[np.integer],
    permutation: Iterable[int],
    *,
    keep_zero: bool = True,
) -> NDArray[np.int64]:
    array = _as_grid(grid, name="grid", allow_zero=keep_zero)
    perm = np.asarray(list(permutation), dtype=np.int64)
    if perm.shape != (VOCAB_SIZE,):
        raise ValueError("permutation must contain exactly 9 digits")
    if sorted(int(value) for value in perm) != list(range(1, 10)):
        raise ValueError("permutation must be a rearrangement of digits 1..9")

    result = array.copy()
    for digit in range(1, 10):
        result[array == digit] = perm[digit - 1]
    return result


def _parse_grid_string(text: str) -> NDArray[np.int64]:
    chars = [char for char in text if char in "0123456789."]
    if len(chars) != NUM_CELLS:
        raise ValueError("Sudoku grid string must contain exactly 81 digit/dot cells")
    values = [0 if char in "0." else int(char) for char in chars]
    return np.asarray(values, dtype=np.int64).reshape(GRID_SIZE, GRID_SIZE)


def _as_grid(
    grid: NDArray[np.integer] | np.ndarray,
    *,
    name: str,
    allow_zero: bool,
) -> NDArray[np.int64]:
    array = np.asarray(grid, dtype=np.int64)
    if array.shape == (NUM_CELLS,):
        array = array.reshape(GRID_SIZE, GRID_SIZE)
    if array.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"{name} must have shape (9, 9) or (81,)")
    lower_bound = 0 if allow_zero else 1
    if np.any(array < lower_bound) or np.any(array > 9):
        range_text = "0..9" if allow_zero else "1..9"
        raise ValueError(f"{name} entries must be in {range_text}")
    return array.astype(np.int64, copy=True)


def _is_unit(values: NDArray[np.integer]) -> bool:
    return sorted(int(value) for value in values) == list(range(1, 10))
