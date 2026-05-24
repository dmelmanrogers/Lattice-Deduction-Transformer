from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ldt.domains.base import DomainSpec, PuzzleInstance, ValidationResult
from ldt.lattice import LatticeState

SNOWFLAKE_ROWS = 15
SNOWFLAKE_COLS = 10
SNOWFLAKE_POSITIONS = SNOWFLAKE_ROWS * SNOWFLAKE_COLS
SNOWFLAKE_VOCAB = 6


@dataclass(frozen=True)
class SnowflakePuzzle:
    givens: NDArray[np.int64]
    groups: tuple[tuple[int, ...], ...]
    active: NDArray[np.bool_]
    solution: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        givens = _as_vector(self.givens, name="givens", allow_zero=True)
        active = np.asarray(self.active, dtype=np.bool_)
        if active.shape != (SNOWFLAKE_POSITIONS,):
            raise ValueError("active must have shape (150,)")
        solution = None
        if self.solution is not None:
            solution = _as_vector(self.solution, name="solution", allow_zero=True)
            if np.any(solution[active] < 1):
                raise ValueError("active solution values must be in 1..6")
        normalized_groups = tuple(
            tuple(int(position) for position in group) for group in self.groups
        )
        for group in normalized_groups:
            for position in group:
                if not 0 <= position < SNOWFLAKE_POSITIONS:
                    raise ValueError("group position out of range")
        object.__setattr__(self, "givens", givens)
        object.__setattr__(self, "groups", normalized_groups)
        object.__setattr__(self, "active", active.copy())
        object.__setattr__(self, "solution", solution)


class SnowflakeDomain:
    """Snowflake Sudoku benchmark adapter over a fixed 15x10 covering grid."""

    def __init__(self, active: NDArray[np.bool_] | None = None) -> None:
        active_array = (
            np.ones(SNOWFLAKE_POSITIONS, dtype=np.bool_)
            if active is None
            else np.asarray(active, dtype=np.bool_)
        )
        self._spec = DomainSpec(
            name="snowflake",
            num_positions=SNOWFLAKE_POSITIONS,
            vocab_size=SNOWFLAKE_VOCAB,
            active=active_array,
            position_shape=(SNOWFLAKE_ROWS, SNOWFLAKE_COLS),
            metadata={"extra_input_channels": 1},
        )

    @property
    def spec(self) -> DomainSpec:
        return self._spec

    def encode(self, puzzle: SnowflakePuzzle, *, puzzle_id: str = "") -> PuzzleInstance:
        candidates = np.ones((SNOWFLAKE_POSITIONS, SNOWFLAKE_VOCAB), dtype=np.bool_)
        candidates[~puzzle.active, :] = False
        for position, digit in enumerate(puzzle.givens):
            if not puzzle.active[position] or digit == 0:
                continue
            candidates[position, :] = False
            candidates[position, digit - 1] = True

        solutions = None
        if puzzle.solution is not None:
            solution = np.full(SNOWFLAKE_POSITIONS, -1, dtype=np.int64)
            solution[puzzle.active] = puzzle.solution[puzzle.active] - 1
            solutions = solution.reshape(1, SNOWFLAKE_POSITIONS)

        return PuzzleInstance(
            puzzle_id=puzzle_id,
            initial_state=LatticeState(candidates, puzzle.active),
            solutions=solutions,
            raw=puzzle,
        )

    def extra_features(self, puzzle: SnowflakePuzzle, state: LatticeState) -> NDArray[np.float32]:
        del state
        return puzzle.active.astype(np.float32).reshape(SNOWFLAKE_POSITIONS, 1)

    def solution_to_state(self, solution: NDArray[np.integer]) -> LatticeState:
        vector = _as_vector(solution, name="solution", allow_zero=True)
        active = self.spec.active
        if np.any(vector[active] < 1):
            raise ValueError("active solution values must be in 1..6")
        candidates = np.zeros((SNOWFLAKE_POSITIONS, SNOWFLAKE_VOCAB), dtype=np.bool_)
        for position, digit in enumerate(vector):
            if active[position]:
                candidates[position, int(digit) - 1] = True
        return LatticeState(candidates, active)

    def decode_solution(self, state: LatticeState) -> NDArray[np.int64]:
        indices = state.as_solution_indices()
        decoded = np.zeros(SNOWFLAKE_POSITIONS, dtype=np.int64)
        decoded[state.active] = indices[state.active] + 1
        return decoded

    def validate_solution(self, puzzle: SnowflakePuzzle, state: LatticeState) -> ValidationResult:
        if bool(state.is_conflict()):
            return ValidationResult(False, "state is conflicted")
        if not bool(state.is_complete()):
            return ValidationResult(False, "state is not complete")
        grid = self.decode_solution(state)
        if np.any(grid[puzzle.active] < 1) or np.any(grid[puzzle.active] > SNOWFLAKE_VOCAB):
            return ValidationResult(False, "active values are outside 1..6")
        given_mask = (puzzle.givens != 0) & puzzle.active
        if np.any(grid[given_mask] != puzzle.givens[given_mask]):
            return ValidationResult(False, "solution violates puzzle givens")
        for idx, group in enumerate(puzzle.groups):
            values = [int(grid[position]) for position in group if puzzle.active[position]]
            if len(values) != len(set(values)):
                return ValidationResult(False, f"group {idx} has duplicate values")
            if len(values) == SNOWFLAKE_VOCAB and sorted(values) != list(range(1, 7)):
                return ValidationResult(False, f"group {idx} is not a 1..6 permutation")
        if puzzle.solution is not None and not np.array_equal(
            grid[puzzle.active],
            puzzle.solution[puzzle.active],
        ):
            return ValidationResult(False, "solution does not match provided reference solution")
        return ValidationResult(True)


def make_snowflake_puzzle(
    givens: Iterable[int],
    groups: Iterable[Iterable[int]],
    *,
    active: Iterable[bool] | None = None,
    solution: Iterable[int] | None = None,
) -> SnowflakePuzzle:
    active_array = (
        np.ones(SNOWFLAKE_POSITIONS, dtype=np.bool_)
        if active is None
        else np.asarray(list(active), dtype=np.bool_)
    )
    solution_array = None if solution is None else np.asarray(list(solution), dtype=np.int64)
    return SnowflakePuzzle(
        np.asarray(list(givens), dtype=np.int64),
        tuple(tuple(group) for group in groups),
        active_array,
        solution_array,
    )


def _as_vector(
    values: NDArray[np.integer] | np.ndarray,
    *,
    name: str,
    allow_zero: bool,
) -> NDArray[np.int64]:
    vector = np.asarray(values, dtype=np.int64)
    if vector.shape != (SNOWFLAKE_POSITIONS,):
        raise ValueError(f"{name} must have shape (150,)")
    lower = 0 if allow_zero else 1
    if np.any(vector < lower) or np.any(vector > SNOWFLAKE_VOCAB):
        value_range = "0..6" if allow_zero else "1..6"
        raise ValueError(f"{name} values must be in {value_range}")
    return vector.copy()
