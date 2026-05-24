from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ldt.lattice.state import LatticeState


def meet(left: LatticeState, right: LatticeState) -> LatticeState:
    """Greatest lower bound under subset ordering: candidate intersection."""

    left.require_compatible(right)
    return LatticeState(left.candidates & right.candidates, left.active)


def join(left: LatticeState, right: LatticeState) -> LatticeState:
    """Least upper bound under subset ordering: candidate union."""

    left.require_compatible(right)
    return LatticeState(left.candidates | right.candidates, left.active)


def top_like(state: LatticeState) -> LatticeState:
    candidates = np.broadcast_to(state.active[..., :, None], state.candidates.shape)
    return LatticeState(candidates.copy(), state.active)


def bottom_like(state: LatticeState) -> LatticeState:
    return LatticeState(np.zeros_like(state.candidates, dtype=np.bool_), state.active)


def is_leq(left: LatticeState, right: LatticeState) -> NDArray[np.bool_] | np.bool_:
    """Return whether `left <= right`, meaning every left candidate is in right."""

    left.require_compatible(right)
    extra = left.candidates & ~right.candidates
    return ~extra.any(axis=(-2, -1))


def alpha(
    solutions: NDArray[np.integer[Any]] | np.ndarray,
    *,
    vocab_size: int,
    active: NDArray[np.bool_] | np.ndarray | None = None,
    inactive_value: int = -1,
) -> LatticeState:
    """Abstract concrete solutions into the powerset lattice by candidate union.

    `solutions` has shape `(num_solutions, positions)` and contains zero-based
    candidate indices. Inactive positions may use `inactive_value`.
    """

    solution_array = np.asarray(solutions)
    if solution_array.ndim != 2:
        raise ValueError("solutions must have shape (num_solutions, positions)")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    num_solutions, positions = solution_array.shape
    if active is None:
        active_array = np.ones(positions, dtype=np.bool_)
    else:
        active_array = np.asarray(active, dtype=np.bool_)
        if active_array.shape != (positions,):
            raise ValueError("active must have shape (positions,)")

    candidates = np.zeros((positions, vocab_size), dtype=np.bool_)
    for solution_idx in range(num_solutions):
        for position_idx in range(positions):
            if not active_array[position_idx]:
                continue
            value = int(solution_array[solution_idx, position_idx])
            if value == inactive_value:
                continue
            if not 0 <= value < vocab_size:
                raise ValueError(
                    f"solution value {value} at solution {solution_idx}, position {position_idx} "
                    f"is outside [0, {vocab_size})"
                )
            candidates[position_idx, value] = True
    return LatticeState(candidates, active_array)


def consistent_solutions(
    state: LatticeState,
    solutions: NDArray[np.integer[Any]] | np.ndarray,
    *,
    inactive_value: int = -1,
) -> NDArray[np.bool_]:
    """Return a mask for concrete solutions that are still represented by `state`."""

    if state.batch_shape:
        raise ValueError("consistent_solutions expects an unbatched lattice state")

    solution_array = np.asarray(solutions)
    if solution_array.ndim != 2 or solution_array.shape[1] != state.num_positions:
        raise ValueError("solutions must have shape (num_solutions, state.num_positions)")

    consistent = np.ones(solution_array.shape[0], dtype=np.bool_)
    for solution_idx, solution in enumerate(solution_array):
        for position_idx, active in enumerate(state.active):
            if not active:
                continue
            value = int(solution[position_idx])
            if value == inactive_value:
                consistent[solution_idx] = False
                break
            if not 0 <= value < state.vocab_size or not state.candidates[position_idx, value]:
                consistent[solution_idx] = False
                break
    return consistent


def pin_candidate(state: LatticeState, position: int, value: int) -> LatticeState:
    """Restrict one active position to a single currently-live candidate."""

    if state.batch_shape:
        raise ValueError("pin_candidate expects an unbatched lattice state")
    if not 0 <= position < state.num_positions:
        raise IndexError("position out of range")
    if not 0 <= value < state.vocab_size:
        raise IndexError("candidate value out of range")
    if not bool(state.active[position]):
        raise ValueError("cannot pin an inactive position")
    if not bool(state.candidates[position, value]):
        raise ValueError("cannot pin a candidate that is not live in the lattice state")

    candidates = state.candidates.copy()
    candidates[position, :] = False
    candidates[position, value] = True
    return LatticeState(candidates, state.active)


def eliminate_candidates(
    state: LatticeState,
    keep_mask: NDArray[np.bool_] | np.ndarray,
    *,
    preserve_at_least_one: bool = False,
) -> LatticeState:
    """Apply a model-produced keep mask to the lattice state.

    If `preserve_at_least_one` is true, positions where all live candidates would
    be removed are left unchanged. The paper's solver uses conflict detection, so
    callers should opt into preservation only for diagnostics or ablations.
    """

    keep_array = np.asarray(keep_mask, dtype=np.bool_)
    if keep_array.shape != state.candidates.shape:
        raise ValueError("keep_mask must match state.candidates shape")

    candidates = state.candidates & keep_array
    if preserve_at_least_one:
        old_counts = state.candidate_counts()
        new_counts = candidates.sum(axis=-1)
        restore = (old_counts > 0) & (new_counts == 0) & state.active
        candidates[restore, :] = state.candidates[restore, :]
    return LatticeState(candidates, state.active)
