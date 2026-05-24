from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ldt.lattice import LatticeState, consistent_solutions


@dataclass(frozen=True)
class LatticeStepConfig:
    """Configuration for the paper's lattice projection step."""

    theta_elim: float = 0.1
    theta_cls: float = 0.6
    tau_decide: float = 1.5
    enable_branching: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.theta_elim <= 1.0:
            raise ValueError("theta_elim must be in [0, 1]")
        if not 0.0 <= self.theta_cls <= 1.0:
            raise ValueError("theta_cls must be in [0, 1]")
        if self.tau_decide <= 0:
            raise ValueError("tau_decide must be positive")


@dataclass(frozen=True)
class LatticeStepResult:
    """Result of threshold elimination, conflict/solved detection, and branching."""

    state: LatticeState
    conflict: NDArray[np.bool_]
    solved: NDArray[np.bool_]
    branched: NDArray[np.bool_]
    branch_position: NDArray[np.int64]
    branch_value: NDArray[np.int64]
    eliminated_candidates: NDArray[np.int64]
    cls_probability: NDArray[np.float64]


def lattice_projection_step(
    state: LatticeState,
    candidate_logits: Any,
    cls_logits: Any,
    *,
    config: LatticeStepConfig | None = None,
    rng: np.random.Generator | None = None,
    solutions: NDArray[np.integer[Any]] | np.ndarray | None = None,
    inactive_value: int = -1,
) -> LatticeStepResult:
    """Apply the paper's shared lattice projection step.

    This implements Algorithm 2's non-loss part: threshold elimination,
    conflict/solved flags, and stochastic singleton branching. When `solutions`
    are provided, conflict and solved are verified against those concrete
    solutions as in training; otherwise inference-style CLS plus empty-cell
    conflict detection is used.
    """

    step_config = config or LatticeStepConfig()
    generator = rng or np.random.default_rng()

    logits = _as_float_array(candidate_logits)
    cls = _as_float_array(cls_logits)
    _check_logit_shapes(state, logits, cls)

    batch_shape = state.batch_shape
    flat_state, reshape = _flatten_state(state)
    flat_logits = logits.reshape((-1, state.num_positions, state.vocab_size))
    flat_cls = cls.reshape((-1,))

    probabilities = _sigmoid(flat_logits)
    keep_mask = probabilities >= step_config.theta_elim
    after_elim = flat_state & keep_mask

    eliminated = (flat_state & ~after_elim).sum(axis=(-2, -1), dtype=np.int64)
    conflict = _sigmoid(flat_cls) > step_config.theta_cls
    empty_cell_conflict = ((after_elim.sum(axis=-1) == 0) & reshape.active).any(axis=-1)
    conflict = conflict | empty_cell_conflict
    solved = ((after_elim.sum(axis=-1) == 1) | ~reshape.active).all(axis=-1) & ~conflict

    if solutions is not None:
        conflict, solved = _verify_against_solutions(
            after_elim,
            reshape.active,
            solutions,
            inactive_value,
        )

    branched = np.zeros(after_elim.shape[0], dtype=np.bool_)
    branch_position = np.full(after_elim.shape[0], -1, dtype=np.int64)
    branch_value = np.full(after_elim.shape[0], -1, dtype=np.int64)

    if step_config.enable_branching:
        for row_idx in range(after_elim.shape[0]):
            if conflict[row_idx] or solved[row_idx]:
                continue
            position = _sample_branch_position(
                after_elim[row_idx],
                reshape.active[row_idx],
                generator,
            )
            if position is None:
                continue
            value = _sample_branch_value(
                after_elim[row_idx, position],
                flat_logits[row_idx, position],
                step_config.tau_decide,
                generator,
            )
            after_elim[row_idx, position, :] = False
            after_elim[row_idx, position, value] = True
            branched[row_idx] = True
            branch_position[row_idx] = position
            branch_value[row_idx] = value

    next_state = LatticeState(
        after_elim.reshape(state.candidates.shape),
        state.active,
    )

    return LatticeStepResult(
        state=next_state,
        conflict=conflict.reshape(batch_shape),
        solved=solved.reshape(batch_shape),
        branched=branched.reshape(batch_shape),
        branch_position=branch_position.reshape(batch_shape),
        branch_value=branch_value.reshape(batch_shape),
        eliminated_candidates=eliminated.reshape(batch_shape),
        cls_probability=_sigmoid(flat_cls).reshape(batch_shape),
    )


@dataclass(frozen=True)
class _FlatState:
    candidates: NDArray[np.bool_]
    active: NDArray[np.bool_]


def _flatten_state(state: LatticeState) -> tuple[NDArray[np.bool_], _FlatState]:
    flat_candidates = state.candidates.reshape((-1, state.num_positions, state.vocab_size)).copy()
    flat_active = state.active.reshape((-1, state.num_positions)).copy()
    return flat_candidates, _FlatState(flat_candidates, flat_active)


def _as_float_array(value: Any) -> NDArray[np.float64]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _check_logit_shapes(
    state: LatticeState,
    candidate_logits: NDArray[np.float64],
    cls_logits: NDArray[np.float64],
) -> None:
    if candidate_logits.shape != state.candidates.shape:
        raise ValueError(
            "candidate_logits must match state.candidates shape "
            f"{state.candidates.shape}, got {candidate_logits.shape}"
        )
    expected_cls_shape = state.batch_shape
    if expected_cls_shape == ():
        if cls_logits.shape not in {(), (1,)}:
            raise ValueError("cls_logits must be scalar for an unbatched state")
    elif cls_logits.shape != expected_cls_shape:
        raise ValueError(f"cls_logits must have shape {expected_cls_shape}, got {cls_logits.shape}")


def _verify_against_solutions(
    candidates: NDArray[np.bool_],
    active: NDArray[np.bool_],
    solutions: NDArray[np.integer[Any]] | np.ndarray,
    inactive_value: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    conflict = np.zeros(candidates.shape[0], dtype=np.bool_)
    solved = np.zeros(candidates.shape[0], dtype=np.bool_)
    solution_array = np.asarray(solutions)
    if solution_array.ndim == 2:
        per_row_solutions = [solution_array for _ in range(candidates.shape[0])]
    elif solution_array.ndim == 3 and solution_array.shape[0] == candidates.shape[0]:
        per_row_solutions = [solution_array[row_idx] for row_idx in range(candidates.shape[0])]
    else:
        raise ValueError(
            "solutions must have shape (num_solutions, positions) or "
            "(batch, num_solutions, positions)"
        )

    for row_idx, row_solutions in enumerate(per_row_solutions):
        row_state = LatticeState(candidates[row_idx], active[row_idx])
        consistent = consistent_solutions(
            row_state,
            row_solutions,
            inactive_value=inactive_value,
        )
        conflict[row_idx] = not bool(consistent.any())
        solved[row_idx] = (
            not conflict[row_idx]
            and bool(row_state.is_complete())
            and _matches_any_solution(row_state, row_solutions[consistent], active[row_idx])
        )
    return conflict, solved


def _matches_any_solution(
    state: LatticeState,
    solutions: NDArray[np.integer[Any]],
    active: NDArray[np.bool_],
) -> bool:
    if solutions.size == 0:
        return False
    decoded = state.as_solution_indices()
    return bool(np.any(np.all(solutions[:, active] == decoded[active], axis=1)))


def _sample_branch_position(
    candidates: NDArray[np.bool_],
    active: NDArray[np.bool_],
    rng: np.random.Generator,
) -> int | None:
    multicandidate_positions = np.flatnonzero((candidates.sum(axis=-1) >= 2) & active)
    if multicandidate_positions.size == 0:
        return None
    return int(rng.choice(multicandidate_positions))


def _sample_branch_value(
    alive_candidates: NDArray[np.bool_],
    logits: NDArray[np.float64],
    temperature: float,
    rng: np.random.Generator,
) -> int:
    live_values = np.flatnonzero(alive_candidates)
    if live_values.size == 0:
        raise ValueError("cannot branch with no alive candidates")
    live_logits = logits[live_values] / temperature
    probabilities = _softmax(live_logits)
    return int(rng.choice(live_values, p=probabilities))


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    positive = values >= 0
    negative = ~positive
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[negative])
    result[negative] = exp_values / (1.0 + exp_values)
    return result


def _softmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()
