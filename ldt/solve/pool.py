from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ldt.domains.base import PuzzleInstance
from ldt.lattice import LatticeState, alpha, meet
from ldt.solve.step import LatticeStepResult


@dataclass(frozen=True)
class PoolConfig:
    pool_size: int
    max_age: int = 100

    def __post_init__(self) -> None:
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if self.max_age <= 0:
            raise ValueError("max_age must be positive")


@dataclass(frozen=True)
class PoolEntry:
    puzzle_id: str
    raw: object
    initial_state: LatticeState
    state: LatticeState
    solutions: NDArray[np.int64]
    last_nonempty_target: LatticeState
    age: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_instance(cls, instance: PuzzleInstance) -> PoolEntry:
        if instance.solutions is None:
            raise ValueError("on-policy training pool entries require sampled solutions")
        solutions = np.asarray(instance.solutions, dtype=np.int64).copy()
        abstract = alpha(
            solutions,
            vocab_size=instance.initial_state.vocab_size,
            active=instance.initial_state.active,
        )
        target = meet(instance.initial_state, abstract)
        return cls(
            puzzle_id=instance.puzzle_id,
            raw=instance.raw,
            initial_state=instance.initial_state.copy(),
            state=instance.initial_state.copy(),
            solutions=solutions,
            last_nonempty_target=target,
            metadata=dict(instance.metadata),
        )

    def advanced(
        self,
        state: LatticeState,
        last_nonempty_target: LatticeState | None = None,
    ) -> PoolEntry:
        return PoolEntry(
            puzzle_id=self.puzzle_id,
            raw=self.raw,
            initial_state=self.initial_state,
            state=state,
            solutions=self.solutions,
            last_nonempty_target=last_nonempty_target or self.last_nonempty_target,
            age=self.age + 1,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class PoolBatch:
    indices: NDArray[np.int64]
    entries: tuple[PoolEntry, ...]
    state: LatticeState
    solutions: NDArray[np.int64]


@dataclass(frozen=True)
class PoolUpdateStats:
    sampled: int
    retained: int
    refilled: int
    solved: int
    conflicted: int
    aged_out: int


class OnPolicyTrainingPool:
    """Pool of recent partially-deduced states for on-policy LDT training.

    The pool owns the state distribution. A training loop samples a batch, runs
    the shared step operator with ground-truth solutions, computes its loss, then
    returns the step result here. Terminal or stale entries are replaced by fresh
    puzzle instances; nonterminal entries are kept at their new lattice state.
    """

    def __init__(
        self,
        instances: Sequence[PuzzleInstance],
        config: PoolConfig,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not instances:
            raise ValueError("instances must be non-empty")
        self.config = config
        self._instances = tuple(instances)
        self._rng = rng or np.random.default_rng()
        self._next_instance = 0
        self._entries = [self._fresh_entry() for _ in range(config.pool_size)]
        self.total_refills = config.pool_size
        self.total_retained = 0

    @property
    def entries(self) -> tuple[PoolEntry, ...]:
        return tuple(self._entries)

    def sample(self, batch_size: int) -> PoolBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._entries):
            raise ValueError("batch_size cannot exceed pool size")

        indices = self._rng.choice(len(self._entries), size=batch_size, replace=False)
        entries = tuple(self._entries[int(index)] for index in indices)
        state = _stack_states(tuple(entry.state for entry in entries))
        solutions = _stack_solutions(tuple(entry.solutions for entry in entries))
        return PoolBatch(
            indices=np.asarray(indices, dtype=np.int64),
            entries=entries,
            state=state,
            solutions=solutions,
        )

    def apply_step_result(
        self,
        batch: PoolBatch,
        result: LatticeStepResult,
        next_last_nonempty_target: LatticeState | None = None,
    ) -> PoolUpdateStats:
        batch_size = len(batch.entries)
        if result.state.batch_shape != (batch_size,):
            raise ValueError("step result batch shape must match sampled batch")

        solved = np.asarray(result.solved, dtype=np.bool_).reshape(batch_size)
        conflict = np.asarray(result.conflict, dtype=np.bool_).reshape(batch_size)
        retained = 0
        refilled = 0
        aged_out = 0

        for row_idx, pool_idx in enumerate(batch.indices):
            entry = batch.entries[row_idx]
            next_state = LatticeState(
                result.state.candidates[row_idx],
                result.state.active[row_idx],
            )
            next_target = None
            if next_last_nonempty_target is not None:
                next_target = LatticeState(
                    next_last_nonempty_target.candidates[row_idx],
                    next_last_nonempty_target.active[row_idx],
                )
            advanced = entry.advanced(next_state, next_target)
            is_aged_out = advanced.age >= self.config.max_age
            terminal = bool(solved[row_idx] or conflict[row_idx] or is_aged_out)

            if terminal:
                self._entries[int(pool_idx)] = self._fresh_entry()
                refilled += 1
                aged_out += int(is_aged_out and not solved[row_idx] and not conflict[row_idx])
            else:
                self._entries[int(pool_idx)] = advanced
                retained += 1

        self.total_refills += refilled
        self.total_retained += retained
        return PoolUpdateStats(
            sampled=batch_size,
            retained=retained,
            refilled=refilled,
            solved=int(solved.sum()),
            conflicted=int(conflict.sum()),
            aged_out=aged_out,
        )

    def _fresh_entry(self) -> PoolEntry:
        instance = self._instances[self._next_instance]
        self._next_instance = (self._next_instance + 1) % len(self._instances)
        return PoolEntry.from_instance(instance)


def _stack_states(states: tuple[LatticeState, ...]) -> LatticeState:
    if not states:
        raise ValueError("states must be non-empty")
    candidates = np.stack([state.candidates for state in states], axis=0)
    active = np.stack([state.active for state in states], axis=0)
    return LatticeState(candidates, active)


def _stack_solutions(solutions: tuple[NDArray[np.int64], ...]) -> NDArray[np.int64]:
    if not solutions:
        raise ValueError("solutions must be non-empty")
    max_solutions = max(solution.shape[0] for solution in solutions)
    num_positions = solutions[0].shape[1]
    stacked = np.full((len(solutions), max_solutions, num_positions), -1, dtype=np.int64)
    for row_idx, row_solutions in enumerate(solutions):
        if row_solutions.ndim != 2:
            raise ValueError("solutions must have shape (num_solutions, positions)")
        if row_solutions.shape[1] != num_positions:
            raise ValueError("all solution tensors must share num_positions")
        stacked[row_idx, : row_solutions.shape[0], :] = row_solutions
    return stacked
