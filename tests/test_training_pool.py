from __future__ import annotations

import numpy as np

from ldt.domains.base import PuzzleInstance
from ldt.lattice import LatticeState
from ldt.solve import LatticeStepResult, OnPolicyTrainingPool, PoolConfig


def _instance(puzzle_id: str, solutions: list[list[int]]) -> PuzzleInstance:
    return PuzzleInstance(
        puzzle_id=puzzle_id,
        initial_state=LatticeState(np.ones((2, 2), dtype=np.bool_)),
        solutions=np.asarray(solutions, dtype=np.int64),
        raw={"id": puzzle_id},
    )


def _step_result(
    batch_state: LatticeState,
    solved: list[bool],
    conflict: list[bool],
) -> LatticeStepResult:
    batch_size = len(solved)
    return LatticeStepResult(
        state=batch_state,
        conflict=np.asarray(conflict, dtype=np.bool_),
        solved=np.asarray(solved, dtype=np.bool_),
        branched=np.zeros(batch_size, dtype=np.bool_),
        branch_position=np.full(batch_size, -1, dtype=np.int64),
        branch_value=np.full(batch_size, -1, dtype=np.int64),
        eliminated_candidates=np.zeros(batch_size, dtype=np.int64),
        cls_probability=np.zeros(batch_size, dtype=np.float64),
    )


def test_pool_samples_batched_states_and_pads_solution_sets() -> None:
    pool = OnPolicyTrainingPool(
        [
            _instance("one", [[0, 1]]),
            _instance("two", [[1, 0], [0, 1]]),
        ],
        PoolConfig(pool_size=2),
        rng=np.random.default_rng(0),
    )

    batch = pool.sample(2)

    assert batch.state.candidates.shape == (2, 2, 2)
    assert batch.solutions.shape == (2, 2, 2)
    assert np.any(np.all(batch.solutions == -1, axis=-1))


def test_pool_retains_nonterminal_entries_and_refills_terminal_entries() -> None:
    pool = OnPolicyTrainingPool(
        [_instance("one", [[0, 1]]), _instance("two", [[1, 0]]), _instance("three", [[1, 1]])],
        PoolConfig(pool_size=2, max_age=10),
        rng=np.random.default_rng(1),
    )
    batch = pool.sample(2)

    stats = pool.apply_step_result(batch, _step_result(batch.state, [True, False], [False, False]))

    assert stats.sampled == 2
    assert stats.solved == 1
    assert stats.refilled == 1
    assert stats.retained == 1
    ages = [entry.age for entry in pool.entries]
    assert sorted(ages) == [0, 1]
    assert len(pool.entries) == 2


def test_pool_refills_entries_that_reach_max_age() -> None:
    pool = OnPolicyTrainingPool(
        [_instance("one", [[0, 1]]), _instance("two", [[1, 0]])],
        PoolConfig(pool_size=1, max_age=1),
        rng=np.random.default_rng(2),
    )
    batch = pool.sample(1)

    stats = pool.apply_step_result(batch, _step_result(batch.state, [False], [False]))

    assert stats.aged_out == 1
    assert stats.refilled == 1
    assert pool.entries[0].age == 0
