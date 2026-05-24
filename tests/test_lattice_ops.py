from __future__ import annotations

import itertools

import numpy as np
import pytest

from ldt.lattice import LatticeState, alpha, consistent_solutions, join, meet, pin_candidate
from ldt.lattice.ops import bottom_like, eliminate_candidates, is_leq, top_like


def _all_states(num_positions: int = 2, vocab_size: int = 2) -> list[LatticeState]:
    states: list[LatticeState] = []
    for bits in itertools.product([False, True], repeat=num_positions * vocab_size):
        candidates = np.asarray(bits, dtype=np.bool_).reshape(num_positions, vocab_size)
        states.append(LatticeState(candidates))
    return states


def test_meet_join_lattice_laws_on_small_state_space() -> None:
    states = _all_states()
    top = top_like(states[0])
    bottom = bottom_like(states[0])

    for a in states:
        assert np.array_equal(meet(a, a).candidates, a.candidates)
        assert np.array_equal(join(a, a).candidates, a.candidates)
        assert np.array_equal(join(a, bottom).candidates, a.candidates)
        assert np.array_equal(meet(a, top).candidates, a.candidates)

    for a, b in itertools.product(states, repeat=2):
        assert np.array_equal(meet(a, b).candidates, meet(b, a).candidates)
        assert np.array_equal(join(a, b).candidates, join(b, a).candidates)
        assert np.array_equal(meet(a, join(a, b)).candidates, a.candidates)
        assert np.array_equal(join(a, meet(a, b)).candidates, a.candidates)

    for a, b, c in itertools.product(states, repeat=3):
        assert np.array_equal(
            meet(meet(a, b), c).candidates,
            meet(a, meet(b, c)).candidates,
        )
        assert np.array_equal(
            join(join(a, b), c).candidates,
            join(a, join(b, c)).candidates,
        )


def test_alpha_unions_concrete_solution_candidates() -> None:
    abstract = alpha(np.asarray([[0, 1], [1, 1]], dtype=np.int64), vocab_size=2)

    expected = np.asarray([[True, True], [False, True]], dtype=np.bool_)
    assert np.array_equal(abstract.candidates, expected)
    assert not bool(abstract.is_conflict())
    assert not bool(abstract.is_complete())


def test_consistent_solutions_filters_by_current_lattice_state() -> None:
    state = LatticeState(np.asarray([[True, False], [True, True]], dtype=np.bool_))
    solutions = np.asarray([[0, 0], [0, 1], [1, 0]], dtype=np.int64)

    assert np.array_equal(consistent_solutions(state, solutions), [True, True, False])


def test_pin_candidate_restricts_one_live_candidate() -> None:
    state = LatticeState(np.ones((2, 3), dtype=np.bool_))
    pinned = pin_candidate(state, position=1, value=2)

    assert np.array_equal(pinned.candidates[0], [True, True, True])
    assert np.array_equal(pinned.candidates[1], [False, False, True])
    assert pinned.singleton_mask()[1]


def test_inactive_positions_are_excluded_from_conflict_and_completion() -> None:
    candidates = np.asarray([[True, False], [False, False]], dtype=np.bool_)
    active = np.asarray([True, False], dtype=np.bool_)
    state = LatticeState(candidates, active)

    assert not bool(state.is_conflict())
    assert bool(state.is_complete())
    assert np.array_equal(state.as_solution_indices(), [0, -1])


def test_is_leq_uses_candidate_subset_order() -> None:
    small = LatticeState(np.asarray([[True, False], [False, True]], dtype=np.bool_))
    large = LatticeState(np.asarray([[True, True], [False, True]], dtype=np.bool_))

    assert bool(is_leq(small, large))
    assert not bool(is_leq(large, small))


def test_eliminate_candidates_can_create_conflict_or_preserve_state() -> None:
    state = LatticeState(np.asarray([[True, True], [False, True]], dtype=np.bool_))
    keep = np.asarray([[False, False], [False, True]], dtype=np.bool_)

    conflicted = eliminate_candidates(state, keep)
    assert bool(conflicted.is_conflict())

    preserved = eliminate_candidates(state, keep, preserve_at_least_one=True)
    assert not bool(preserved.is_conflict())
    assert np.array_equal(preserved.candidates[0], state.candidates[0])


def test_compatible_shapes_are_enforced() -> None:
    left = LatticeState(np.ones((2, 2), dtype=np.bool_))
    right = LatticeState(np.ones((3, 2), dtype=np.bool_))

    with pytest.raises(ValueError, match="candidate shapes differ"):
        meet(left, right)
