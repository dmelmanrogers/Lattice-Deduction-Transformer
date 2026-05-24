from __future__ import annotations

import numpy as np
import torch

from ldt.lattice import LatticeState
from ldt.solve import LatticeStepConfig, lattice_projection_step


def _logit(probability: float) -> float:
    return float(np.log(probability / (1.0 - probability)))


def test_threshold_elimination_removes_low_confidence_live_candidates_only() -> None:
    state = LatticeState(np.asarray([[True, True, False], [True, True, True]], dtype=np.bool_))
    logits = np.asarray(
        [
            [_logit(0.9), _logit(0.05), _logit(0.01)],
            [_logit(0.2), _logit(0.8), _logit(0.7)],
        ]
    )

    result = lattice_projection_step(
        state,
        logits,
        0.0,
        config=LatticeStepConfig(theta_elim=0.1, enable_branching=False),
    )

    expected = np.asarray([[True, False, False], [True, True, True]], dtype=np.bool_)
    assert np.array_equal(result.state.candidates, expected)
    assert int(result.eliminated_candidates) == 1
    assert not bool(result.conflict)
    assert not bool(result.solved)
    assert not bool(result.branched)


def test_empty_cell_after_elimination_is_conflict_and_does_not_branch() -> None:
    state = LatticeState(np.asarray([[True, True], [True, False]], dtype=np.bool_))
    logits = np.asarray([[_logit(0.01), _logit(0.02)], [_logit(0.8), _logit(0.9)]])

    result = lattice_projection_step(
        state,
        logits,
        0.0,
        config=LatticeStepConfig(theta_elim=0.1),
    )

    assert bool(result.conflict)
    assert not bool(result.solved)
    assert not bool(result.branched)
    assert np.array_equal(result.state.candidates[0], [False, False])


def test_cls_threshold_can_mark_conflict_without_empty_cells() -> None:
    state = LatticeState(np.ones((2, 2), dtype=np.bool_))
    logits = np.full((2, 2), _logit(0.9))

    result = lattice_projection_step(
        state,
        logits,
        _logit(0.95),
        config=LatticeStepConfig(theta_cls=0.6),
    )

    assert bool(result.conflict)
    assert not bool(result.branched)
    assert result.cls_probability.shape == ()
    assert float(result.cls_probability) > 0.6


def test_complete_singleton_state_is_solved_and_does_not_branch() -> None:
    state = LatticeState(np.ones((2, 2), dtype=np.bool_))
    logits = np.asarray([[_logit(0.95), _logit(0.01)], [_logit(0.01), _logit(0.95)]])

    result = lattice_projection_step(
        state,
        logits,
        _logit(0.1),
        config=LatticeStepConfig(theta_elim=0.1),
    )

    assert np.array_equal(result.state.candidates, [[True, False], [False, True]])
    assert bool(result.solved)
    assert not bool(result.conflict)
    assert not bool(result.branched)


def test_branching_pins_one_uniformly_selected_multicandidate_position() -> None:
    state = LatticeState(
        np.asarray(
            [
                [True, False, False],
                [True, True, False],
                [False, True, True],
            ],
            dtype=np.bool_,
        )
    )
    logits = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
        ]
    )

    result = lattice_projection_step(
        state,
        logits,
        _logit(0.1),
        config=LatticeStepConfig(theta_elim=0.1, tau_decide=0.25),
        rng=np.random.default_rng(2),
    )

    assert bool(result.branched)
    assert int(result.branch_position) in {1, 2}
    assert result.state.candidate_counts()[int(result.branch_position)] == 1
    assert result.state.candidates.sum() == state.candidates.sum() - 1


def test_branch_sampling_uses_softmax_over_alive_candidates_only() -> None:
    state = LatticeState(np.asarray([[True, True, False]], dtype=np.bool_))
    logits = np.asarray([[0.0, 10.0, 1000.0]])

    result = lattice_projection_step(
        state,
        logits,
        _logit(0.1),
        config=LatticeStepConfig(theta_elim=0.1, tau_decide=0.1),
        rng=np.random.default_rng(0),
    )

    assert bool(result.branched)
    assert int(result.branch_position) == 0
    assert int(result.branch_value) == 1
    assert np.array_equal(result.state.candidates[0], [False, True, False])


def test_training_solutions_override_cls_conflict_and_verify_solved_state() -> None:
    state = LatticeState(np.ones((2, 2), dtype=np.bool_))
    logits = np.asarray([[_logit(0.95), _logit(0.01)], [_logit(0.01), _logit(0.95)]])
    solutions = np.asarray([[0, 1]], dtype=np.int64)

    result = lattice_projection_step(
        state,
        logits,
        _logit(0.99),
        config=LatticeStepConfig(theta_elim=0.1),
        solutions=solutions,
    )

    assert not bool(result.conflict)
    assert bool(result.solved)
    assert not bool(result.branched)


def test_training_solutions_mark_bad_elimination_as_conflict() -> None:
    state = LatticeState(np.ones((2, 2), dtype=np.bool_))
    logits = np.asarray([[_logit(0.01), _logit(0.95)], [_logit(0.01), _logit(0.95)]])
    solutions = np.asarray([[0, 1]], dtype=np.int64)

    result = lattice_projection_step(
        state,
        logits,
        _logit(0.1),
        config=LatticeStepConfig(theta_elim=0.1),
        solutions=solutions,
    )

    assert bool(result.conflict)
    assert not bool(result.solved)
    assert not bool(result.branched)


def test_batched_projection_handles_rows_independently_and_accepts_torch_logits() -> None:
    state = LatticeState(np.ones((2, 2, 2), dtype=np.bool_))
    logits = torch.tensor(
        [
            [[_logit(0.95), _logit(0.01)], [_logit(0.01), _logit(0.95)]],
            [[_logit(0.95), _logit(0.95)], [_logit(0.01), _logit(0.95)]],
        ],
        dtype=torch.float32,
    )
    cls_logits = torch.tensor([_logit(0.1), _logit(0.99)], dtype=torch.float32)

    result = lattice_projection_step(
        state,
        logits,
        cls_logits,
        config=LatticeStepConfig(theta_elim=0.1, theta_cls=0.6),
        rng=np.random.default_rng(0),
    )

    assert np.array_equal(result.solved, [True, False])
    assert np.array_equal(result.conflict, [False, True])
    assert np.array_equal(result.branched, [False, False])
    assert result.state.candidates.shape == (2, 2, 2)
