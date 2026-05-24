from __future__ import annotations

import numpy as np
import torch

from ldt.lattice import LatticeState
from ldt.train import LDTLossConfig, compute_loss, compute_targets


def test_compute_targets_uses_alpha_over_surviving_solutions() -> None:
    state = LatticeState(np.asarray([[True, True], [False, True]], dtype=np.bool_))
    solutions = np.asarray([[0, 1], [1, 1]], dtype=np.int64)

    targets = compute_targets(state, solutions)

    assert torch.equal(targets.candidate_target.cpu(), torch.tensor([[1.0, 1.0], [0.0, 1.0]]))
    assert not bool(targets.no_survivors)
    assert torch.equal(targets.cls_target.cpu(), torch.tensor(0.0))


def test_compute_targets_uses_last_nonempty_fallback_when_no_solution_survives() -> None:
    state = LatticeState(np.asarray([[False, True], [True, False]], dtype=np.bool_))
    previous = LatticeState(np.asarray([[True, False], [True, True]], dtype=np.bool_))
    solutions = np.asarray([[0, 1]], dtype=np.int64)

    targets = compute_targets(state, solutions, last_nonempty_target=previous)

    expected = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    assert torch.equal(targets.candidate_target.cpu(), expected)
    assert bool(targets.no_survivors)
    assert torch.equal(targets.cls_target.cpu(), torch.tensor(1.0))


def test_compute_loss_supervises_all_internal_iterations() -> None:
    targets = compute_targets(
        LatticeState(np.asarray([[True, False], [False, True]], dtype=np.bool_)),
        np.asarray([[0, 1]], dtype=np.int64),
    )
    candidate_logits = torch.zeros(3, 1, 2, 2, requires_grad=True)
    cls_logits = torch.zeros(3, 1, requires_grad=True)

    loss = compute_loss(
        candidate_logits,
        cls_logits,
        targets,
        LDTLossConfig(lambda_cls=0.1, lambda_ce=0.2),
    )
    loss.total.backward()

    assert loss.total.item() > 0
    assert candidate_logits.grad is not None
    assert cls_logits.grad is not None
