from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from ldt.domains.base import PuzzleInstance
from ldt.lattice import LatticeState
from ldt.solve import LatticeStepConfig, OnPolicyTrainingPool, PoolConfig
from ldt.train import run_training_step


class TinyTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2, 2))
        self.cls = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        candidates: Tensor,
        extra_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del candidates, extra_features
        batch = 1
        candidate_logits = self.logits.expand(batch, -1, -1).unsqueeze(0)
        cls_logits = self.cls.expand(batch).unsqueeze(0)
        return candidate_logits, cls_logits


def test_run_training_step_can_dry_run_without_optimizer() -> None:
    instance = PuzzleInstance(
        puzzle_id="one",
        initial_state=LatticeState(np.ones((2, 2), dtype=np.bool_)),
        solutions=np.asarray([[0, 1]], dtype=np.int64),
        raw=None,
    )
    pool = OnPolicyTrainingPool([instance], PoolConfig(pool_size=1), rng=np.random.default_rng(0))
    model = TinyTrainModel()

    output = run_training_step(
        model,
        pool,
        batch_size=1,
        step_config=LatticeStepConfig(enable_branching=False),
    )

    assert output.loss.total.item() > 0
    assert output.pool_stats.sampled == 1
    assert pool.entries[0].age in {0, 1}
