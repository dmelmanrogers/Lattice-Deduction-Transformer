from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from ldt.domains.base import PuzzleInstance
from ldt.lattice import LatticeState
from ldt.solve import LatticeStepConfig, OnPolicyTrainingPool, PoolConfig
from ldt.train import (
    AdamWConfig,
    TrainingLoopConfig,
    create_adamw,
    create_warmup_cosine_scheduler,
    run_training_steps,
)


class TinyLoopModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.cls = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        candidates: Tensor,
        extra_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del extra_features
        return self.proj(candidates).unsqueeze(0), self.cls.expand(candidates.shape[0]).unsqueeze(0)


def test_training_runner_executes_bounded_loop_when_called() -> None:
    instance = PuzzleInstance(
        puzzle_id="one",
        initial_state=LatticeState(np.ones((2, 2), dtype=np.bool_)),
        solutions=np.asarray([[0, 1]], dtype=np.int64),
        raw=None,
    )
    pool = OnPolicyTrainingPool([instance], PoolConfig(pool_size=1), rng=np.random.default_rng(0))
    model = TinyLoopModel()
    optimizer = create_adamw(model, AdamWConfig(lr=1e-3, weight_decay=0.0))
    scheduler = create_warmup_cosine_scheduler(optimizer, total_steps=2)

    stats = run_training_steps(
        model,
        pool,
        optimizer,
        config=TrainingLoopConfig(steps=2, batch_size=1),
        step_config=LatticeStepConfig(enable_branching=False),
        scheduler=scheduler,
    )

    assert stats.steps == 2
    assert stats.last_loss is not None
    assert stats.retained + stats.refilled == 2
