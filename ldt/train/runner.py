from __future__ import annotations

from dataclasses import dataclass

import torch

from ldt.solve import LatticeStepConfig, OnPolicyTrainingPool
from ldt.train.loss import LDTLossConfig
from ldt.train.step import ExtraFeaturesFn, TrainingStepOutput, run_training_step


@dataclass(frozen=True)
class TrainingLoopConfig:
    steps: int
    batch_size: int
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")


@dataclass(frozen=True)
class TrainingRunStats:
    steps: int
    last_loss: float | None
    retained: int
    refilled: int
    solved: int
    conflicted: int


def run_training_steps(
    model: torch.nn.Module,
    pool: OnPolicyTrainingPool,
    optimizer: torch.optim.Optimizer,
    *,
    config: TrainingLoopConfig,
    loss_config: LDTLossConfig | None = None,
    step_config: LatticeStepConfig | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: str | torch.device = "cpu",
    extra_features_fn: ExtraFeaturesFn | None = None,
) -> TrainingRunStats:
    """Run a bounded training loop when a caller explicitly invokes it."""

    last_output: TrainingStepOutput | None = None
    retained = 0
    refilled = 0
    solved = 0
    conflicted = 0
    for _ in range(config.steps):
        last_output = run_training_step(
            model,
            pool,
            batch_size=config.batch_size,
            optimizer=optimizer,
            loss_config=loss_config,
            step_config=step_config,
            device=device,
            extra_features_fn=extra_features_fn,
            grad_clip=config.grad_clip,
        )
        if scheduler is not None:
            scheduler.step()
        retained += last_output.pool_stats.retained
        refilled += last_output.pool_stats.refilled
        solved += last_output.pool_stats.solved
        conflicted += last_output.pool_stats.conflicted

    return TrainingRunStats(
        steps=config.steps,
        last_loss=None if last_output is None else float(last_output.loss.total.detach().cpu()),
        retained=retained,
        refilled=refilled,
        solved=solved,
        conflicted=conflicted,
    )
